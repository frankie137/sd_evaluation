#!/usr/bin/env python3
"""Benchmark the PEAV Sortformer model over the whole diarization suite, windowed.

The PEAV model is trained on 45 s windows with NO cross-window speaker
association, so (exactly like sd_evaluation/peav_windowed_eval.py) we run it
independently per 45 s window and score with diar_eval in per-window mode.

Scoring settings are taken verbatim from sd_evaluation/benchmark.py (the
Sortformer run): per-dataset collar, EXEMPT datasets scored as one row,
AMI/Alimeeting split per leaf collection, and every other dataset split into
<=4 / >4 speaker subsets.

Two phases (both resumable):
  infer  -- per file: 45 s windowed inference (batched) -> cache a windowed RTTM
            under PRED_ROOT/<collection>/<key>.rttm whose speaker labels are
            window-prefixed (w000s0, w000s1, w001s0, ...) so each window stays
            locally consistent, plus a <key>.json sidecar holding the window
            tiling. Files already cached are skipped.
  score  -- group per the benchmark.py rules, score each file with
            diar_eval.evaluate(ref, pred, windows=...), pool raw
            miss/FA/confusion/reference seconds across files+windows, emit table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ALM_ROOT = Path("/workspace/ALM")
SD_EVAL = Path("/workspace/sd_evaluation")
for p in (ALM_ROOT, SD_EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

import benchmark as B  # cfg_for, dataset_of, n_speakers, EXEMPT, SPLIT_BY_LEAF, SUBSET_ORDER
import diar_eval
from peav_windowed_eval import load_audio, load_peav_model, probs_to_segments
from tasks.dia.metrics import get_der_frame_length_sec

ROOT = Path("/workspace/sd_full_benchmark")
PRED_ROOT = Path("/workspace/sd_evaluation/out/peav_preds")
OUT_DIR = Path("/workspace/sd_evaluation/out")
WINDOW_SEC = 45.0
DEFAULT_BATCH = 8


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_collections():
    cols = []
    for wavs in sorted(ROOT.rglob("wavs")):
        rttms = wavs.parent / "rttms"
        if not (wavs.is_dir() and rttms.is_dir()):
            continue
        rel = wavs.parent.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        cols.append((str(rel), wavs, rttms))
    return cols


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _forward_batch(model, device, chunks, lengths):
    maxlen = max(lengths)
    bt = torch.zeros(len(chunks), maxlen, dtype=chunks[0].dtype)
    for j, c in enumerate(chunks):
        bt[j, : c.shape[0]] = c
    bt = bt.to(device)
    ln = torch.tensor(lengths, device=device, dtype=torch.long)
    with torch.no_grad(), torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
    ):
        out = model(audio_signal=bt, audio_signal_length=ln)
    return out["preds"].float().cpu(), out["lengths"].cpu()


def _segs_from_pred(pred, outlen, t0, t1, frame_len_sec):
    valid = min(int(outlen), pred.shape[0])
    wl = t1 - t0
    segs = []
    for s, e, spk in probs_to_segments(pred, valid, frame_len_sec):
        a, b = t0 + s, t0 + min(e, wl)
        if b > a:
            segs.append((round(a, 3), round(b, 3), spk))
    return segs


def infer_file(model, device, frame_len_sec, wav_path, sr, max_batch):
    wav, _ = load_audio(wav_path, sr)
    total = wav.shape[0] / sr
    windows = diar_eval.make_windows(total, WINDOW_SEC)
    results = [None] * len(windows)

    # IMPORTANT: the model corrupts outputs when a batch mixes sequence lengths
    # (zero-pad leakage contaminates every item in the batch). So we ONLY batch
    # equal-length windows together. All full 45 s windows share one length and
    # batch freely; the lone final short window runs in its own (size-1) group.
    from collections import defaultdict
    spans = [(int(t0 * sr), int(t1 * sr)) for (t0, t1) in windows]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, (s0, s1) in enumerate(spans):
        by_len[s1 - s0].append(i)

    for idxs in by_len.values():
        k = 0
        batch = max_batch
        while k < len(idxs):
            grp = idxs[k: k + batch]
            chunks = [wav[spans[i][0]: spans[i][1]] for i in grp]
            lengths = [int(c.shape[0]) for c in chunks]
            try:
                preds, outlens = _forward_batch(model, device, chunks, lengths)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch > 1:
                    torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    continue
                raise
            for j, i in enumerate(grp):
                results[i] = _segs_from_pred(
                    preds[j], int(outlens[j].item()),
                    windows[i][0], windows[i][1], frame_len_sec)
            k += len(grp)
    return total, windows, results


def write_windowed_rttm(windows, per_window_segs, uri, out_rttm, out_json, total):
    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rttm, "w") as f:
        for i, segs in enumerate(per_window_segs):
            for a, b, spk in segs:
                f.write(f"SPEAKER {uri} 1 {a:.3f} {b - a:.3f} <NA> <NA> "
                        f"w{i:03d}s{spk} <NA> <NA>\n")
    out_json.write_text(json.dumps(
        {"uri": uri, "duration": round(total, 3),
         "windows": [[round(t0, 3), round(t1, 3)] for t0, t1 in windows],
         "n_windows": len(windows)}))


def infer_all(only_datasets=None, limit_files=None, max_batch=DEFAULT_BATCH):
    import soundfile as sf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_peav_model(
        "/workspace/peaf_conformer_40M_head/config.yaml",
        "/workspace/peaf_conformer_40M_head/checkpoints/step=25000-val_der=0.112199.pt",
        device)
    sr = int(cfg.model.sample_rate)
    frame_len_sec = get_der_frame_length_sec(cfg)
    print(f"[infer] device={device} sr={sr} frame_len={frame_len_sec}s "
          f"window={WINDOW_SEC}s batch={max_batch}", flush=True)

    total_new = 0
    for name, wavs, rttms in find_collections():
        ds = B.dataset_of(name)
        if only_datasets and ds not in only_datasets:
            continue
        pred_dir = PRED_ROOT / name
        # only files that have a reference rttm
        wav_files = [w for w in sorted(wavs.glob("*.wav"))
                     if (rttms / (w.stem + ".rttm")).exists()]
        if limit_files:
            wav_files = wav_files[:limit_files]
        todo = [w for w in wav_files if not (pred_dir / (w.stem + ".json")).exists()]
        if not todo:
            print(f"[infer] {name}: all {len(wav_files)} cached, skip", flush=True)
            continue
        dur = sum(sf.info(str(w)).duration for w in todo)
        print(f"[infer] {name}: {len(todo)}/{len(wav_files)} files "
              f"({dur/3600:.2f}h) ...", flush=True)
        t0 = time.time()
        for k, w in enumerate(todo, 1):
            total_sec, windows, segs = infer_file(
                model, device, frame_len_sec, str(w), sr, max_batch)
            write_windowed_rttm(windows, segs, w.stem,
                                pred_dir / (w.stem + ".rttm"),
                                pred_dir / (w.stem + ".json"), total_sec)
            total_new += 1
            if k % 25 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[infer] {name}: {k}/{len(todo)} "
                      f"({el:.0f}s, {el/k:.2f}s/file)", flush=True)
        print(f"[infer] {name}: done {len(todo)} in {time.time()-t0:.0f}s", flush=True)
    print(f"\n[infer] complete, {total_new} new files\n", flush=True)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def build_groups(only_datasets=None, limit_files=None):
    """row -> {subset: [(ref_rttm, pred_rttm, windows), ...]}."""
    groups = {}
    for name, wavs, rttms in find_collections():
        ds = B.dataset_of(name)
        if only_datasets and ds not in only_datasets:
            continue
        row = name if ds in B.SPLIT_BY_LEAF else ds
        pred_dir = PRED_ROOT / name
        rttm_files = sorted(rttms.glob("*.rttm"))
        if limit_files:
            keys = {w.stem for w in sorted(wavs.glob("*.wav"))[:limit_files]}
            rttm_files = [r for r in rttm_files if r.stem in keys]
        for ref in rttm_files:
            pred = pred_dir / (ref.stem + ".rttm")
            side = pred_dir / (ref.stem + ".json")
            if not side.exists():
                continue
            windows = json.loads(side.read_text())["windows"]
            if ds in B.EXEMPT:
                subset = "all"
            else:
                subset = "<=4 spk" if B.n_speakers(ref) <= 4 else ">4 spk"
            groups.setdefault(row, {}).setdefault(subset, []).append((ref, pred, windows))
    return groups


def score_group(pairs, collar, skip_overlap):
    tot_miss = tot_fa = tot_conf = tot_ref = 0.0
    n_win = 0
    for ref, pred, windows in pairs:
        res = diar_eval.evaluate(str(ref), str(pred), windows=windows,
                                 collar=collar, skip_overlap=skip_overlap)
        for wr in res.windows:
            tot_miss += wr.miss
            tot_fa += wr.false_alarm
            tot_conf += wr.confusion
            tot_ref += wr.reference
            n_win += 1
    err = tot_miss + tot_fa + tot_conf
    der = (err / tot_ref * 100) if tot_ref > 0 else 0.0
    pct = lambda x: (x / tot_ref * 100) if tot_ref > 0 else 0.0
    return dict(der=der, miss=pct(tot_miss), fa=pct(tot_fa), conf=pct(tot_conf),
                ref_s=tot_ref, n_windows=n_win)


def score_all(only_datasets=None, limit_files=None):
    groups = build_groups(only_datasets, limit_files)
    rows = []
    for row in sorted(groups):
        ds = B.dataset_of(row)
        c = B.cfg_for(ds)
        for subset in sorted(groups[row], key=lambda s: B.SUBSET_ORDER.get(s, 9)):
            pairs = groups[row][subset]
            r = score_group(pairs, c["collar"], c["skip_overlap"])
            rows.append({"dataset": row, "subset": subset, "n_files": len(pairs),
                         "collar": c["collar"], "skip_overlap": c["skip_overlap"], **r})
            print(f"  {row:16s} {subset:8s} n={len(pairs):4d}  DER={r['der']:6.2f}  "
                  f"miss={r['miss']:6.2f}  fa={r['fa']:6.2f}  conf={r['conf']:6.2f}  "
                  f"(collar={c['collar']})", flush=True)
    return rows


def render_table(rows):
    h = ("| Dataset | Subset | #files | collar | DER % | Miss % | FA % | Conf % |\n"
         "|---|---|--:|--:|--:|--:|--:|--:|\n")
    body = "".join(
        f"| {r['dataset']} | {r['subset']} | {r['n_files']} | {r['collar']} | "
        f"{r['der']:.2f} | {r['miss']:.2f} | {r['fa']:.2f} | {r['conf']:.2f} |\n"
        for r in rows)
    return h + body


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "infer", "score"], default="all")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset names")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--table", default=str(OUT_DIR / "peav_benchmark_table.md"))
    ap.add_argument("--json", default=str(OUT_DIR / "peav_benchmark.json"))
    args = ap.parse_args()
    only = set(args.datasets.split(",")) if args.datasets else None

    if args.stage in ("all", "infer"):
        infer_all(only_datasets=only, limit_files=args.limit_files, max_batch=args.batch)
    if args.stage in ("all", "score"):
        print("\n=== scoring (PEAV windowed, Sortformer settings) ===", flush=True)
        rows = score_all(only_datasets=only, limit_files=args.limit_files)
        table = render_table(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        Path(args.table).write_text(table)
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print("\n" + table, flush=True)
        print(f"saved -> {args.table} , {args.json}", flush=True)


if __name__ == "__main__":
    main()
