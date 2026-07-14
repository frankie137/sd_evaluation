#!/usr/bin/env python3
"""Benchmark offline Sortformer (diar_sortformer_4spk-v1) over the suite, windowed.

https://huggingface.co/nvidia/diar_sortformer_4spk-v1

The offline 4-spk model is run independently per fixed window (default 90 s),
so speaker ids are only locally consistent inside a window — exactly the
setting peav_benchmark.py evaluates. Scoring therefore uses diar_eval in
per-window mode: each window gets its own optimal speaker mapping and raw
miss/FA/confusion/reference seconds are pooled across windows and files.

Scoring settings are taken verbatim from benchmark.py: per-dataset collar,
EXEMPT datasets scored as one row, AMI/Alimeeting split per leaf collection,
every other dataset split into <=4 / >4 speaker subsets.

Two phases (both resumable):
  infer  -- per file: slice the wav into windows, run diarize() on the chunks,
            cache a windowed RTTM under PRED_ROOT/<collection>/<key>.rttm with
            window-prefixed speaker labels (w000_speaker_0, ...) plus a
            <key>.json sidecar holding the window tiling. Cached files skipped.
  score  -- group per benchmark.py rules, score each file with
            diar_eval.evaluate(ref, pred, windows=...), pool, emit table.

Post-processing defaults to the model card's DIHARD3-dev optimized parameters
(sortformer_v1_postprocess_dh3.yaml); pass --postprocessing-yaml '' to disable
(binarization only).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

import soundfile as sf

import benchmark as B
import diar_eval

MODEL = "nvidia/diar_sortformer_4spk-v1"
OUT_DIR = Path("/workspace/sd_evaluation/out")
WINDOW_SEC = 90.0
# Sortformer normalises the waveform by the BATCH max, so predictions depend on
# batch composition; batch 1 matches the NeMo model-card inference path.
DEFAULT_BATCH = 1
POSTPROC_YAML = str(SD_EVAL / "sortformer_v1_postprocess_dh3.yaml")


def pred_root(window_sec: float) -> Path:
    return OUT_DIR / f"sortformer_v1_win{window_sec:g}_preds"


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def load_model():
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(MODEL)
    m.eval()
    return m.to("cuda" if torch.cuda.is_available() else "cpu")


def slice_windows(wav_path: Path, window_sec: float, tmp_dir: Path):
    """Cut the wav into window chunks written as temp wavs (native sr/channels).

    diarize() resamples/downmixes on load, so chunks keep the source format.
    """
    info = sf.info(str(wav_path))
    total = info.frames / info.samplerate
    windows = diar_eval.make_windows(total, window_sec)
    chunk_paths = []
    with sf.SoundFile(str(wav_path)) as f:
        for i, (t0, t1) in enumerate(windows):
            s0 = int(t0 * info.samplerate)
            s1 = min(int(t1 * info.samplerate), info.frames)
            f.seek(s0)
            data = f.read(s1 - s0, dtype="float32", always_2d=True)
            p = tmp_dir / f"{wav_path.stem}_w{i:04d}.wav"
            sf.write(str(p), data, info.samplerate)
            chunk_paths.append(p)
    return total, windows, chunk_paths


def infer_file(model, wav_path: Path, window_sec: float, batch: int,
               postprocessing_yaml, tmp_dir: Path):
    total, windows, chunks = slice_windows(wav_path, window_sec, tmp_dir)
    try:
        preds = model.diarize(audio=[str(p) for p in chunks], batch_size=batch,
                              verbose=False,
                              postprocessing_yaml=postprocessing_yaml)
    finally:
        for p in chunks:
            p.unlink(missing_ok=True)

    per_window = []
    for (t0, t1), seg_lines in zip(windows, preds):
        win_len = t1 - t0
        segs = []
        for line in seg_lines:
            p = line.split()
            a, b, spk = float(p[0]), float(p[1]), p[2]
            a, b = t0 + max(a, 0.0), t0 + min(b, win_len)
            if b > a:
                segs.append((round(a, 3), round(b, 3), spk))
        per_window.append(segs)
    return total, windows, per_window


def write_windowed_rttm(windows, per_window_segs, uri, out_rttm, out_json, total):
    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rttm, "w") as f:
        for i, segs in enumerate(per_window_segs):
            for a, b, spk in segs:
                f.write(f"SPEAKER {uri} 1 {a:.3f} {b - a:.3f} <NA> <NA> "
                        f"w{i:03d}_{spk} <NA> <NA>\n")
    out_json.write_text(json.dumps(
        {"uri": uri, "duration": round(total, 3),
         "windows": [[round(t0, 3), round(t1, 3)] for t0, t1 in windows],
         "n_windows": len(windows)}))


def infer_all(window_sec, only_datasets=None, limit_files=None,
              batch=DEFAULT_BATCH, postprocessing_yaml=None):
    root = pred_root(window_sec)
    model = load_model()
    print(f"[infer] model={MODEL} window={window_sec:g}s batch={batch} "
          f"postproc={'on' if postprocessing_yaml else 'off'}\n", flush=True)
    total_new = 0
    with tempfile.TemporaryDirectory(prefix="sfv1_chunks_") as td:
        tmp_dir = Path(td)
        for name, wavs, rttms in B.find_collections():
            ds = B.dataset_of(name)
            if only_datasets and ds not in only_datasets:
                continue
            pred_dir = root / name
            wav_files = [w for w in sorted(wavs.glob("*.wav"))
                         if (rttms / (w.stem + ".rttm")).exists()]
            if limit_files:
                wav_files = wav_files[:limit_files]
            todo = [w for w in wav_files
                    if not (pred_dir / (w.stem + ".json")).exists()]
            if not todo:
                print(f"[infer] {name}: all {len(wav_files)} cached, skip", flush=True)
                continue
            dur = sum(sf.info(str(w)).duration for w in todo)
            print(f"[infer] {name}: {len(todo)}/{len(wav_files)} files "
                  f"({dur/3600:.2f}h) ...", flush=True)
            t0 = time.time()
            for k, w in enumerate(todo, 1):
                total_sec, windows, segs = infer_file(
                    model, w, window_sec, batch, postprocessing_yaml, tmp_dir)
                write_windowed_rttm(windows, segs, w.stem,
                                    pred_dir / (w.stem + ".rttm"),
                                    pred_dir / (w.stem + ".json"), total_sec)
                total_new += 1
                if k % 25 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"[infer] {name}: {k}/{len(todo)} "
                          f"({el:.0f}s, {el/k:.2f}s/file)", flush=True)
            el = time.time() - t0
            print(f"[infer] {name}: done {len(todo)} in {el:.0f}s "
                  f"(RTF={el/max(dur,1e-9):.4f})", flush=True)
    print(f"\n[infer] complete, {total_new} new files\n", flush=True)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def build_groups(window_sec, only_datasets=None, limit_files=None):
    """row -> {subset: [(ref_rttm, pred_rttm, windows), ...]}."""
    root = pred_root(window_sec)
    groups = {}
    for name, wavs, rttms in B.find_collections():
        ds = B.dataset_of(name)
        if only_datasets and ds not in only_datasets:
            continue
        row = name if ds in B.SPLIT_BY_LEAF else ds
        pred_dir = root / name
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
            groups.setdefault(row, {}).setdefault(subset, []).append(
                (ref, pred, windows))
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


def score_all(window_sec, only_datasets=None, limit_files=None):
    groups = build_groups(window_sec, only_datasets, limit_files)
    rows = []
    for row in sorted(groups):
        ds = B.dataset_of(row)
        c = B.cfg_for(ds)
        for subset in sorted(groups[row], key=lambda s: B.SUBSET_ORDER.get(s, 9)):
            pairs = groups[row][subset]
            r = score_group(pairs, c["collar"], c["skip_overlap"])
            rows.append({"dataset": row, "subset": subset, "n_files": len(pairs),
                         "collar": c["collar"], "skip_overlap": c["skip_overlap"],
                         **r})
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
    ap.add_argument("--window", type=float, default=WINDOW_SEC,
                    help="window length in seconds")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset names")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--postprocessing-yaml", default=POSTPROC_YAML,
                    help="model-card DH3-dev post-processing; pass '' to disable")
    ap.add_argument("--table", default=None, help="output markdown table path")
    ap.add_argument("--json", default=None, help="output json path")
    args = ap.parse_args()
    only = set(args.datasets.split(",")) if args.datasets else None
    table_path = Path(args.table or OUT_DIR /
                      f"sortformer_v1_win{args.window:g}_table.md")
    json_path = Path(args.json or OUT_DIR /
                     f"sortformer_v1_win{args.window:g}.json")

    if args.stage in ("all", "infer"):
        infer_all(args.window, only_datasets=only, limit_files=args.limit_files,
                  batch=args.batch,
                  postprocessing_yaml=(args.postprocessing_yaml or None))
    if args.stage in ("all", "score"):
        print(f"\n=== scoring (offline Sortformer v1, {args.window:g}s windows) ===",
              flush=True)
        rows = score_all(args.window, only_datasets=only,
                         limit_files=args.limit_files)
        table = render_table(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table_path.write_text(table)
        json_path.write_text(json.dumps(rows, indent=2))
        print("\n" + table, flush=True)
        print(f"saved -> {table_path} , {json_path}", flush=True)


if __name__ == "__main__":
    main()
