#!/usr/bin/env python3
"""Benchmark streaming Sortformer (4spk v2.1) over the whole diarization suite.

Two phases (both resumable; pass --score-only to skip inference):

  1. infer  -- run the model on every wav, cache the prediction as an RTTM under
               out/preds/<collection>/<key>.rttm (files already cached are skipped).
  2. score  -- pool DER per dataset with pyannote.metrics (via diar_eval), split
               into <=4 / >4 speaker subsets where required, and emit a table.

Per-dataset scoring config (collar / skip_overlap) lives in DATASET_CFG below and
is easy to edit. Datasets in EXEMPT are scored as a single row (no speaker split).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from diar_eval import evaluate_collection, load_rttm

ROOT = Path("/workspace/sd_full_benchmark")
PRED_ROOT = Path("/workspace/sd_evaluation/out/preds")
OUT_DIR = Path("/workspace/sd_evaluation/out")
MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
CHUNK_FILES = 64   # diarize this many files per call, then write RTTMs (resumable)
# Post-processing (onset/offset/padding/min-duration) required to reproduce the
# DER reported on the NeMo model card; by default diarize() only binarizes.
POSTPROC_YAML = "/workspace/sd_evaluation/sortformer_postprocess.yaml"

# Per-dataset scoring parameters. Default: collar=0.25, skip_overlap=False.
# AMI / Alimeeting / NOTSOFAR use collar=0. `batch` is the inference batch size.
# collar is per-side (md-eval convention, ±collar around each reference
# boundary); diar_eval doubles it for pyannote, whose param is the total window.
DEFAULT_CFG = dict(collar=0.25, skip_overlap=False, batch=1)
DATASET_CFG = {
    "AISHELL-4":        dict(collar=0.25, skip_overlap=False, batch=1),
    # Forced-align AMI annotations share AMI's scoring convention (collar=0).
    "AMI_forced_align": dict(collar=0.0,  skip_overlap=False, batch=1),
    "AVA-AVD":          dict(collar=0.25, skip_overlap=False, batch=2),
    "Alimeeting":       dict(collar=0.0,  skip_overlap=False, batch=1),
    "MSDWild":          dict(collar=0.25, skip_overlap=False, batch=4),
    "MagicData-RAMC":   dict(collar=0.25, skip_overlap=False, batch=1),
    "NOTSOFAR":         dict(collar=0.0,  skip_overlap=False, batch=2),
    "VoxConverse":      dict(collar=0.25, skip_overlap=False, batch=1),
}
# Datasets scored as one row (no <=4 / >4 split).
EXEMPT = {"Alimeeting", "AMI_forced_align", "MagicData-RAMC"}
# Datasets reported per leaf collection (separate rows) instead of pooled.
SPLIT_BY_LEAF = {"AMI_forced_align", "Alimeeting"}
# Datasets present under ROOT but excluded from the whole pipeline.
# AMI_v1.6.2 is the older, less accurate annotation; AMI_forced_align is the
# annotation of record for AMI audio.
EXCLUDED_DATASETS = {"AMI_v1.6.2"}


def cfg_for(dataset):
    return DATASET_CFG.get(dataset, DEFAULT_CFG)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_collections():
    """Every (name, wavs_dir, rttms_dir) where name is the path under ROOT.

    Hidden directories (e.g. .cache/huggingface) and EXCLUDED_DATASETS are
    skipped.
    """
    cols = []
    for wavs in sorted(ROOT.rglob("wavs")):
        rttms = wavs.parent / "rttms"
        if not (wavs.is_dir() and rttms.is_dir()):
            continue
        rel = wavs.parent.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.parts[0] in EXCLUDED_DATASETS:
            continue
        cols.append((str(rel), wavs, rttms))
    return cols


def dataset_of(collection_name):
    return collection_name.split("/")[0]


def n_speakers(rttm_path):
    spk = set()
    for line in open(rttm_path):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            spk.add(p[7])
    return len(spk)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def load_model():
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(MODEL)
    m.eval()
    m = m.to("cuda" if torch.cuda.is_available() else "cpu")
    sm = m.sortformer_modules           # high-latency preset (best accuracy)
    sm.chunk_len = 340
    sm.chunk_right_context = 40
    sm.fifo_len = 40
    sm.spkcache_update_period = 300
    sm.spkcache_len = 188
    sm._check_streaming_parameters()
    return m


def write_rttm(seg_lines, uri, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in seg_lines:
            p = s.split()
            start, end, spk = float(p[0]), float(p[1]), p[2]
            f.write(f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} "
                    f"<NA> <NA> {spk} <NA> <NA>\n")


def infer_all(limit_files=None, only_datasets=None, postprocessing_yaml=None):
    import soundfile as sf
    model = load_model()
    print(f"[infer] model loaded; postprocessing={'on' if postprocessing_yaml else 'off'}\n",
          flush=True)
    total_new = 0
    for name, wavs, _ in find_collections():
        ds = dataset_of(name)
        if only_datasets and ds not in only_datasets:
            continue
        pred_dir = PRED_ROOT / name
        wav_files = sorted(wavs.glob("*.wav"))
        if limit_files:
            wav_files = wav_files[:limit_files]
        todo = [w for w in wav_files if not (pred_dir / (w.stem + ".rttm")).exists()]
        if not todo:
            print(f"[infer] {name}: all {len(wav_files)} cached, skip", flush=True)
            continue
        dur = sum(sf.info(str(w)).duration for w in todo)
        print(f"[infer] {name}: {len(todo)} files ({dur/3600:.2f}h) "
              f"bs=1 postproc={'on' if postprocessing_yaml else 'off'} ...",
              flush=True)
        t0 = time.time()
        # Process in chunks and write RTTMs after each chunk, so an interruption
        # loses at most one chunk. batch_size=1 + post-processing reproduces the
        # NeMo model-card DER (per the NeMo diarization configs doc).
        for i in range(0, len(todo), CHUNK_FILES):
            sub = todo[i:i + CHUNK_FILES]
            preds = model.diarize(audio=[str(w) for w in sub],
                                  batch_size=1, verbose=False,
                                  postprocessing_yaml=postprocessing_yaml)
            for w, seg_lines in zip(sub, preds):
                write_rttm(seg_lines, w.stem, pred_dir / (w.stem + ".rttm"))
            if len(todo) > CHUNK_FILES:
                print(f"[infer] {name}: {min(i + CHUNK_FILES, len(todo))}/{len(todo)} "
                      f"written ({time.time() - t0:.0f}s)", flush=True)
        dt = time.time() - t0
        total_new += len(todo)
        print(f"[infer] {name}: done {len(todo)} in {dt:.1f}s "
              f"(RTF={dt/max(dur,1e-9):.4f})", flush=True)
    print(f"\n[infer] complete, {total_new} new files\n", flush=True)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def build_groups(only_datasets=None, limit_files=None):
    """dataset -> {subset_label: [(ref_rttm, pred_rttm), ...]}."""
    groups = {}
    for name, wavs, rttms in find_collections():
        ds = dataset_of(name)
        if only_datasets and ds not in only_datasets:
            continue
        row = name if ds in SPLIT_BY_LEAF else ds   # leaf rows for AMI/Alimeeting
        pred_dir = PRED_ROOT / name
        rttm_files = sorted(rttms.glob("*.rttm"))
        if limit_files:
            # keep only those matching the (also limited) wav order
            keys = {w.stem for w in sorted(wavs.glob("*.wav"))[:limit_files]}
            rttm_files = [r for r in rttm_files if r.stem in keys]
        for ref in rttm_files:
            pred = pred_dir / (ref.stem + ".rttm")
            if not pred.exists():
                continue
            if ds in EXEMPT:
                subset = "all"
            else:
                subset = "<=4 spk" if n_speakers(ref) <= 4 else ">4 spk"
            groups.setdefault(row, {}).setdefault(subset, []).append((ref, pred))
    return groups


SUBSET_ORDER = {"all": 0, "<=4 spk": 1, ">4 spk": 2}


def score_all(only_datasets=None, limit_files=None):
    groups = build_groups(only_datasets, limit_files)
    rows = []
    for ds in sorted(groups):
        c = cfg_for(dataset_of(ds))   # config keyed by top-level dataset
        for subset in sorted(groups[ds], key=lambda s: SUBSET_ORDER.get(s, 9)):
            pairs = groups[ds][subset]
            res = evaluate_collection(pairs, collar=c["collar"],
                                      skip_overlap=c["skip_overlap"])
            rows.append({
                "dataset": ds, "subset": subset, "n_files": len(pairs),
                "collar": c["collar"], "skip_overlap": c["skip_overlap"],
                "der": res.der, "miss": res.miss_pct, "fa": res.fa_pct,
                "conf": res.conf_pct, "ref_s": res.reference,
            })
            print(f"  {ds:16s} {subset:8s} n={len(pairs):4d}  "
                  f"DER={res.der:6.2f}  miss={res.miss_pct:6.2f}  "
                  f"fa={res.fa_pct:6.2f}  conf={res.conf_pct:6.2f}  "
                  f"(collar={c['collar']})", flush=True)
    return rows


def render_table(rows):
    h = ("| Dataset | Subset | #files | collar | DER % | Miss % | FA % | Conf % |\n"
         "|---|---|--:|--:|--:|--:|--:|--:|\n")
    body = ""
    for r in rows:
        body += (f"| {r['dataset']} | {r['subset']} | {r['n_files']} | "
                 f"{r['collar']} | {r['der']:.2f} | {r['miss']:.2f} | "
                 f"{r['fa']:.2f} | {r['conf']:.2f} |\n")
    return h + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "infer", "score"], default="all")
    ap.add_argument("--score-only", action="store_true", help="alias for --stage score")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset names to limit to")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="only first N files per collection (smoke test)")
    ap.add_argument("--table", default=str(OUT_DIR / "benchmark_table.md"),
                    help="output markdown table path")
    ap.add_argument("--json", default=str(OUT_DIR / "benchmark.json"),
                    help="output json path")
    ap.add_argument("--postprocessing-yaml", default=POSTPROC_YAML,
                    help="post-processing params to reproduce model-card DER; "
                         "pass '' to disable (binarization only)")
    args = ap.parse_args()

    stage = "score" if args.score_only else args.stage
    only = set(args.datasets.split(",")) if args.datasets else None

    if stage in ("all", "infer"):
        infer_all(limit_files=args.limit_files, only_datasets=only,
                  postprocessing_yaml=(args.postprocessing_yaml or None))
    if stage in ("all", "score"):
        print("\n=== scoring ===", flush=True)
        rows = score_all(only_datasets=only, limit_files=args.limit_files)
        table = render_table(rows)
        Path(args.table).parent.mkdir(parents=True, exist_ok=True)
        Path(args.table).write_text(table)
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print("\n" + table, flush=True)
        print(f"saved -> {args.table} , {args.json}", flush=True)


if __name__ == "__main__":
    main()
