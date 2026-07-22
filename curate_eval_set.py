#!/usr/bin/env python3
"""Curate a human-review eval set from the Sortformer v1 windowed results.

Builds one eval set per subset (10 total: mic-variant collections such as
Alimeeting Far/Near and AMI IHM/SDM are independent subsets; every other
dataset is one subset). Each set targets 60 windows of 90 s, stratified into
four DER bins (<10, 10-20, 20-30, >=30 %), 15 windows per bin.

Scoring convention (intentionally different from benchmark.py): a single
UNIFIED collar=0.25, skip_overlap=False for every dataset, per-window optimal
speaker mapping (diar_eval). Only windows with >=20 s of reference speech and
<=4 reference speakers are eligible.

Stages (all resumable / re-runnable):
  score  -- rescore every v1-predicted file per window, write one JSONL row
            per window to out/curation/v1_window_scores.jsonl
  select -- stratified random sampling per dataset (fixed seed, max windows
            per source session), write out/curation/manifest.json
  export -- cut per-window wav + window-local reference/prediction RTTMs under
            out/curation/<dataset>/{wavs,rttms,preds}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

import benchmark as B
import diar_eval

OUT_DIR = Path("/workspace/sd_evaluation/out/curation")
PRED_ROOT = Path("/workspace/sd_evaluation/out/sortformer_v1_win90_preds")
SCORES_PATH = OUT_DIR / "v1_window_scores.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"

COLLAR = 0.25
SKIP_OVERLAP = False
MIN_REF_SEC = 20.0
MAX_REF_SPEAKERS = 4
BINS = [("der<10", 0.0, 10.0), ("der10-20", 10.0, 20.0),
        ("der20-30", 20.0, 30.0), ("der>=30", 30.0, float("inf"))]
PER_BIN = 15
MAX_PER_SESSION = 2
MAX_PER_SESSION_RELAXED = 3
# Datasets whose mic-variant collections (Alimeeting Far/Near, AMI IHM/SDM)
# are curated as independent subsets, each sampled on its own.
SPLIT_BY_COLLECTION = {"Alimeeting", "AMI_forced_align"}


def subset_of(w):
    """Sampling unit: the collection for split datasets, else the dataset."""
    if w["dataset"] in SPLIT_BY_COLLECTION:
        return w["collection"]
    return w["dataset"]


# --------------------------------------------------------------------------- #
# stage: score
# --------------------------------------------------------------------------- #
def score_stage():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, wavs, rttms in B.find_collections():
        ds = B.dataset_of(name)
        pred_dir = PRED_ROOT / name
        n_files = 0
        for ref in sorted(rttms.glob("*.rttm")):
            pred = pred_dir / (ref.stem + ".rttm")
            side = pred_dir / (ref.stem + ".json")
            if not side.exists():
                continue
            windows = json.loads(side.read_text())["windows"]
            res = diar_eval.evaluate(str(ref), str(pred), windows=windows,
                                     collar=COLLAR, skip_overlap=SKIP_OVERLAP)
            n_files += 1
            for i, wr in enumerate(res.windows):
                rows.append({
                    "dataset": ds, "collection": name, "key": ref.stem,
                    "win_idx": i, "t0": wr.start, "t1": wr.end,
                    "ref_sec": round(wr.reference, 3),
                    "miss": round(wr.miss, 3),
                    "fa": round(wr.false_alarm, 3),
                    "conf": round(wr.confusion, 3),
                    "der": round(wr.der, 3) if wr.der is not None else None,
                    "n_ref_speakers": wr.n_ref_speakers,
                })
        print(f"[score] {name}: {n_files} files", flush=True)
    with open(SCORES_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[score] wrote {len(rows)} windows -> {SCORES_PATH}", flush=True)


def load_scores():
    return [json.loads(l) for l in open(SCORES_PATH)]


# --------------------------------------------------------------------------- #
# stage: select
# --------------------------------------------------------------------------- #
def bin_of(der):
    for label, lo, hi in BINS:
        if lo <= der < hi:
            return label
    return None


def sample_bin(cands, rng, taken_per_session, cap, used_regions, want):
    """Greedy random draw honouring the per-session window cap.

    `used_regions` holds (key, win_idx) picked anywhere in the subset, so the
    same time region never enters twice within one subset.
    """
    picked = []
    for w in rng.sample(cands, len(cands)):
        sess = (subset_of(w), w["key"])
        if (w["key"], w["win_idx"]) in used_regions:
            continue
        if taken_per_session[sess] >= cap:
            continue
        used_regions.add((w["key"], w["win_idx"]))
        taken_per_session[sess] += 1
        picked.append(w)
        if len(picked) == want:
            break
    return picked


def select_stage(seed):
    scores = load_scores()
    eligible = defaultdict(list)
    for w in scores:
        if (w["der"] is not None and w["ref_sec"] >= MIN_REF_SEC
                and w["n_ref_speakers"] <= MAX_REF_SPEAKERS):
            eligible[subset_of(w)].append(w)

    manifest = {"collar": COLLAR, "skip_overlap": SKIP_OVERLAP,
                "min_ref_sec": MIN_REF_SEC, "max_ref_speakers": MAX_REF_SPEAKERS,
                "seed": seed, "per_bin": PER_BIN, "datasets": {}}
    for ds in sorted(eligible):
        rng = random.Random(seed)
        by_bin = defaultdict(list)
        for w in eligible[ds]:
            by_bin[bin_of(w["der"])].append(w)
        selected = {}
        used_regions = set()
        for label, _, _ in BINS:
            cands = by_bin.get(label, [])
            # Per-bin session cap: small subsets (e.g. Alimeeting Far, 20
            # sessions) cannot fill 60 windows under a subset-wide cap.
            taken = defaultdict(int)
            picked = sample_bin(cands, rng, taken, MAX_PER_SESSION,
                                used_regions, PER_BIN)
            if len(picked) < PER_BIN:
                picked += sample_bin(cands, rng, taken, MAX_PER_SESSION_RELAXED,
                                     used_regions, PER_BIN - len(picked))
            selected[label] = sorted(picked, key=lambda w: (w["key"], w["win_idx"]))
            short = PER_BIN - len(picked)
            note = f"  !! short by {short}" if short > 0 else ""
            print(f"[select] {ds:16s} {label:9s} eligible={len(cands):5d} "
                  f"picked={len(picked):2d}{note}", flush=True)
        manifest["datasets"][ds] = selected
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for d in manifest["datasets"].values() for v in d.values())
    print(f"[select] {total} windows -> {MANIFEST_PATH}", flush=True)


# --------------------------------------------------------------------------- #
# stage: export
# --------------------------------------------------------------------------- #
def window_id(w):
    return f"{w['key']}_w{w['win_idx']:04d}"


def load_rttm_segments(path):
    segs = []
    for line in open(path):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            segs.append((start, start + dur, spk))
    return segs


def clip_segments(segs, t0, t1):
    """Segments intersecting [t0, t1], clipped and shifted to window-local time."""
    out = []
    for a, b, spk in segs:
        a2, b2 = max(a, t0) - t0, min(b, t1) - t0
        if b2 - a2 > 1e-3:
            out.append((round(a2, 3), round(b2, 3), spk))
    return sorted(out)


def write_rttm(path, uri, segs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for a, b, spk in segs:
            f.write(f"SPEAKER {uri} 1 {a:.3f} {b - a:.3f} <NA> <NA> {spk} <NA> <NA>\n")


def cut_wav(wav_path, t0, t1, out_path):
    import numpy as np
    import soundfile as sf
    info = sf.info(str(wav_path))
    s0 = int(t0 * info.samplerate)
    s1 = min(int(t1 * info.samplerate), info.frames)
    with sf.SoundFile(str(wav_path)) as f:
        f.seek(s0)
        data = f.read(s1 - s0, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), data, info.samplerate)


def export_stage():
    manifest = json.loads(MANIFEST_PATH.read_text())
    ref_dirs = {name: (wavs, rttms) for name, wavs, rttms in B.find_collections()}
    n = 0
    for ds, bins in manifest["datasets"].items():
        ds_dir = OUT_DIR / ds
        for label, wins in bins.items():
            for w in wins:
                wavs, rttms = ref_dirs[w["collection"]]
                wid = window_id(w)
                t0, t1 = w["t0"], w["t1"]
                out_wav = ds_dir / "wavs" / f"{wid}.wav"
                if not out_wav.exists():
                    cut_wav(wavs / f"{w['key']}.wav", t0, t1, out_wav)
                ref_segs = clip_segments(
                    load_rttm_segments(rttms / f"{w['key']}.rttm"), t0, t1)
                write_rttm(ds_dir / "rttms" / f"{wid}.rttm", wid, ref_segs)
                pred_all = load_rttm_segments(
                    PRED_ROOT / w["collection"] / f"{w['key']}.rttm")
                prefix = f"w{w['win_idx']:03d}_"
                pred_segs = [(a, b, spk[len(prefix):])
                             for a, b, spk in clip_segments(pred_all, t0, t1)
                             if spk.startswith(prefix)]
                write_rttm(ds_dir / "preds" / f"{wid}.rttm", wid, pred_segs)
                n += 1
        print(f"[export] {ds}: done", flush=True)
    print(f"[export] {n} windows -> {OUT_DIR}/<dataset>/", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "score", "select", "export"],
                    default="all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.stage in ("all", "score"):
        score_stage()
    if args.stage in ("all", "select"):
        select_stage(args.seed)
    if args.stage in ("all", "export"):
        export_stage()


if __name__ == "__main__":
    main()
