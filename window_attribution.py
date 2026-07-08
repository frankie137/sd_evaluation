#!/usr/bin/env python3
"""Fine-grained error attribution for the diarization benchmark.

Cuts every session into fixed windows (default 90 s), computes DER components
(Miss / FA / Confusion) per window under the session-level optimal speaker
mapping, and tags each window with attributes for bucketed aggregation:

  from the reference annotation:
    active speaker count, overlap ratio, mean turn duration, short (<1 s)
    turns, absolute/relative position in the session, speaker re-entry after
    long silence, first appearances of new speakers;
  from the audio:
    per-window SNR estimate (speech vs. non-speech frame energy, guided by the
    reference), session-level blind reverberation proxy (energy decay time
    after turn offsets).

Design notes
------------
* The speaker mapping is optimised ONCE per session (like benchmark.py does per
  file) and then kept FIXED across windows, so slot drift / label flicker shows
  up as Confusion in later windows instead of being re-absorbed by per-window
  remapping. A per-window locally-optimal Confusion is also stored; the excess
  (conf - conf_local) measures mapping drift.
* Scoring collar / skip_overlap follow DATASET_CFG in benchmark.py, so pooled
  numbers stay comparable with the existing benchmark table (up to small
  window-boundary collar effects).
* Per-speaker stats: detection latency and attribution latency after first
  appearance, label flicker rate and purity on solo speech.

Outputs one JSON per session under OUT_ROOT/records/<collection>/<key>.json
(existing files are skipped, so the stage is resumable). Aggregation and the
HTML report are done by attribution_report.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

from benchmark import SPLIT_BY_LEAF, cfg_for, dataset_of, find_collections
from diar_eval import load_rttm

PRED_ROOT = Path("/workspace/sd_evaluation/out/preds")
OUT_ROOT = Path("/workspace/sd_evaluation/out/attribution")
# Prediction dirs were produced before the AMI dataset was split into
# annotation variants; both variants score the same predictions.
PRED_ALIAS = {"AMI_v1.6.2": "AMI", "AMI_forced_align": "AMI"}

WINDOW_SEC = 90.0
FRAME = 0.01            # activity-mask resolution (s)
E_HOP, E_WIN = 0.016, 0.032   # energy framing (s)
TURN_MERGE_GAP = 0.5    # same-speaker segments closer than this form one turn
SHORT_TURN_SEC = 1.0
REENTRY_GAP_SEC = 30.0


def pred_dir_for(collection: str) -> Path:
    p = PRED_ROOT / collection
    if p.is_dir():
        return p
    parts = collection.split("/")
    if parts[0] in PRED_ALIAS:
        alias = PRED_ROOT / "/".join([PRED_ALIAS[parts[0]]] + parts[1:])
        if alias.is_dir():
            return alias
    return p


# --------------------------------------------------------------------------- #
# Reference-side helpers
# --------------------------------------------------------------------------- #
def merge_turns(segments, gap=TURN_MERGE_GAP):
    """Merge sorted (start, end) segments of one speaker into turns."""
    turns = []
    for s, e in sorted(segments):
        if turns and s - turns[-1][1] <= gap:
            turns[-1][1] = max(turns[-1][1], e)
        else:
            turns.append([s, e])
    return [(s, e) for s, e in turns]


def activity_masks(ann: Annotation, n_frames: int):
    """Per-label boolean frame masks at FRAME resolution."""
    masks = {}
    for seg, _, label in ann.itertracks(yield_label=True):
        i0 = max(0, int(seg.start / FRAME))
        i1 = min(n_frames, int(math.ceil(seg.end / FRAME)))
        if i1 <= i0:
            continue
        m = masks.get(label)
        if m is None:
            m = masks[label] = np.zeros(n_frames, dtype=bool)
        m[i0:i1] = True
    return masks


# --------------------------------------------------------------------------- #
# Audio-side helpers
# --------------------------------------------------------------------------- #
def frame_energy(wav_path: Path, duration: float):
    """Mean power per energy frame (hop E_HOP, win E_WIN), first channel only."""
    info = sf.info(str(wav_path))
    sr = info.samplerate
    hop, win = int(sr * E_HOP), int(sr * E_WIN)
    n_frames = max(1, int((info.frames - win) // hop) + 1)
    energy = np.zeros(n_frames, dtype=np.float32)
    block_frames = int(60.0 / E_HOP)                 # ~60 s of frames per block
    with sf.SoundFile(str(wav_path)) as f:
        for b0 in range(0, n_frames, block_frames):
            b1 = min(b0 + block_frames, n_frames)
            s0 = b0 * hop
            s1 = min((b1 - 1) * hop + win, info.frames)
            f.seek(s0)
            x = f.read(s1 - s0, dtype="float32", always_2d=True)[:, 0]
            n_full = (len(x) - win) // hop + 1
            if n_full <= 0:
                break
            idx = np.arange(win)[None, :] + hop * np.arange(n_full)[:, None]
            energy[b0:b0 + n_full] = np.mean(x[idx] ** 2, axis=1)
    return energy


def estimate_reverb_t60(energy_db, decay_win=0.30, min_drop=10.0):
    """Blind reverb proxy from free decays in the energy envelope.

    Scans the dB envelope for near-monotonic drops of >= min_drop dB within
    decay_win seconds starting from a loud level, fits a slope to each, and
    takes the 20th percentile of the implied decay-to-60dB times (fastest
    credible decays approximate free decay; slower ones are contaminated by
    ongoing speech). Annotation-independent, good enough for bucketing.
    """
    n = int(decay_win / E_HOP)
    if len(energy_db) < n + 1:
        return None
    w = np.lib.stride_tricks.sliding_window_view(energy_db, n)
    drop = w[:, 0] - w[:, -1]
    max_rise = np.max(np.diff(w, axis=1), axis=1)
    loud = w[:, 0] >= np.percentile(energy_db, 99) - 35
    cand = (drop >= min_drop) & (max_rise <= 2.0) & loud
    if cand.sum() < 20:
        return None
    x = np.arange(n) * E_HOP
    xc = x - x.mean()
    slopes = (w[cand] @ xc) / (xc @ xc)             # dB/s per candidate
    t60 = -60.0 / slopes[slopes < 0]
    return round(float(np.percentile(t60, 20)), 3)


def window_snr_db(energy, speech_e, i0, i1, noise_floor):
    """SNR from ref-labelled speech/non-speech frame energy inside a window."""
    sp = energy[i0:i1][speech_e[i0:i1]]
    ns = energy[i0:i1][~speech_e[i0:i1]]
    if len(sp) < 10:
        return None
    noise_pow = float(np.mean(ns)) if len(ns) * E_HOP >= 1.0 else noise_floor
    noise_pow = max(noise_pow, 1e-12)
    sig = max(float(np.mean(sp)) - noise_pow, noise_pow * 1e-4)
    return round(10.0 * math.log10(sig / noise_pow), 1)


# --------------------------------------------------------------------------- #
# Per-session worker
# --------------------------------------------------------------------------- #
def process_session(task):
    t0 = time.time()
    try:
        rec = _process_session(task)
        out = Path(task["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        return task["key"], time.time() - t0, None
    except Exception as e:  # keep the pool alive; report and continue
        return task["key"], time.time() - t0, f"{type(e).__name__}: {e}"


def _process_session(task):
    collar, skip_ov, window = task["collar"], task["skip_overlap"], task["window"]
    key = task["key"]
    ref = load_rttm(task["ref"], uri=key)
    hyp = load_rttm(task["pred"], uri=key)
    wav_dur = sf.info(task["wav"]).duration
    ref_ext = ref.get_timeline().extent()
    hyp_ext = hyp.get_timeline().extent()
    duration = max(wav_dur, ref_ext.end if ref_ext else 0.0,
                   hyp_ext.end if hyp_ext else 0.0)

    # ---- session-level optimal mapping, then rename hyp labels to ref labels.
    # Unmapped hyp labels get reserved names so they can never collide with a
    # reference label (they count as confusion / FA, never as correct).
    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_ov)
    mapping = dict(der_metric.optimal_mapping(ref, hyp))
    uniq = {l: f"__h{i}__" for i, l in enumerate(hyp.labels())}
    to_ref = {uniq[l]: mapping.get(l, f"__unmapped_{i}__")
              for i, l in enumerate(hyp.labels())}
    hyp_mapped = hyp.rename_labels(mapping=uniq).rename_labels(mapping=to_ref)

    # ---- frame masks (FRAME resolution) for attributes & speaker stats
    n_frames = int(math.ceil(duration / FRAME)) + 1
    ref_masks = activity_masks(ref, n_frames)
    hyp_masks = activity_masks(hyp_mapped, n_frames)
    ref_count = (np.sum([m for m in ref_masks.values()], axis=0)
                 if ref_masks else np.zeros(n_frames, dtype=np.int8))
    speech_mask = ref_count >= 1
    overlap_mask = ref_count >= 2
    hyp_any = (np.any([m for m in hyp_masks.values()], axis=0)
               if hyp_masks else np.zeros(n_frames, dtype=bool))

    # ---- turns / re-entries / first appearances
    seg_by_spk = {}
    for seg, _, label in ref.itertracks(yield_label=True):
        seg_by_spk.setdefault(label, []).append((seg.start, seg.end))
    turns_by_spk = {s: merge_turns(v) for s, v in seg_by_spk.items()}
    all_turns = sorted(t for ts in turns_by_spk.values() for t in ts)
    reentries, first_seen = [], {}
    for spk, turns in turns_by_spk.items():
        first_seen[spk] = turns[0][0]
        for prev, cur in zip(turns, turns[1:]):
            if cur[0] - prev[1] >= REENTRY_GAP_SEC:
                reentries.append(cur[0])

    # ---- audio energy, noise floor, reverb proxy
    energy = frame_energy(Path(task["wav"]), duration)
    energy_db = 10.0 * np.log10(energy + 1e-12)
    e_centers = (np.arange(len(energy)) * E_HOP + E_WIN / 2)
    e_idx = np.minimum((e_centers / FRAME).astype(int), n_frames - 1)
    speech_e = speech_mask[e_idx]
    nonspeech = energy[~speech_e]
    noise_floor = float(np.percentile(nonspeech, 5)) if len(nonspeech) > 100 \
        else float(np.percentile(energy, 5))
    reverb_t60 = estimate_reverb_t60(energy_db)

    # ---- per-window scoring + attributes
    ier = IdentificationErrorRate(collar=collar, skip_overlap=skip_ov)
    der_local = DiarizationErrorRate(collar=collar, skip_overlap=skip_ov)
    n_win = max(1, int(math.ceil(duration / window)))
    windows = []
    for w in range(n_win):
        ws, we = w * window, min((w + 1) * window, duration)
        region = Segment(ws, we)
        ref_w = ref.crop(region, mode="intersection")
        d = ier(ref_w, hyp_mapped.crop(region, mode="intersection"),
                detailed=True)
        conf_loc = der_local(ref_w, hyp.crop(region, mode="intersection"),
                             detailed=True)["confusion"]

        i0, i1 = int(ws / FRAME), min(int(we / FRAME), n_frames)
        speech_s = float(np.sum(speech_mask[i0:i1])) * FRAME
        overlap_s = float(np.sum(overlap_mask[i0:i1])) * FRAME
        active = [s for s, m in ref_masks.items() if m[i0:i1].any()]
        w_turns = [t for t in all_turns if t[0] < we and t[1] > ws]
        short = [t for t in w_turns if t[1] - t[0] < SHORT_TURN_SEC]
        ei0 = int(ws / E_HOP)
        ei1 = min(int(we / E_HOP), len(energy))
        windows.append({
            "t0": round(ws, 1),
            "ref": round(float(d["total"]), 2),
            "miss": round(float(d["missed detection"]), 2),
            "fa": round(float(d["false alarm"]), 2),
            "conf": round(float(d["confusion"]), 2),
            "conf_loc": round(float(conf_loc), 2),
            "nspk": len(active),
            "speech": round(speech_s, 1),
            "ovl": round(overlap_s / speech_s, 3) if speech_s > 0 else None,
            "turn": round(float(np.mean([t[1] - t[0] for t in w_turns])), 2)
                    if w_turns else None,
            "nshort": len(short),
            "nreent": sum(1 for t in reentries if ws <= t < we),
            "nnew": sum(1 for t in first_seen.values() if ws <= t < we),
            "pos": round((ws + we) / 2 / duration, 3),
            "snr": window_snr_db(energy, speech_e, ei0, ei1, noise_floor),
        })

    speakers = speaker_stats(ref_masks, hyp_masks, hyp_any, ref_count,
                             first_seen, to_ref)

    return {
        "key": key, "collection": task["collection"], "row": task["row"],
        "dataset": task["dataset"], "duration": round(duration, 1),
        "collar": collar, "window": window, "reverb": reverb_t60,
        "n_ref_spk": len(ref_masks), "n_hyp_spk": len(hyp.labels()),
        "windows": windows, "speakers": speakers,
    }


def speaker_stats(ref_masks, hyp_masks, hyp_any, ref_count, first_seen, to_ref):
    """Detection/attribution latency after first appearance, flicker, purity."""
    mapped_ref = set(to_ref.values())
    # frame-wise "some hyp label" id for flicker assignment on solo frames
    labels = sorted(hyp_masks)
    label_of = np.full(len(hyp_any), -1, dtype=np.int16)
    for i, l in reversed(list(enumerate(labels))):
        label_of[hyp_masks[l]] = i
    stats = []
    for spk, m in ref_masks.items():
        own = hyp_masks.get(spk)          # mapped hyp activity for this speaker
        frames = np.flatnonzero(m)
        if len(frames) == 0:
            continue
        f0 = int(first_seen[spk] / FRAME)
        after = frames[frames >= f0]
        det = after[hyp_any[after]]
        att = after[own[after]] if own is not None else np.array([], int)
        solo = frames[ref_count[frames] == 1]
        n_solo = len(solo)
        correct = own[solo] if own is not None else np.zeros(n_solo, bool)
        assigned = np.where(correct, -2, label_of[solo])   # -2 = correct label
        covered = assigned != -1
        # flicker: label switches between consecutive solo frames <=1 s apart
        switches = 0
        if n_solo > 1:
            close = np.diff(solo) * FRAME <= 1.0
            both = covered[:-1] & covered[1:]
            switches = int(np.sum(close & both &
                                  (assigned[:-1] != assigned[1:])))
        solo_s = n_solo * FRAME
        stats.append({
            "spk": spk,
            "mapped": spk in mapped_ref,
            "first": round(first_seen[spk], 1),
            "det_lat": round(float(det[0] - f0) * FRAME, 2) if len(det) else None,
            "att_lat": round(float(att[0] - f0) * FRAME, 2) if len(att) else None,
            "solo_s": round(solo_s, 1),
            "purity": round(float(np.mean(correct)), 3) if n_solo else None,
            "miss_frac": round(float(np.mean(~covered)), 3) if n_solo else None,
            "switch_per_min": round(switches / (solo_s / 60.0), 2)
                              if solo_s >= 5.0 else None,
        })
    return stats


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_tasks(window, only_datasets=None, limit_files=None):
    tasks = []
    for name, wavs, rttms in find_collections():
        ds = dataset_of(name)
        if only_datasets and ds not in only_datasets:
            continue
        cfg = cfg_for(ds)
        row = name if ds in SPLIT_BY_LEAF else ds
        pred_dir = pred_dir_for(name)
        refs = sorted(rttms.glob("*.rttm"))
        if limit_files:
            refs = refs[:limit_files]
        for ref in refs:
            pred = pred_dir / ref.name
            wav = wavs / (ref.stem + ".wav")
            out = OUT_ROOT / "records" / name / (ref.stem + ".json")
            if not pred.exists() or not wav.exists():
                continue
            if out.exists():
                continue
            tasks.append(dict(
                key=ref.stem, collection=name, row=row, dataset=ds,
                ref=str(ref), pred=str(pred), wav=str(wav), out=str(out),
                collar=cfg["collar"], skip_overlap=cfg["skip_overlap"],
                window=window,
            ))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=WINDOW_SEC)
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset names to limit to")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    only = set(args.datasets.split(",")) if args.datasets else None
    tasks = build_tasks(args.window, only, args.limit_files)
    print(f"[attr] {len(tasks)} sessions to annotate "
          f"(window={args.window:.0f}s, workers={args.workers})", flush=True)
    if not tasks:
        return
    done, failed, t0 = 0, 0, time.time()
    with Pool(args.workers) as pool:
        for key, dt, err in pool.imap_unordered(process_session, tasks):
            done += 1
            if err:
                failed += 1
                print(f"[attr] FAIL {key}: {err}", flush=True)
            if done % 50 == 0 or done == len(tasks):
                print(f"[attr] {done}/{len(tasks)} "
                      f"({time.time() - t0:.0f}s, {failed} failed)", flush=True)
    print(f"[attr] complete: {done - failed} ok, {failed} failed "
          f"-> {OUT_ROOT / 'records'}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
