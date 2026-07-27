#!/usr/bin/env python3
"""Benchmark Sortformer v1 on the calibrated eval set (out/curation/final_vad).

For every window under out/curation/final_vad/<dir>: reference is
rttms/<wid>.rttm (built by apply_vad_edits.py); hypothesis is the exported v1
prediction out/curation/<dir>/preds/<wid>.rttm. Scoring follows the curation
convention: whole window, collar=0.25, skip_overlap=False, per-window optimal
speaker mapping.

Prints a per-subset table (with the pre-calibration DER next to it for the
calibrated subsets, scored against out/curation/final references) and writes
out/curation/final_vad/final_benchmark.{jsonl,md}.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

import diar_eval
from vad_review import FINAL_DIR, window_dirs

FINAL_VAD_DIR = FINAL_DIR.parent / "final_vad"
PREDS_ROOT = FINAL_DIR.parent
COLLAR = 0.25


def score_window(ref: Path, pred: Path, dur: float):
    res = diar_eval.evaluate(str(ref), str(pred), windows=[(0.0, dur)],
                             collar=COLLAR, skip_overlap=False)
    return res.windows[0]


def main():
    calibrated_subsets = set(json.loads(
        (FINAL_VAD_DIR / "calibrated_subsets.json").read_text()))
    rows = []
    totals = defaultdict(lambda: defaultdict(float))
    for ds in sorted(d.name for d in FINAL_VAD_DIR.iterdir() if d.is_dir()):
        for d in window_dirs(FINAL_VAD_DIR / ds):
            rel = str(d.relative_to(FINAL_VAD_DIR))
            calibrated = rel in calibrated_subsets
            for ref in sorted((d / "rttms").glob("*.rttm")):
                wid = ref.stem
                pred = PREDS_ROOT / rel / "preds" / f"{wid}.rttm"
                info = sf.info(str(d / "wavs" / f"{wid}.wav"))
                dur = info.frames / info.samplerate
                wr = score_window(ref, pred, dur)
                row = {"subset": rel, "wid": wid, "calibrated": calibrated,
                       "ref_sec": round(wr.reference, 3),
                       "miss": round(wr.miss, 3),
                       "fa": round(wr.false_alarm, 3),
                       "conf": round(wr.confusion, 3),
                       "der": round(wr.der, 3) if wr.der is not None else None}
                t = totals[rel]
                t["ref"] += wr.reference; t["miss"] += wr.miss
                t["fa"] += wr.false_alarm; t["conf"] += wr.confusion
                t["n"] += 1
                if calibrated:
                    orig = FINAL_DIR / rel / "rttms" / f"{wid}.rttm"
                    wo = score_window(orig, pred, dur)
                    row["der_precalib"] = round(wo.der, 3) if wo.der is not None else None
                    t["ref0"] += wo.reference; t["miss0"] += wo.miss
                    t["fa0"] += wo.false_alarm; t["conf0"] += wo.confusion
                rows.append(row)
            print(f"[bench] {rel}: {int(totals[rel]['n'])} windows", flush=True)

    def der(t, suf=""):
        ref = t["ref" + (suf or "")]
        if not ref:
            return None
        return (t["miss" + suf] + t["fa" + suf] + t["conf" + suf]) / ref * 100

    header = (f"{'subset':22s} {'win':>4s} {'ref(s)':>8s} {'miss%':>7s} "
              f"{'fa%':>7s} {'conf%':>7s} {'DER%':>7s} {'precalib':>9s}")
    lines = [header, "-" * len(header)]
    for rel, t in sorted(totals.items()):
        pre = der(t, "0") if "ref0" in t else None
        lines.append(
            f"{rel:22s} {int(t['n']):4d} {t['ref']:8.1f} "
            f"{t['miss'] / t['ref'] * 100:7.2f} {t['fa'] / t['ref'] * 100:7.2f} "
            f"{t['conf'] / t['ref'] * 100:7.2f} {der(t):7.2f} "
            f"{('%9.2f' % pre) if pre is not None else '        -'}")
    table = "\n".join(lines)
    print(table)

    with open(FINAL_VAD_DIR / "final_benchmark.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md = ("# Sortformer v1 benchmark on the final curated eval set\n\n"
          f"Scoring: whole 90 s window, collar={COLLAR}, skip_overlap=False, "
          "per-window optimal mapping.\n"
          "`precalib` = DER against the pre-calibration reference "
          "(calibrated subsets only).\n\n```\n" + table + "\n```\n")
    (FINAL_VAD_DIR / "final_benchmark.md").write_text(md)
    print(f"[bench] wrote final_benchmark.jsonl / final_benchmark.md "
          f"-> {FINAL_VAD_DIR}", flush=True)


if __name__ == "__main__":
    main()
