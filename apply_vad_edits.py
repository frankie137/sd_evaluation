#!/usr/bin/env python3
"""Build the boundary-calibrated eval set under out/curation/final_vad.

The output mirrors the structure of out/curation/final: one directory per
subset with wavs/ (copied) and rttms/. For subsets with a human edits file
(out/curation/vad_edits/<dir>_vad_edits.json, downloaded from
vad_review.html), edited windows use the human segments and the rest use the
raw Silero-VAD calibration (rttms_vad). Subsets without an edits file keep
their original references unchanged.

Writes final_vad/calibrated_subsets.json listing which subsets were
calibrated, for downstream scripts.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

from curate_eval_set import write_rttm
from vad_review import FINAL_DIR, window_dirs

FINAL_VAD_DIR = FINAL_DIR.parent / "final_vad"
EDITS_DIR = FINAL_DIR.parent / "vad_edits"


def main():
    calibrated = []
    for ds in sorted(d.name for d in FINAL_DIR.iterdir() if d.is_dir()):
        for d in window_dirs(FINAL_DIR / ds):
            rel = str(d.relative_to(FINAL_DIR))
            out_dir = FINAL_VAD_DIR / rel
            (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
            (out_dir / "rttms").mkdir(parents=True, exist_ok=True)
            edits_path = EDITS_DIR / (rel.replace("/", "_") + "_vad_edits.json")
            edits = (json.loads(edits_path.read_text())
                     if edits_path.exists() else None)
            if edits is not None:
                calibrated.append(rel)
            n_edited = n_vad = n_orig = 0
            for orig in sorted((d / "rttms").glob("*.rttm")):
                wid = orig.stem
                shutil.copy(d / "wavs" / f"{wid}.wav",
                            out_dir / "wavs" / f"{wid}.wav")
                e = edits.get(wid) if edits else None
                if e and e.get("edited"):
                    segs = sorted((round(a, 3), round(b, 3), spk)
                                  for a, b, spk in e["segments"])
                    write_rttm(out_dir / "rttms" / f"{wid}.rttm", wid, segs)
                    n_edited += 1
                elif edits is not None:
                    shutil.copy(d / "rttms_vad" / f"{wid}.rttm",
                                out_dir / "rttms" / f"{wid}.rttm")
                    n_vad += 1
                else:
                    shutil.copy(orig, out_dir / "rttms" / f"{wid}.rttm")
                    n_orig += 1
            print(f"[final-vad] {rel}: {n_edited} edited + {n_vad} vad-as-is "
                  f"+ {n_orig} original -> {out_dir}", flush=True)
    (FINAL_VAD_DIR / "calibrated_subsets.json").write_text(
        json.dumps(calibrated, indent=2))
    print(f"[final-vad] calibrated subsets: {calibrated}", flush=True)


if __name__ == "__main__":
    main()
