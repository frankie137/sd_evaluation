#!/usr/bin/env python3
"""Apply human edits from the review pages to build the final eval sets.

Reads out/curation/edits/<dataset>_edits.json (the file downloaded from
review.html; '/' in dataset names is replaced by '_') and the manifest, then
writes out/curation/final/<dataset>/{wavs,rttms}/ per window:
  - discarded windows are dropped
  - edited windows get their reference RTTM rewritten from the edited segments
  - untouched windows are copied as-is
Datasets without an edits file are skipped. Ends with a per-bin summary and a
rescored DER (v1 pred vs curated reference) for cross-checking.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

import diar_eval
from curate_eval_set import OUT_DIR, MANIFEST_PATH, window_id, write_rttm

EDITS_DIR = OUT_DIR / "edits"
FINAL_DIR = OUT_DIR / "final"


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for ds, bins in manifest["datasets"].items():
        edits_path = EDITS_DIR / (ds.replace("/", "_") + "_edits.json")
        if not edits_path.exists():
            print(f"[apply] {ds}: no edits file ({edits_path.name}), skip", flush=True)
            continue
        edits = json.loads(edits_path.read_text())
        ds_dir = OUT_DIR / ds
        out_dir = FINAL_DIR / ds
        stats = defaultdict(lambda: dict(kept=0, dropped=0, edited=0))
        pooled = dict(err=0.0, ref=0.0)
        for label, wins in bins.items():
            for w in wins:
                wid = window_id(w)
                e = edits.get(wid)
                if e is None:
                    print(f"[apply] {ds}: {wid} missing from edits, keeping original",
                          flush=True)
                    e = {"action": "keep", "edited": False}
                if e["action"] == "discard":
                    stats[label]["dropped"] += 1
                    continue
                (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
                (out_dir / "rttms").mkdir(parents=True, exist_ok=True)
                shutil.copy2(ds_dir / "wavs" / f"{wid}.wav",
                             out_dir / "wavs" / f"{wid}.wav")
                if e.get("edited"):
                    segs = sorted((round(a, 3), round(b, 3), spk)
                                  for a, b, spk in e["segments"] if b > a)
                    write_rttm(out_dir / "rttms" / f"{wid}.rttm", wid, segs)
                    stats[label]["edited"] += 1
                else:
                    shutil.copy2(ds_dir / "rttms" / f"{wid}.rttm",
                                 out_dir / "rttms" / f"{wid}.rttm")
                stats[label]["kept"] += 1
                dur = w["t1"] - w["t0"]
                res = diar_eval.evaluate(str(out_dir / "rttms" / f"{wid}.rttm"),
                                         str(ds_dir / "preds" / f"{wid}.rttm"),
                                         windows=[(0.0, dur)], collar=0.25)
                pooled["err"] += res.error
                pooled["ref"] += res.reference
        der = pooled["err"] / pooled["ref"] * 100 if pooled["ref"] else 0.0
        parts = "  ".join(f"{l}: keep={s['kept']} (edit={s['edited']}) drop={s['dropped']}"
                          for l, s in sorted(stats.items()))
        print(f"[apply] {ds}: {parts}", flush=True)
        print(f"[apply] {ds}: v1-vs-curated pooled DER={der:.2f}% "
              f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
