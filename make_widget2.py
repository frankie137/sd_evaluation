#!/usr/bin/env python3
"""Emit a tiny base64-packed payload (one JS object literal line) for the widget."""
import base64
import json
import math
from pathlib import Path

from visualize_compare import parse_rttm, score_pa

AUDIO_DUR = 1776.99
GRID = 900
GT = "/workspace/speaker_diarization_benchmark/Alimeeting/Far/rttms/R8005_M8009.rttm"
PRED = "/workspace/sortformer_diar/out/R8005_M8009.pred.rttm"
META = json.loads(Path("/workspace/sortformer_diar/out/R8005_M8009.meta.json").read_text())
COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]


def lane_to_b64(segs, spk):
    bits = bytearray(GRID)
    for s, e, c in segs:
        if c != spk:
            continue
        lo = max(0, int(s / AUDIO_DUR * GRID))
        hi = min(GRID, int(math.ceil(e / AUDIO_DUR * GRID)))
        for i in range(lo, hi):
            bits[i] = 1
    packed = bytearray((GRID + 7) // 8)
    for i in range(GRID):
        if bits[i]:
            packed[i >> 3] |= 1 << (7 - (i & 7))
    return base64.b64encode(bytes(packed)).decode()


def main():
    gt_segs = parse_rttm(GT)
    pr_segs = parse_rttm(PRED)
    pa = score_pa(GT, PRED, "R8005_M8009")  # pyannote.metrics: DER + mapping
    gt_lanes = pa["ref_labels"]
    ref2hyp = pa["ref2hyp"]
    pr_lanes = [ref2hyp.get(g) for g in gt_lanes]

    D = {
        "g": [lane_to_b64(gt_segs, s) for s in gt_lanes],
        "p": [lane_to_b64(pr_segs, s) if s else "" for s in pr_lanes],
        "gl": [s.replace("N_", "") for s in gt_lanes],
        "pl": [(p.replace("speaker_", "spk") if p else "-") for p in pr_lanes],
        "c": COLORS, "n": GRID, "dur": round(AUDIO_DUR),
        "dc": pa["std"]["der"], "dn": pa["strict"]["der"],
        "pa": pa["pa025"]["der"], "jer": pa["jer"],
        "ms": pa["std"]["miss_pct"], "fa": pa["std"]["fa_pct"], "cf": pa["std"]["conf_pct"],
        "rtf": META.get("rtf"), "inf": META.get("infer_seconds"),
    }
    line = "const D=" + json.dumps(D, separators=(",", ":")) + ";"
    print(line)
    print("\n--- LEN:", len(line), "chars ---")


if __name__ == "__main__":
    main()
