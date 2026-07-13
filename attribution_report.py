#!/usr/bin/env python3
"""Aggregate window-attribution records into a self-contained HTML report.

Reads the per-session JSONs produced by window_attribution.py, flattens them
into compact columnar arrays, and injects them into
attribution_report_template.html (placeholder /*__DATA__*/). The resulting page
is fully offline: all bucketing / filtering / charting happens in inline JS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import EXEMPT, dataset_of

OUT_ROOT = Path("/workspace/sd_evaluation/out/attribution")
TEMPLATE = Path(__file__).parent / "attribution_report_template.html"

WIN_FIELDS = ["r", "s", "t0", "ref", "miss", "fa", "conf", "cl", "k", "sp",
              "ov", "tn", "ns", "re", "nw", "pos", "snr"]
SPK_FIELDS = ["s", "map", "first", "det", "att", "solo", "pur", "mf", "sw"]


def collect(records_dir: Path):
    rows, row_idx = [], {}
    sessions = []
    win = {k: [] for k in WIN_FIELDS}
    spk = {k: [] for k in SPK_FIELDS}
    window_sec = None
    for path in sorted(records_dir.rglob("*.json")):
        rec = json.loads(path.read_text())
        window_sec = rec["window"]
        row = rec["row"]
        # Same speaker-count split as benchmark.py: non-EXEMPT datasets are
        # reported as two subsets, <=4 and >4 reference speakers.
        if dataset_of(row) not in EXEMPT:
            subset = "<=4 spk" if rec["n_ref_spk"] <= 4 else ">4 spk"
            row = f"{row} ({subset})"
        if row not in row_idx:
            row_idx[row] = len(rows)
            rows.append(row)
        r = row_idx[row]
        s = len(sessions)
        sums = {k: 0.0 for k in ("ref", "miss", "fa", "conf", "cl")}
        for w in rec["windows"]:
            win["r"].append(r)
            win["s"].append(s)
            win["t0"].append(w["t0"])
            win["ref"].append(w["ref"])
            win["miss"].append(w["miss"])
            win["fa"].append(w["fa"])
            win["conf"].append(w["conf"])
            win["cl"].append(w["conf_loc"])
            win["k"].append(w["nspk"])
            win["sp"].append(w["speech"])
            win["ov"].append(w["ovl"])
            win["tn"].append(w["turn"])
            win["ns"].append(w["nshort"])
            win["re"].append(w["nreent"])
            win["nw"].append(w["nnew"])
            win["pos"].append(w["pos"])
            win["snr"].append(w["snr"])
            sums["ref"] += w["ref"]
            sums["miss"] += w["miss"]
            sums["fa"] += w["fa"]
            sums["conf"] += w["conf"]
            sums["cl"] += w["conf_loc"]
        for p in rec["speakers"]:
            spk["s"].append(s)
            spk["map"].append(1 if p["mapped"] else 0)
            spk["first"].append(p["first"])
            spk["det"].append(p["det_lat"])
            spk["att"].append(p["att_lat"])
            spk["solo"].append(p["solo_s"])
            spk["pur"].append(p["purity"])
            spk["mf"].append(p["miss_frac"])
            spk["sw"].append(p["switch_per_min"])
        sessions.append({
            "key": rec["key"], "r": r, "dur": rec["duration"],
            "rev": rec["reverb"], "collar": rec["collar"],
            "nspk": rec["n_ref_spk"], "nhyp": rec["n_hyp_spk"],
            **{k: round(v, 2) for k, v in sums.items()},
        })
    # Sort rows alphabetically so both subsets of a dataset sit next to each
    # other in the legend, then remap the row indices accordingly.
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    remap = {old: new for new, old in enumerate(order)}
    rows = [rows[i] for i in order]
    win["r"] = [remap[r] for r in win["r"]]
    for sess in sessions:
        sess["r"] = remap[sess["r"]]
    return {"rows": rows, "sessions": sessions, "win": win, "spk": spk,
            "window": window_sec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(OUT_ROOT / "records"))
    ap.add_argument("--out", default=str(OUT_ROOT / "attribution_report.html"))
    args = ap.parse_args()

    data = collect(Path(args.records))
    n_win = len(data["win"]["r"])
    print(f"[report] {len(data['sessions'])} sessions, {n_win} windows, "
          f"{len(data['spk']['s'])} speakers, {len(data['rows'])} rows")

    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[report] saved -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
