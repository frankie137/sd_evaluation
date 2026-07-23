#!/usr/bin/env python3
"""Error attribution for Sortformer v1 on the curated (human-corrected) eval set.

Rescoring of the old windowed attribution (out/v1_error_analysis_report.md) on
the final curated windows only (out/curation/final/<dataset>/rttms, the windows
kept after human review, with corrected reference labels). For every kept
window this script scores the exported v1 window prediction against

  * the CURATED reference (primary numbers), and
  * the ORIGINAL reference (out/curation/<dataset>/rttms) to quantify how much
    of the previously reported error was annotation noise,

both at the curation collar (0.25) and at collar 0 (boundary/FA view), and
recomputes all reference-side factors (overlap ratio, turn duration, active
speaker count, per-speaker speech, speaker-count correctness) from the curated
labels. Bin edges match the old report so tables are directly comparable.

Outputs:
  out/curation/curated_window_attribution.jsonl  -- one row per kept window
  (report tables are printed by --report, consumed by the markdown report)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

import diar_eval
from curate_eval_set import (BINS, MANIFEST_PATH, OUT_DIR, SCORES_PATH,
                             bin_of, load_rttm_segments, window_id)

FINAL_DIR = OUT_DIR / "final"
ROWS_PATH = OUT_DIR / "curated_window_attribution.jsonl"

COLLAR_MAIN = 0.25
FRAME = 0.01
TURN_MERGE_GAP = 0.5
SHORT_TURN_SEC = 1.0

OVL_BINS = [("0", 0.0, 1e-9), ("0-5%", 1e-9, 0.05), ("5-15%", 0.05, 0.15),
            ("15-30%", 0.15, 0.30), (">=30%", 0.30, float("inf"))]
TURN_BINS = [("<1.5s", 0, 1.5), ("1.5-2.5s", 1.5, 2.5), ("2.5-4s", 2.5, 4.0),
             ("4-8s", 4.0, 8.0), ("8-15s", 8.0, 15.0), (">=15s", 15.0, float("inf"))]


def bucket(value, bins):
    if value is None:
        return None
    for label, lo, hi in bins:
        if lo <= value < hi:
            return label
    return None


def merge_turns(segments, gap=TURN_MERGE_GAP):
    turns = []
    for s, e in sorted(segments):
        if turns and s - turns[-1][1] <= gap:
            turns[-1][1] = max(turns[-1][1], e)
        else:
            turns.append([s, e])
    return [(s, e) for s, e in turns]


def ref_factors(segs, dur):
    """Reference-side factors from window-local (start, end, spk) segments."""
    n_frames = int(math.ceil(dur / FRAME)) + 1
    by_spk = defaultdict(list)
    for a, b, spk in segs:
        by_spk[spk].append((a, b))
    count = np.zeros(n_frames, dtype=np.int16)
    speech_by_spk = {}
    for spk, ss in by_spk.items():
        m = np.zeros(n_frames, dtype=bool)
        for a, b in ss:
            i0, i1 = max(0, int(a / FRAME)), min(n_frames, int(math.ceil(b / FRAME)))
            if i1 > i0:
                m[i0:i1] = True
        count += m
        speech_by_spk[spk] = round(float(m.sum()) * FRAME, 2)
    speech_s = float((count >= 1).sum()) * FRAME
    overlap_s = float((count >= 2).sum()) * FRAME
    turns = [t for ss in by_spk.values() for t in merge_turns(ss)]
    turn_durs = [e - s for s, e in turns]
    return {
        "nspk": len(by_spk),
        "speech": round(speech_s, 2),
        "ovl": round(overlap_s / speech_s, 4) if speech_s > 0 else None,
        "turn": round(float(np.mean(turn_durs)), 2) if turn_durs else None,
        "nshort": sum(1 for d in turn_durs if d < SHORT_TURN_SEC),
        "spk_speech": speech_by_spk,
    }, count


def miss_fa_split(ref_count, hyp_count):
    """Count-based frame decomposition of miss into overlap vs solo regions."""
    miss = np.maximum(ref_count - hyp_count, 0)
    ovl_miss = float(miss[ref_count >= 2].sum()) * FRAME
    solo_miss = float(miss[ref_count == 1].sum()) * FRAME
    return round(ovl_miss, 2), round(solo_miss, 2)


def hyp_count_mask(segs, dur):
    n_frames = int(math.ceil(dur / FRAME)) + 1
    count = np.zeros(n_frames, dtype=np.int16)
    by_spk = defaultdict(list)
    for a, b, spk in segs:
        by_spk[spk].append((a, b))
    for spk, ss in by_spk.items():
        m = np.zeros(n_frames, dtype=bool)
        for a, b in ss:
            i0, i1 = max(0, int(a / FRAME)), min(n_frames, int(math.ceil(b / FRAME)))
            if i1 > i0:
                m[i0:i1] = True
        count += m
    return count, len(by_spk)


def score(ref_path, pred_path, dur, collar):
    res = diar_eval.evaluate(str(ref_path), str(pred_path),
                             windows=[(0.0, dur)], collar=collar,
                             skip_overlap=False)
    w = res.windows[0]
    return w


def build_rows():
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = []
    for ds, bins in manifest["datasets"].items():
        n_kept = 0
        for label, wins in bins.items():
            for w in wins:
                wid = window_id(w)
                ref_cur = FINAL_DIR / ds / "rttms" / f"{wid}.rttm"
                if not ref_cur.exists():        # discarded during review
                    continue
                ref_orig = OUT_DIR / ds / "rttms" / f"{wid}.rttm"
                pred = OUT_DIR / ds / "preds" / f"{wid}.rttm"
                dur = w["t1"] - w["t0"]
                cur_segs = load_rttm_segments(ref_cur)
                orig_segs = load_rttm_segments(ref_orig)
                factors, ref_count = ref_factors(cur_segs, dur)
                hyp_count, n_hyp = hyp_count_mask(load_rttm_segments(pred), dur)
                n = min(len(ref_count), len(hyp_count))
                ovl_miss, solo_miss = miss_fa_split(ref_count[:n], hyp_count[:n])
                sc = score(ref_cur, pred, dur, COLLAR_MAIN)
                sc0 = score(ref_cur, pred, dur, 0.0)
                so = score(ref_orig, pred, dur, COLLAR_MAIN)
                mapped_ref = set(sc.mapping.values())
                dropped = [(spk, s) for spk, s in factors["spk_speech"].items()
                           if spk not in mapped_ref]
                edited = any(abs(a - c) > 1e-6 or abs(b - d) > 1e-6 or x != y
                             for (a, b, x), (c, d, y)
                             in zip(sorted(cur_segs), sorted(orig_segs))) \
                    or len(cur_segs) != len(orig_segs)
                rows.append({
                    "dataset": ds, "key": w["key"], "wid": wid,
                    "sel_bin": label, "edited": edited,
                    "ref_sec": round(sc.reference, 2),
                    "miss": round(sc.miss, 2), "fa": round(sc.false_alarm, 2),
                    "conf": round(sc.confusion, 2), "der": sc.der,
                    "miss0": round(sc0.miss, 2), "fa0": round(sc0.false_alarm, 2),
                    "conf0": round(sc0.confusion, 2),
                    "ref_sec0": round(sc0.reference, 2),
                    "orig_ref_sec": round(so.reference, 2),
                    "orig_miss": round(so.miss, 2),
                    "orig_fa": round(so.false_alarm, 2),
                    "orig_conf": round(so.confusion, 2), "orig_der": so.der,
                    "nspk": factors["nspk"], "n_hyp": n_hyp,
                    "speech": factors["speech"], "ovl": factors["ovl"],
                    "turn": factors["turn"], "nshort": factors["nshort"],
                    "spk_speech": factors["spk_speech"],
                    "dropped_spk": dropped,
                    "ovl_miss": ovl_miss, "solo_miss": solo_miss,
                })
                n_kept += 1
        print(f"[attr] {ds}: {n_kept} windows", flush=True)
    with open(ROWS_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[attr] wrote {len(rows)} windows -> {ROWS_PATH}", flush=True)
    return rows


def load_rows():
    return [json.loads(l) for l in open(ROWS_PATH)]


# --------------------------------------------------------------------------- #
# aggregation helpers
# --------------------------------------------------------------------------- #
def pool(rows, miss="miss", fa="fa", conf="conf", ref="ref_sec", w=None):
    """Pooled err%% components over rows; optional per-row weights."""
    get_w = (lambda r: w[(r["dataset"], r["sel_bin"])]) if w else (lambda r: 1.0)
    tref = sum(r[ref] * get_w(r) for r in rows)
    if tref <= 0:
        return None
    out = {}
    for name, k in (("miss", miss), ("fa", fa), ("conf", conf)):
        out[name] = sum(r[k] * get_w(r) for r in rows) / tref * 100
    out["der"] = out["miss"] + out["fa"] + out["conf"]
    out["ref_h"] = tref / 3600
    return out


def table(rows, key_fn, bins_order, **pool_kw):
    groups = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if k is not None:
            groups[k].append(r)
    lines = []
    for label in bins_order:
        g = groups.get(label, [])
        p = pool(g, **pool_kw) if g else None
        if p:
            lines.append((label, len(g), p))
    return lines


def natural_weights(kept_rows):
    """Weight = eligible-population count / kept count, per (subset, DER bin)."""
    eligible = defaultdict(int)
    from curate_eval_set import (MAX_REF_SPEAKERS, MIN_REF_SEC, subset_of)
    for l in open(SCORES_PATH):
        s = json.loads(l)
        if (s["der"] is not None and s["ref_sec"] >= MIN_REF_SEC
                and s["n_ref_speakers"] <= MAX_REF_SPEAKERS):
            eligible[(subset_of(s), bin_of(s["der"]))] += 1
    kept = defaultdict(int)
    for r in kept_rows:
        kept[(r["dataset"], r["sel_bin"])] += 1
    return {k: eligible[k] / n for k, n in kept.items()}


def report_stage():
    """Print every table used by out/curation/v1_curated_error_report.md."""
    rows = load_rows()

    def show(title, key_fn, order, rs=rows, **kw):
        print(f"--- {title}")
        groups = defaultdict(list)
        for r in rs:
            k = key_fn(r)
            if k is not None:
                groups[k].append(r)
        for label in order:
            g = groups.get(label, [])
            if not g:
                continue
            p = pool(g, **kw)
            print(f"{str(label):22s} n={len(g):3d}  DER {p['der']:6.2f}  "
                  f"miss {p['miss']:6.2f}  fa {p['fa']:5.2f}  conf {p['conf']:5.2f}")

    c0 = dict(miss="miss0", fa="fa0", conf="conf0", ref="ref_sec0")
    orig = dict(miss="orig_miss", fa="orig_fa", conf="orig_conf", ref="orig_ref_sec")
    for name, kw in (("curated c=0.25", {}), ("curated c=0", c0),
                     ("original labels c=0.25", orig)):
        p = pool(rows, **kw)
        print(f"[headline] {name}: DER {p['der']:.2f} miss {p['miss']:.2f} "
              f"fa {p['fa']:.2f} conf {p['conf']:.2f}")
    pw = pool(rows, w=natural_weights(rows))
    print(f"[headline] natural-weighted curated: DER {pw['der']:.2f} "
          f"miss {pw['miss']:.2f} fa {pw['fa']:.2f} conf {pw['conf']:.2f}")
    tot_err = sum(r["miss"] + r["fa"] + r["conf"] for r in rows)
    print("[headline] error shares:",
          "  ".join(f"{k} {sum(r[k] for r in rows) / tot_err * 100:.1f}%"
                    for k in ("miss", "fa", "conf")))

    datasets = sorted({r["dataset"] for r in rows})
    show("dataset (curated c=0.25)", lambda r: r["dataset"], datasets)
    show("dataset (original labels c=0.25)", lambda r: r["dataset"], datasets, **orig)
    show("dataset (curated c=0)", lambda r: r["dataset"], datasets, **c0)
    show("ovl bins (c=0.25)", lambda r: bucket(r["ovl"], OVL_BINS),
         [b[0] for b in OVL_BINS])
    show("ovl bins (c=0)", lambda r: bucket(r["ovl"], OVL_BINS),
         [b[0] for b in OVL_BINS], **c0)
    show("turn bins (c=0.25)", lambda r: bucket(r["turn"], TURN_BINS),
         [b[0] for b in TURN_BINS])
    show("turn bins (c=0)", lambda r: bucket(r["turn"], TURN_BINS),
         [b[0] for b in TURN_BINS], **c0)
    show("nspk (c=0.25)", lambda r: r["nspk"], [1, 2, 3, 4])
    show("nshort (c=0.25)", lambda r: ("0" if r["nshort"] == 0 else
                                       "1-5" if r["nshort"] <= 5 else
                                       "6-10" if r["nshort"] <= 10 else ">10"),
         ["0", "1-5", "6-10", ">10"])

    print("--- contribution shares (c=0.25)")
    tot_ref = sum(r["ref_sec"] for r in rows)
    for name, cond in [
            ("ovl>=15%", lambda r: r["ovl"] is not None and r["ovl"] >= 0.15),
            ("turn<2.5s", lambda r: r["turn"] is not None and r["turn"] < 2.5),
            ("nspk>=3", lambda r: r["nspk"] >= 3),
            ("nshort>10", lambda r: r["nshort"] > 10)]:
        g = [r for r in rows if cond(r)]
        e = sum(r["miss"] + r["fa"] + r["conf"] for r in g)
        f = sum(r["ref_sec"] for r in g)
        print(f"{name:10s} ref {f / tot_ref * 100:4.1f}%  err {e / tot_err * 100:4.1f}%  "
              f"lift {e / tot_err / (f / tot_ref):.2f}")

    print("--- speaker count by true nspk")
    for n in (1, 2, 3, 4):
        g = [r for r in rows if r["nspk"] == n]
        if not g:
            continue
        ok = sum(1 for r in g if r["n_hyp"] == n)
        und = sum(1 for r in g if r["n_hyp"] < n)
        print(f"nspk={n} n={len(g):3d} correct {ok / len(g) * 100:5.1f}%  "
              f"under {und / len(g) * 100:5.1f}%  "
              f"over {(len(g) - ok - und) / len(g) * 100:5.1f}%")

    drop = [s for r in rows for _, s in r["dropped_spk"]]
    if drop:
        q = np.percentile(drop, [25, 50, 75, 90]).round(1)
        print(f"--- dropped speakers: {len(drop)}; speech p25/50/75/90 = {list(q)}; "
              f"<10s {np.mean([s < 10 for s in drop]):.0%}  "
              f"<20s {np.mean([s < 20 for s in drop]):.0%}")
    und_rows = [r for r in rows if r["n_hyp"] < r["nspk"] and r["nspk"] >= 2]
    ok_rows = [r for r in rows if r["n_hyp"] == r["nspk"] and r["nspk"] >= 2]
    med_min = lambda rs: float(np.median([min(r["spk_speech"].values()) for r in rs]))
    print(f"min-talker median speech: correct {med_min(ok_rows):.1f}s vs "
          f"undercount {med_min(und_rows):.1f}s; DER {pool(ok_rows)['der']:.1f} "
          f"vs {pool(und_rows)['der']:.1f}")

    print("--- ghost rate on nspk=2 windows by ovl bin")
    for b in [x[0] for x in OVL_BINS]:
        g = [r for r in rows if r["nspk"] == 2 and bucket(r["ovl"], OVL_BINS) == b]
        if g:
            gh = sum(1 for r in g if r["n_hyp"] > 2)
            print(f"{b:8s} n={len(g):3d} ghost {gh / len(g) * 100:5.1f}%")

    om = sum(r["ovl_miss"] for r in rows)
    sm = sum(r["solo_miss"] for r in rows)
    print(f"--- miss split (frame-level, no collar): overlap "
          f"{om / (om + sm) * 100:.1f}%  solo {sm / (om + sm) * 100:.1f}%")

    print("--- annotation-noise map: DER original -> curated per dataset (c=0.25)")
    for ds in datasets:
        g = [r for r in rows if r["dataset"] == ds]
        pc, po = pool(g), pool(g, **orig)
        ed = sum(r["edited"] for r in g)
        print(f"{ds:22s} edited {ed:2d}/{len(g):3d}  "
              f"{po['der']:6.2f} -> {pc['der']:6.2f}  (delta {po['der'] - pc['der']:+5.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "report"], default="score")
    args = ap.parse_args()
    if args.stage == "score":
        build_rows()
    else:
        report_stage()


if __name__ == "__main__":
    main()
