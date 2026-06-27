#!/usr/bin/env python3
"""Authoritative DER scoring with pyannote.metrics (the standard toolkit).

Loads GT + predicted RTTM, computes DER under several standard collar settings,
and prints the optimal speaker mapping + error breakdown.
"""
import argparse
import json

from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def load_rttm_as_annotation(path, uri):
    ann = Annotation(uri=uri)
    for line in open(path):
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur = float(p[3]), float(p[4])
        ann[Segment(start, start + dur)] = p[7]
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/workspace/speaker_diarization_benchmark/Alimeeting/Far/rttms/R8005_M8009.rttm")
    ap.add_argument("--pred", default="/workspace/sortformer_diar/out/R8005_M8009.pred.rttm")
    ap.add_argument("--uri", default="R8005_M8009")
    args = ap.parse_args()

    ref = load_rttm_as_annotation(args.gt, args.uri)
    hyp = load_rttm_as_annotation(args.pred, args.uri)
    print(f"ref speakers: {sorted(ref.labels())}  ({ref.get_timeline().duration():.1f}s talk)")
    print(f"hyp speakers: {sorted(hyp.labels())}  ({hyp.get_timeline().duration():.1f}s talk)\n")

    settings = [
        ("collar=0.0,  overlap scored (strict)", dict(collar=0.0, skip_overlap=False)),
        ("collar=0.25, overlap scored",          dict(collar=0.25, skip_overlap=False)),
        ("collar=0.50, overlap scored (~md-eval -c 0.25)", dict(collar=0.50, skip_overlap=False)),
        ("collar=0.25, overlap SKIPPED",         dict(collar=0.25, skip_overlap=True)),
    ]

    print(f"{'setting':<48} {'DER%':>7} {'Miss%':>7} {'FA%':>7} {'Conf%':>7}")
    print("-" * 80)
    results = {}
    for name, kw in settings:
        m = DiarizationErrorRate(**kw)
        c = m(ref, hyp, detailed=True)
        total = c["total"]
        der = c["diarization error rate"] * 100
        miss = c["missed detection"] / total * 100
        fa = c["false alarm"] / total * 100
        conf = c["confusion"] / total * 100
        results[name] = dict(der=round(der, 2), miss=round(miss, 2),
                             fa=round(fa, 2), conf=round(conf, 2),
                             total_ref_s=round(total, 1))
        print(f"{name:<48} {der:7.2f} {miss:7.2f} {fa:7.2f} {conf:7.2f}")

    jer = JaccardErrorRate(collar=0.25)(ref, hyp) * 100
    print(f"\nJaccard Error Rate (JER, collar=0.25): {jer:.2f}%")

    der_map = DiarizationErrorRate(collar=0.25)
    mapping = der_map.optimal_mapping(ref, hyp)
    print("\noptimal mapping (hyp -> ref):")
    for h, r in mapping.items():
        print(f"  {h}  ->  {r}")

    json.dump({"results": results, "jer": round(jer, 2),
               "mapping": {h: r for h, r in mapping.items()}},
              open("/workspace/sortformer_diar/out/R8005_M8009.pyannote.json", "w"),
              indent=2)
    print("\nsaved -> out/R8005_M8009.pyannote.json")


if __name__ == "__main__":
    main()
