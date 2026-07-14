#!/usr/bin/env python3
"""Windowed speaker-diarization evaluation with pooled DER.

Backend: pyannote.metrics (the standard toolkit). This module is meant to be
imported by your own evaluation code.

Why windowed / pooled?
----------------------
A segment/streaming model emits a separate prediction per window, and the
speaker ids inside each window are only *locally* consistent. So we:

  1. cut the long reference (ground truth) into windows,
  2. score each window's prediction against that window's reference,
     optimising the speaker mapping INDEPENDENTLY per window,
  3. pool the raw error / reference durations across all windows.

    pooled_DER = sum_w (miss_w + false_alarm_w + confusion_w)
                 ---------------------------------------------
                            sum_w reference_speech_w

reference = ground truth, hypothesis = model prediction.

Public API
----------
    load_rttm(path, uri="file") -> Annotation
    make_windows(duration, window_sec, hop_sec=None, start=0.0) -> list[(s, e)]
    evaluate(reference, hypothesis, windows=None, *, window_sec=30.0,
             hop_sec=None, duration=None, collar=0.0, skip_overlap=False,
             uri="file") -> PooledResult

`hypothesis` accepts either form:
    * one long prediction  (str | Path | Annotation)         -> cropped per window
    * one prediction per window (Sequence[str|Path|Annotation]) -> len == len(windows)

Examples
--------
    from diar_eval import evaluate

    # (a) your segment model: one prediction per window
    windows = [(0, 30), (30, 60), (60, 90)]
    hyps    = [model_predict(wav, s, e) for (s, e) in windows]   # rttm paths / Annotations
    res = evaluate("gt.rttm", hyps, windows=windows, collar=0.0)
    print(res.der)                       # pooled DER (%)

    # (b) one long prediction, auto-windowed at 30 s
    res = evaluate("gt.rttm", "pred.rttm", window_sec=30, collar=0.0)
    for w in res.windows:
        print(w.start, w.end, w.der)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

__all__ = ["load_rttm", "make_windows", "evaluate", "evaluate_collection",
           "WindowResult", "PooledResult"]

AnnotationLike = Union[str, Path, Annotation]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_rttm(path: Union[str, Path], uri: str = "file") -> Annotation:
    """Parse an RTTM file into a pyannote Annotation (zero/negative-dur lines skipped)."""
    ann = Annotation(uri=uri)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8 or p[0] != "SPEAKER":
                continue
            try:
                start, dur = float(p[3]), float(p[4])
            except ValueError:
                continue
            if dur <= 0:
                continue
            ann[Segment(start, start + dur)] = p[7]
    return ann


def _as_annotation(x: AnnotationLike, uri: str = "file") -> Annotation:
    return x if isinstance(x, Annotation) else load_rttm(x, uri=uri)


def make_windows(
    duration: float,
    window_sec: Optional[float],
    hop_sec: Optional[float] = None,
    start: float = 0.0,
) -> List[Tuple[float, float]]:
    """Tile [start, duration) into windows.

    window_sec None / <=0 / inf  -> a single window covering the whole range.
    hop_sec None                 -> = window_sec (non-overlapping, contiguous).
    The last window is clipped to `duration`.
    """
    if not window_sec or window_sec <= 0 or window_sec == float("inf"):
        return [(start, duration)]
    hop = hop_sec or window_sec
    wins: List[Tuple[float, float]] = []
    t = start
    while t < duration - 1e-6:
        wins.append((t, min(t + window_sec, duration)))
        t += hop
    return wins or [(start, duration)]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class WindowResult:
    start: float
    end: float
    reference: float          # reference (GT) speech seconds in the window
    miss: float               # missed-detection seconds
    false_alarm: float        # false-alarm seconds
    confusion: float          # speaker-confusion seconds
    error: float              # miss + false_alarm + confusion
    der: Optional[float]      # window DER (%) = error / reference * 100 (None if reference==0)
    mapping: Dict[str, str]   # optimal hyp_label -> ref_label, within this window


@dataclass
class PooledResult:
    der: float                # pooled DER (%) = sum(error) / sum(reference) * 100
    reference: float
    miss: float
    false_alarm: float
    confusion: float
    error: float
    miss_pct: float           # sum(miss) / sum(reference) * 100
    fa_pct: float
    conf_pct: float
    n_windows: int
    collar: float
    skip_overlap: bool
    windows: List[WindowResult] = field(default_factory=list)

    def to_dict(self, include_windows: bool = True) -> dict:
        d = asdict(self)
        if not include_windows:
            d.pop("windows")
        return d


# --------------------------------------------------------------------------- #
# Core scoring
# --------------------------------------------------------------------------- #
def _score_window(
    reference: Annotation,
    hypothesis: Annotation,
    region: Segment,
    collar: float,
    skip_overlap: bool,
) -> WindowResult:
    ref_w = reference.crop(region, mode="intersection")
    hyp_w = hypothesis.crop(region, mode="intersection")
    # `collar` is per-side (md-eval convention, ±collar around each reference
    # boundary); pyannote's parameter is the TOTAL no-score window centred on
    # the boundary, so it gets twice the user-facing value.
    metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip_overlap)
    c = metric(ref_w, hyp_w, detailed=True)
    miss = float(c["missed detection"])
    fa = float(c["false alarm"])
    conf = float(c["confusion"])
    ref = float(c["total"])
    error = miss + fa + conf
    try:
        mapping = dict(metric.optimal_mapping(ref_w, hyp_w))
    except Exception:
        mapping = {}
    der = round(error / ref * 100, 2) if ref > 0 else None
    return WindowResult(
        start=round(region.start, 3), end=round(region.end, 3),
        reference=ref, miss=miss, false_alarm=fa, confusion=conf,
        error=error, der=der, mapping=mapping,
    )


def evaluate(
    reference: AnnotationLike,
    hypothesis: Union[AnnotationLike, Sequence[AnnotationLike]],
    windows: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    window_sec: float = 30.0,
    hop_sec: Optional[float] = None,
    duration: Optional[float] = None,
    collar: float = 0.0,
    skip_overlap: bool = False,
    uri: str = "file",
) -> PooledResult:
    """Window the reference, score each window, and pool into a single DER.

    See the module docstring for the pooled-DER definition and `hypothesis`
    accepted forms. Speaker mapping is optimised independently per window.

    Note on collar: `collar` is per-side, md-eval style — ±collar seconds
    around each reference boundary are excluded (pyannote receives 2*collar,
    since its parameter is the total window). With collar > 0, window
    boundaries introduce extra no-score collars around the artificial edges.
    For pooled windowed DER, collar=0 (the default) is the cleanest; pass a
    collar only if you know you want it.
    """
    reference = _as_annotation(reference, uri)
    per_window_hyp = isinstance(hypothesis, (list, tuple))

    if per_window_hyp:
        if windows is None:
            raise ValueError("`windows` is required when `hypothesis` is a per-window list.")
        if len(hypothesis) != len(windows):
            raise ValueError(
                f"len(hypothesis)={len(hypothesis)} != len(windows)={len(windows)}"
            )
        hyps = [_as_annotation(h, uri) for h in hypothesis]
    else:
        hyp_ann = _as_annotation(hypothesis, uri)
        if windows is None:
            if duration is None:
                ref_ext = reference.get_timeline().extent()
                hyp_ext = hyp_ann.get_timeline().extent()
                ends = [s.end for s in (ref_ext, hyp_ext) if s]
                duration = max(ends) if ends else 0.0
            windows = make_windows(duration, window_sec, hop_sec)
        hyps = [hyp_ann] * len(windows)

    results = [
        _score_window(reference, hyp, Segment(ws, we), collar, skip_overlap)
        for (ws, we), hyp in zip(windows, hyps)
    ]

    return _pool(results, collar, skip_overlap)


def _pool(results: List[WindowResult], collar: float, skip_overlap: bool) -> PooledResult:
    """Pool raw error / reference durations across units: sum(error)/sum(reference)."""
    tot_ref = sum(r.reference for r in results)
    tot_miss = sum(r.miss for r in results)
    tot_fa = sum(r.false_alarm for r in results)
    tot_conf = sum(r.confusion for r in results)
    tot_err = tot_miss + tot_fa + tot_conf
    der = (tot_err / tot_ref * 100) if tot_ref > 0 else 0.0
    return PooledResult(
        der=round(der, 2),
        reference=round(tot_ref, 2),
        miss=round(tot_miss, 2),
        false_alarm=round(tot_fa, 2),
        confusion=round(tot_conf, 2),
        error=round(tot_err, 2),
        miss_pct=round(tot_miss / tot_ref * 100, 2) if tot_ref else 0.0,
        fa_pct=round(tot_fa / tot_ref * 100, 2) if tot_ref else 0.0,
        conf_pct=round(tot_conf / tot_ref * 100, 2) if tot_ref else 0.0,
        n_windows=len(results),
        collar=collar,
        skip_overlap=skip_overlap,
        windows=results,
    )


def evaluate_collection(
    pairs,
    *,
    collar: float = 0.0,
    skip_overlap: bool = False,
    uri: str = "file",
) -> PooledResult:
    """Pool DER across whole-file (reference, hypothesis) pairs.

    Each pair is scored over its full extent with its OWN optimal speaker
    mapping, then raw error/reference durations are pooled across the pairs
    (pooled_DER = sum(error)/sum(reference)). This is the dataset-level DER for
    a collection of recordings. `pairs` is an iterable of (reference, hypothesis)
    where each side is an rttm path or an Annotation. The returned PooledResult's
    `windows` list holds one WindowResult per file.
    """
    results = []
    for ref, hyp in pairs:
        ref_a = _as_annotation(ref, uri)
        hyp_a = _as_annotation(hyp, uri)
        exts = [e for e in (ref_a.get_timeline().extent(),
                            hyp_a.get_timeline().extent()) if e]
        hi = max((e.end for e in exts), default=0.0)
        region = Segment(0.0, max(hi, 1e-3))
        results.append(_score_window(ref_a, hyp_a, region, collar, skip_overlap))
    return _pool(results, collar, skip_overlap)


# --------------------------------------------------------------------------- #
# Reporting / CLI
# --------------------------------------------------------------------------- #
def print_report(res: PooledResult, show_windows: int = 8) -> None:
    print(f"pooled DER = {res.der}%   "
          f"(miss {res.miss_pct}% / FA {res.fa_pct}% / conf {res.conf_pct}%)")
    print(f"  error={res.error:.1f}s  reference={res.reference:.1f}s  "
          f"windows={res.n_windows}  collar={res.collar}  skip_overlap={res.skip_overlap}")
    if show_windows and res.windows:
        print(f"\n  {'window':>16}  {'ref_s':>7}  {'DER%':>7}  {'miss':>6}  {'FA':>6}  {'conf':>6}")
        for w in res.windows[:show_windows]:
            der = "  n/a" if w.der is None else f"{w.der:6.2f}"
            print(f"  {w.start:6.1f}-{w.end:6.1f}  {w.reference:7.1f}  {der}  "
                  f"{w.miss:6.1f}  {w.false_alarm:6.1f}  {w.confusion:6.1f}")
        if len(res.windows) > show_windows:
            print(f"  ... (+{len(res.windows) - show_windows} more windows)")


def main():
    ap = argparse.ArgumentParser(description="Windowed pooled-DER diarization eval.")
    ap.add_argument("--ref", default="/workspace/speaker_diarization_benchmark/Alimeeting/Far/rttms/R8005_M8009.rttm",
                    help="ground-truth RTTM")
    ap.add_argument("--hyp", default="/workspace/sortformer_diar/out/R8005_M8009.pred.rttm",
                    help="prediction RTTM (single long file)")
    ap.add_argument("--window", type=float, default=30.0, help="window length (s); 0 = whole file")
    ap.add_argument("--hop", type=float, default=None, help="hop (s); default = window")
    ap.add_argument("--collar", type=float, default=0.0)
    ap.add_argument("--skip-overlap", action="store_true")
    ap.add_argument("--uri", default="file")
    ap.add_argument("--json", default=None, help="optional path to dump full result as JSON")
    args = ap.parse_args()

    res = evaluate(args.ref, args.hyp, window_sec=args.window, hop_sec=args.hop,
                   collar=args.collar, skip_overlap=args.skip_overlap, uri=args.uri)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=2))
        print(f"\nsaved -> {args.json}")


if __name__ == "__main__":
    main()
