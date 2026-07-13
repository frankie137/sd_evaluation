#!/usr/bin/env python3
"""Build a self-contained listening page for the most errorful windows.

For each target row, picks the windows with the largest Miss / Confusion / FA
error (in seconds), clips the corresponding audio, and renders one card per
case: DER stats, window attributes, an aligned reference/prediction timeline
with per-frame error strips, and an embedded audio player whose playhead is
synced to the timeline (click the timeline to seek).

Speaker mapping and collar follow the same conventions as benchmark.py /
window_attribution.py (session-level optimal mapping, per-dataset collar).
"""
from __future__ import annotations

import base64
import html as html_mod
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from pyannote.core import Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from benchmark import cfg_for
from diar_eval import load_rttm
from window_attribution import OUT_ROOT, PRED_ROOT

ROOT = Path("/workspace/sd_full_benchmark")
ROWS = ["AMI_forced_align/IHM", "AMI_forced_align/SDM",
        "Alimeeting/Far", "Alimeeting/Near", "MagicData-RAMC"]
CASES_PER_ROW = [("miss", "Miss 最高"), ("conf", "Confusion 最高"), ("fa", "FA 最高")]
MIN_REF_SEC = 20.0
FRAME = 0.01
PALETTE = ["#4269d0", "#e8710a", "#3ca951", "#d5478d", "#8862c8",
           "#00a2b8", "#b8860b", "#e03231"]
ERR_COLORS = {"miss": "#5b8fc9", "fa": "#e8a13c", "conf": "#d95f5f"}
CN = {"miss": "Miss", "fa": "FA", "conf": "Confusion"}


# --------------------------------------------------------------------------- #
# Case selection from attribution records
# --------------------------------------------------------------------------- #
def pick_cases():
    cases = {}  # (row, key, t0) -> case dict (reasons merged)
    for row in ROWS:
        recs = []
        for path in (OUT_ROOT / "records" / row).rglob("*.json"):
            recs.append(json.loads(path.read_text()))
        wins = [(r, w) for r in recs for w in r["windows"]
                if w["ref"] >= MIN_REF_SEC]
        used_sessions = set()
        for comp, reason in CASES_PER_ROW:
            for r, w in sorted(wins, key=lambda p: p[1][comp], reverse=True):
                ck = (row, r["key"], w["t0"])
                if ck in cases:                      # same window tops 2 comps
                    cases[ck]["reasons"].append(reason)
                    break
                if r["key"] in used_sessions:
                    continue
                used_sessions.add(r["key"])
                cases[ck] = {"row": row, "rec": r, "win": w,
                             "reasons": [reason], "comp": comp}
                break
    return list(cases.values())


# --------------------------------------------------------------------------- #
# Per-case detail: aligned segments + frame-level error strips
# --------------------------------------------------------------------------- #
def mask_of(segs, t0, n):
    m = np.zeros(n, dtype=bool)
    for s, e in segs:
        i0, i1 = max(0, int((s - t0) / FRAME)), min(n, int(math.ceil((e - t0) / FRAME)))
        if i1 > i0:
            m[i0:i1] = True
    return m


def intervals_of(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start * FRAME, i * FRAME))
            start = None
    if start is not None:
        out.append((start * FRAME, len(mask) * FRAME))
    return out


def case_detail(case):
    row, rec, w = case["row"], case["rec"], case["win"]
    key, t0 = rec["key"], w["t0"]
    t1 = min(t0 + rec["window"], rec["duration"])
    col = row  # collection path == row for these leaf-split rows
    ref_path = ROOT / col / "rttms" / f"{key}.rttm"
    wav_path = ROOT / col / "wavs" / f"{key}.wav"
    pred_path = PRED_ROOT / col / f"{key}.rttm"
    ref = load_rttm(ref_path, uri=key)
    hyp = load_rttm(pred_path, uri=key)
    cfg = cfg_for(row.split("/")[0])
    metric = DiarizationErrorRate(collar=cfg["collar"],
                                  skip_overlap=cfg["skip_overlap"])
    mapping = dict(metric.optimal_mapping(ref, hyp))   # hyp label -> ref label

    region = Segment(t0, t1)
    ref_segs, hyp_segs = {}, {}
    for seg, _, lb in ref.crop(region, mode="intersection").itertracks(yield_label=True):
        ref_segs.setdefault(lb, []).append((seg.start, seg.end))
    for seg, _, lb in hyp.crop(region, mode="intersection").itertracks(yield_label=True):
        hyp_segs.setdefault(lb, []).append((seg.start, seg.end))

    n = int(math.ceil((t1 - t0) / FRAME))
    ref_masks = {s: mask_of(v, t0, n) for s, v in ref_segs.items()}
    hyp_masks = {s: mask_of(v, t0, n) for s, v in hyp_segs.items()}
    nref = np.sum(list(ref_masks.values()), axis=0) if ref_masks else np.zeros(n, int)
    nhyp = np.sum(list(hyp_masks.values()), axis=0) if hyp_masks else np.zeros(n, int)
    ncorr = np.zeros(n, dtype=int)
    for h, r in mapping.items():
        if r in ref_masks and h in hyp_masks:
            ncorr += ref_masks[r] & hyp_masks[h]
    strips = {
        "miss": intervals_of(nref - nhyp > 0),
        "fa": intervals_of(nhyp - nref > 0),
        "conf": intervals_of(np.minimum(nref, nhyp) - ncorr > 0),
    }
    return {"t0": t0, "t1": t1, "wav": wav_path, "mapping": mapping,
            "ref_segs": ref_segs, "hyp_segs": hyp_segs, "strips": strips,
            "ref_order": sorted(ref_segs, key=lambda s: -sum(e - b for b, e in ref_segs[s]))}


def encode_clip(wav, t0, dur):
    with tempfile.NamedTemporaryFile(suffix=".m4a") as f:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(dur),
             "-i", str(wav), "-ac", "1", "-ar", "16000", "-c:a", "aac",
             "-b:a", "40k", "-movflags", "+faststart", f.name],
            check=True)
        return base64.b64encode(Path(f.name).read_bytes()).decode()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
X0, PLOT_W, VB_W = 140, 780, 940
LANE_H, STRIP_H = 17, 13


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def timeline_svg(d):
    t0, t1 = d["t0"], d["t1"]
    dur = t1 - t0
    x = lambda t: X0 + (t - t0) / dur * PLOT_W
    ref_order = d["ref_order"]
    # colors cover window speakers first, then globally-mapped speakers that
    # happen to be silent in this window (their slots still deserve a color)
    color = {}
    for s in ref_order + [r for r in d["mapping"].values() if r not in ref_order]:
        color.setdefault(s, PALETTE[len(color) % len(PALETTE)])
    mapped = d["mapping"]
    hyp_order = sorted(d["hyp_segs"],
                       key=lambda h: (h not in mapped,
                                      ref_order.index(mapped[h])
                                      if mapped.get(h) in ref_order else 98))
    lanes = [("参考 " + s, d["ref_segs"][s], color[s]) for s in ref_order]
    lanes += [("预测 → " + mapped[h] + ("" if mapped[h] in ref_order else " (本窗无参考)")
               if h in mapped else "预测 " + h + " (未映射槽位)",
               d["hyp_segs"][h],
               color.get(mapped.get(h), "#9aa4ae")) for h in hyp_order]
    h_lanes = len(lanes) * LANE_H
    height = 22 + h_lanes + 8 + 3 * STRIP_H + 26
    out = [f'<svg viewBox="0 0 {VB_W} {height}" data-x0="{X0}" data-xw="{PLOT_W}">']
    # time grid + labels
    step = 10 if dur <= 120 else 30
    t = math.ceil(t0 / step) * step
    while t <= t1:
        out.append(f'<line x1="{x(t):.0f}" x2="{x(t):.0f}" y1="18" y2="{height - 22}" stroke="#edf0f4"/>'
                   f'<text x="{x(t):.0f}" y="{height - 8}" font-size="10" fill="#8a94a0" text-anchor="middle">{mmss(t)}</text>')
        t += step
    y = 22
    for name, segs, c in lanes:
        is_hyp = name.startswith("预测")
        out.append(f'<text x="{X0 - 6}" y="{y + LANE_H - 6}" font-size="10.5" fill="#3c4650" text-anchor="end">{html_mod.escape(name)}</text>')
        for s, e in segs:
            out.append(f'<rect x="{x(s):.1f}" y="{y + (3 if is_hyp else 1)}" '
                       f'width="{max((e - s) / dur * PLOT_W, 0.8):.1f}" height="{LANE_H - (6 if is_hyp else 4)}" '
                       f'fill="{c}" opacity="{0.55 if is_hyp else 0.95}" rx="1.5">'
                       f'<title>{html_mod.escape(name)}  {mmss(s)}–{mmss(e)} ({e - s:.1f}s)</title></rect>')
        y += LANE_H
    y += 8
    for comp in ("miss", "fa", "conf"):
        out.append(f'<text x="{X0 - 6}" y="{y + STRIP_H - 3}" font-size="10.5" fill="{ERR_COLORS[comp]}" font-weight="600" text-anchor="end">{CN[comp]}</text>')
        for s, e in d["strips"][comp]:
            out.append(f'<rect x="{x(t0 + s):.1f}" y="{y + 1}" width="{max((e - s) / dur * PLOT_W, 0.8):.1f}" '
                       f'height="{STRIP_H - 2}" fill="{ERR_COLORS[comp]}">'
                       f'<title>{CN[comp]}  {mmss(t0 + s)}–{mmss(t0 + e)} ({e - s:.1f}s)</title></rect>')
        y += STRIP_H
    out.append(f'<line class="ph" x1="{X0}" x2="{X0}" y1="18" y2="{height - 22}" stroke="#111" stroke-width="1.4"/>')
    out.append("</svg>")
    return "".join(out)


def top_error_hints(d, comp, k=3):
    ivs = sorted(d["strips"][comp], key=lambda p: p[1] - p[0], reverse=True)[:k]
    return "、".join(f"{mmss(d['t0'] + s)}–{mmss(d['t0'] + e)}（{e - s:.1f}s）"
                    for s, e in sorted(ivs))


def case_card(case, d, audio_b64, idx):
    w, rec = case["win"], case["rec"]
    dur = d["t1"] - d["t0"]
    der = (w["miss"] + w["fa"] + w["conf"]) / w["ref"] * 100
    comp = case["comp"]
    attrs = (f'{w["nspk"]} 人活跃 · 重叠 {"–" if w["ovl"] is None else f"{w['ovl'] * 100:.0f}%"} · '
             f'平均 turn {"–" if w["turn"] is None else f"{w['turn']:.1f}s"} · '
             f'短turn×{w["nshort"]} · SNR {"–" if w["snr"] is None else f"{w['snr']:.0f}dB"} · '
             f'混响 {"–" if rec["reverb"] is None else f"{rec['reverb']:.2f}s"}')
    stats = (f'窗口 DER <b>{der:.0f}%</b>（Miss {w["miss"]:.1f}s / FA {w["fa"]:.1f}s / '
             f'Conf {w["conf"]:.1f}s，参考 {w["ref"]:.0f}s）')
    hints = top_error_hints(d, comp)
    return f'''<div class="case card" data-dur="{dur:.2f}">
  <div class="case-head"><span class="badge">{" + ".join(case["reasons"])}</span>
    <b>{html_mod.escape(rec["key"])}</b> · {html_mod.escape(case["row"])} ·
    窗口 {mmss(d["t0"])}–{mmss(d["t1"])}（会话全长 {mmss(rec["duration"])}）</div>
  <div class="case-meta">{stats}<br>{attrs}<br>
    <span class="hint">主要 {CN[comp]} 区段：{hints}（点击时间轴任意位置可跳转播放）</span></div>
  <audio controls preload="none" src="data:audio/mp4;base64,{audio_b64}"></audio>
  <div class="tl">{timeline_svg(d)}</div>
</div>'''


PAGE_HEAD = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>分离错误案例听音</title><style>
  body { margin:0; padding:24px 28px 80px; font:14px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; color:#1d2733; background:#f6f8fa; }
  h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:30px 0 10px; }
  .sub { color:#68737f; font-size:12.5px; }
  .card { background:#fff; border:1px solid #e3e8ee; border-radius:10px; padding:14px 16px; margin:12px 0; }
  .badge { background:#fdecec; color:#b3382e; border-radius:99px; padding:2px 10px; font-size:11.5px; font-weight:600; margin-right:8px; }
  .case-head { font-size:13.5px; margin-bottom:6px; }
  .case-meta { font-size:12.5px; color:#3c4650; margin-bottom:8px; }
  .hint { color:#68737f; }
  audio { width:100%; height:34px; margin-bottom:6px; }
  .tl { overflow-x:auto; } .tl svg { min-width:820px; cursor:crosshair; display:block; }
  table { border-collapse:collapse; font-size:12.5px; }
  th,td { padding:4px 10px; border-bottom:1px solid #e3e8ee; text-align:right; font-variant-numeric:tabular-nums; }
  td:first-child,th:first-child { text-align:left; }
</style></head><body>
<h1>分离错误案例听音页</h1>
<div class="sub">每个集合各选 Miss / Confusion / FA 错误秒数最高的 90s 窗口。上半部分色块为参考标注（每说话人一行）与预测（按全档最优映射对齐着色，半透明；灰色 = 未映射的多余预测槽位）；下方三条为逐帧错误区间。播放时黑色竖线跟随进度，点击时间轴跳转。</div>
'''

PAGE_JS = '''<script>
document.querySelectorAll(".case").forEach(c => {
  const audio = c.querySelector("audio"), svg = c.querySelector("svg");
  const ph = svg.querySelector(".ph"), dur = +c.dataset.dur;
  const x0 = +svg.dataset.x0, xw = +svg.dataset.xw;
  audio.addEventListener("timeupdate", () => {
    const x = x0 + Math.min(audio.currentTime / dur, 1) * xw;
    ph.setAttribute("x1", x); ph.setAttribute("x2", x);
  });
  svg.addEventListener("click", e => {
    const r = svg.getBoundingClientRect();
    const vx = (e.clientX - r.left) / r.width * svg.viewBox.baseVal.width;
    if (vx < x0) return;
    audio.currentTime = Math.max(0, Math.min((vx - x0) / xw, 1)) * dur;
    audio.play();
  });
});
</script></body></html>'''


def main():
    cases = pick_cases()
    print(f"[cases] {len(cases)} cases selected")
    sections = {row: [] for row in ROWS}
    manifest = []
    for i, case in enumerate(cases):
        d = case_detail(case)
        audio = encode_clip(d["wav"], d["t0"], d["t1"] - d["t0"])
        sections[case["row"]].append(case_card(case, d, audio, i))
        w = case["win"]
        manifest.append({"row": case["row"], "key": case["rec"]["key"],
                         "t0": d["t0"], "t1": d["t1"], "reasons": case["reasons"],
                         "miss": w["miss"], "fa": w["fa"], "conf": w["conf"],
                         "ref": w["ref"], "wav": str(d["wav"])})
        print(f"[cases] {case['row']} {case['rec']['key']} t0={d['t0']:.0f} "
              f"({'+'.join(case['reasons'])}) audio {len(audio) // 1024}KB")
    body = "".join(f"<h2>{html_mod.escape(row)}</h2>" + "".join(cards)
                   for row, cards in sections.items())
    out = OUT_ROOT / "error_cases.html"
    out.write_text(PAGE_HEAD + body + PAGE_JS)
    (OUT_ROOT / "error_cases.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[cases] saved -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
