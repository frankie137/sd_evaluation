#!/usr/bin/env python3
"""Generate a read-only viewer page for the calibrated eval set (final_vad).

For every directory under out/curation/final_vad with wavs/ and rttms/,
writes view.html showing each window: audio player, waveform, the reference
lanes paired with the v1 prediction lanes (renamed by the window-optimal
mapping, same color, adjacent). Per-window DER stats come from
final_benchmark.jsonl. No editing — viewing only.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import soundfile as sf

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

import diar_eval
from curate_eval_set import load_rttm_segments
from curation_review import wave_peaks
from vad_review import FINAL_DIR, window_dirs

FINAL_VAD_DIR = FINAL_DIR.parent / "final_vad"
PREDS_ROOT = FINAL_DIR.parent
BENCH_PATH = FINAL_VAD_DIR / "final_benchmark.jsonl"

PAGE_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background:#14161a; color:#dde3ea; font:14px/1.5 system-ui,sans-serif; margin:0; padding:20px 28px 120px; }
h1 { font-size:20px; }
.topbar { position:sticky; top:0; z-index:50; background:#14161acc; backdrop-filter:blur(6px);
  padding:8px 0; display:flex; gap:14px; align-items:center; border-bottom:1px solid #2a2e36; }
button { background:#2b3442; color:#dde3ea; border:1px solid #3d4756; border-radius:6px;
  padding:5px 12px; cursor:pointer; font-size:13px; }
button:hover { background:#38445a; }
.card { background:#1b1e24; border:1px solid #2a2e36; border-radius:10px; padding:14px 16px; margin:14px 0; }
.hdr { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin-bottom:8px; }
.hdr .wid { font-weight:600; font-family:ui-monospace,monospace; }
.stat { color:#8b93a1; font-size:12.5px; }
.tag { font-size:11px; padding:1px 8px; border-radius:10px; background:#2b3442; }
.tag.calib { background:#2b5c3c; }
audio { width:100%; height:32px; margin:4px 0 8px; }
.scrollwrap { overflow-x:auto; overflow-y:hidden; }
.lanes { position:relative; user-select:none; min-width:100%; }
.ruler { position:relative; height:16px; margin-left:110px; cursor:pointer; }
.ruler span { position:absolute; top:0; font-size:10px; color:#6b7280; transform:translateX(-50%); }
.lane { display:flex; align-items:center; height:26px; margin:2px 0; }
.lane.wave { height:48px; }
.lane.wave .track { height:44px; cursor:pointer; }
.lane.wave canvas { position:absolute; inset:0; width:100%; height:100%; }
.lane .lbl { width:110px; flex:none; font-size:12px; color:#aab2c0;
  display:flex; align-items:center; justify-content:flex-end; gap:2px;
  padding-right:6px; font-family:ui-monospace,monospace; overflow:hidden; white-space:nowrap;
  position:sticky; left:0; z-index:3; background:#1b1e24; }
.lane .lbl .pfx { color:#6b7280; flex:none; }
.lane .track { position:relative; flex:1; height:22px; background:#12151a; border-radius:4px; }
.lane.pred .seg { opacity:.55; }
.seg { position:absolute; top:1px; bottom:1px; border-radius:3px; opacity:.85; }
.playhead { position:absolute; top:0; bottom:0; width:1px; background:#ff5c5c; left:110px; pointer-events:none; z-index:5; }
.rowbtns { margin-top:6px; display:flex; gap:8px; align-items:center; }
.hint { color:#6b7280; font-size:12px; margin-top:6px; }
</style>
</head>
<body>
<div class="topbar">
  <h1 style="margin:0">__TITLE__</h1>
  <span class="stat">ref: = 最终参考标注 · hyp: = v1 预测(半透明,按窗口内最优映射改名) · 已匹配说话人相邻同色 · 点刻度尺/波形跳播</span>
</div>
<div id="cards"></div>
<script>
const WINDOWS = __DATA__;
const COLORS = ["#5b8ff9","#5ad8a6","#f6bd16","#e8684a","#6dc8ec","#9270ca","#ff9d4d","#78d3f8"];

const cards = [];
function render() {
  const root = document.getElementById("cards");
  for (const w of WINDOWS) root.appendChild(buildCard(w));
}

function buildCard(w) {
  const el = document.createElement("div");
  el.className = "card";
  const card = { w, el, cmap: new Map(), zoom: 1, peaks: decodePeaks(w.peaks) };
  cards.push(card);
  const derStat = w.der === null ? "" :
    `DER ${w.der.toFixed(1)}% · miss ${w.miss.toFixed(1)}s · fa ${w.fa.toFixed(1)}s · conf ${w.conf.toFixed(1)}s`;
  el.innerHTML = `
    <div class="hdr">
      <span class="wid">${w.id}</span>
      <span class="stat">${derStat}</span>
      <span class="tag${w.calibrated ? " calib" : ""}">${w.calibrated ? "边界已校准" : "原始标注"}</span>
    </div>
    <audio controls preload="metadata" src="wavs/${w.id}.wav"></audio>
    <div class="scrollwrap">
    <div class="lanes">
      <div class="ruler"></div>
      <div class="lane wave"><div class="lbl">波形</div>
        <div class="track"><canvas></canvas></div></div>
      <div data-role="lanes"></div>
      <div class="playhead"></div>
    </div>
    </div>
    <div class="rowbtns">
      <button data-role="zoomout">缩放 −</button>
      <span data-role="zoomlbl" class="stat">1×</span>
      <button data-role="zoomin">缩放 +</button>
    </div>`;
  const audio = el.querySelector("audio");
  // Native seeking needs HTTP Range support, which simple static servers
  // lack; fetch the wav into a blob URL once so seeks always work.
  const seekTo = async t => {
    if (!audio.dataset.blob) {
      audio.dataset.blob = "1";
      try {
        const b = await (await fetch(audio.getAttribute("src"))).blob();
        audio.src = URL.createObjectURL(b);
      } catch (e) { /* file:// etc. — keep the original src */ }
    }
    if (audio.readyState < 1) {
      audio.load();
      await new Promise(res =>
        audio.addEventListener("loadedmetadata", res, { once: true }));
    }
    audio.currentTime = t; audio.play();
  };
  const seekClick = ev => {
    const r = ev.currentTarget.getBoundingClientRect();
    seekTo((ev.clientX - r.left) / r.width * w.dur);
  };
  el.querySelector(".ruler").onclick = seekClick;
  el.querySelector(".lane.wave .track").onclick = seekClick;
  const setZoom = z => {
    card.zoom = Math.min(Math.max(z, 1), 32);
    el.querySelector("[data-role=zoomlbl]").textContent = card.zoom + "×";
    el.querySelector(".lanes").style.width = (card.zoom * 100) + "%";
    paintRuler(card); drawWave(card);
  };
  el.querySelector("[data-role=zoomin]").onclick = () => setZoom(card.zoom * 2);
  el.querySelector("[data-role=zoomout]").onclick = () => setZoom(card.zoom / 2);
  setInterval(() => {
    el.querySelector(".playhead").style.left =
      `calc(110px + (100% - 110px) * ${Math.min(audio.currentTime / w.dur, 1)})`; }, 100);
  paintRuler(card);
  drawWave(card);
  paintAllLanes(card);
  return el;
}

function colorFor(card, spk) {
  if (!card.cmap.has(spk)) card.cmap.set(spk, COLORS[card.cmap.size % COLORS.length]);
  return card.cmap.get(spk);
}

function groupBy(segs) {
  const m = new Map();
  for (const s of segs) { if (!m.has(s[2])) m.set(s[2], []); m.get(s[2]).push(s); }
  return m;
}

// Matched speakers render as adjacent pairs in the same color: hyp lane
// first, ref lane right below; one-sided speakers follow after the pairs.
function paintAllLanes(card) {
  const { w, el } = card;
  const hyp = groupBy(w.pred);
  const ref = groupBy(w.ref);
  const matched = [...ref.keys()].filter(s => hyp.has(s)).sort();
  const refOnly = [...ref.keys()].filter(s => !hyp.has(s)).sort();
  const hypOnly = [...hyp.keys()].filter(s => !ref.has(s)).sort();
  const rows = [];
  for (const s of matched) {
    rows.push({ spk: s, segs: hyp.get(s), isRef: false });
    rows.push({ spk: s, segs: ref.get(s), isRef: true });
  }
  for (const s of refOnly) rows.push({ spk: s, segs: ref.get(s), isRef: true });
  for (const s of hypOnly) rows.push({ spk: s, segs: hyp.get(s), isRef: false });
  const root = el.querySelector("[data-role=lanes]");
  for (const r of rows) root.appendChild(buildLane(card, r));
}

function buildLane(card, { spk, segs, isRef }) {
  const { w } = card;
  const lane = document.createElement("div");
  lane.className = "lane " + (isRef ? "ref" : "pred");
  const lbl = document.createElement("div"); lbl.className = "lbl";
  const pfx = document.createElement("span"); pfx.className = "pfx";
  pfx.textContent = isRef ? "ref:" : "hyp:";
  const name = document.createElement("span"); name.textContent = spk;
  lbl.appendChild(pfx); lbl.appendChild(name);
  lane.appendChild(lbl);
  const track = document.createElement("div"); track.className = "track";
  lane.appendChild(track);
  for (const seg of segs) {
    const d = document.createElement("div"); d.className = "seg";
    d.style.background = colorFor(card, spk);
    d.style.left = (seg[0] / w.dur * 100) + "%";
    d.style.width = (Math.max(seg[1] - seg[0], 0.05) / w.dur * 100) + "%";
    track.appendChild(d);
  }
  return lane;
}

function decodePeaks(b64) {
  const raw = atob(b64);
  const a = new Int8Array(raw.length);
  for (let i = 0; i < raw.length; i++) a[i] = (raw.charCodeAt(i) << 24) >> 24;
  return a;
}

function paintRuler(card) {
  const { w, el } = card;
  const ruler = el.querySelector(".ruler");
  ruler.innerHTML = "";
  const px = ruler.clientWidth || 1100 * card.zoom;
  let step = 30;
  for (const s of [30, 10, 5, 2, 1, 0.5, 0.2, 0.1])
    if (w.dur / s * 46 <= px) step = s;
  for (let i = 0; i * step <= w.dur + 1e-6; i++) {
    const t = i * step;
    const s = document.createElement("span");
    s.style.left = (t / w.dur * 100) + "%";
    s.textContent = (step < 1 ? t.toFixed(1) : t) + "s";
    ruler.appendChild(s);
  }
}

function drawWave(card) {
  const cv = card.el.querySelector(".lane.wave canvas");
  const track = cv.parentElement;
  const wpx = track.clientWidth || 1100 * card.zoom;
  const hpx = 44;
  cv.width = wpx; cv.height = hpx;
  const g = cv.getContext("2d");
  g.clearRect(0, 0, wpx, hpx);
  g.fillStyle = "#4a5f82";
  const peaks = card.peaks, n = peaks.length / 2, mid = hpx / 2, amp = hpx / 2 - 1;
  for (let x = 0; x < wpx; x++) {
    const i0 = Math.floor(x / wpx * n);
    const i1 = Math.max(Math.floor((x + 1) / wpx * n), i0 + 1);
    let mn = 127, mx = -127;
    for (let i = i0; i < i1 && i < n; i++) {
      if (peaks[2 * i] < mn) mn = peaks[2 * i];
      if (peaks[2 * i + 1] > mx) mx = peaks[2 * i + 1];
    }
    const y0 = mid - mx / 127 * amp, y1 = mid - mn / 127 * amp;
    g.fillRect(x, y0, 1, Math.max(y1 - y0, 1));
  }
  g.fillStyle = "#2a2e36";
  g.fillRect(0, mid, wpx, 1);
}

render();
requestAnimationFrame(() => { for (const c of cards) { paintRuler(c); drawWave(c); } });
</script>
</body>
</html>
"""


def load_bench():
    stats = {}
    if BENCH_PATH.exists():
        for line in open(BENCH_PATH):
            r = json.loads(line)
            stats[(r["subset"], r["wid"])] = r
    return stats


def main():
    bench = load_bench()
    calibrated_subsets = set(json.loads(
        (FINAL_VAD_DIR / "calibrated_subsets.json").read_text()))
    for ds in sorted(d.name for d in FINAL_VAD_DIR.iterdir() if d.is_dir()):
        for d in window_dirs(FINAL_VAD_DIR / ds):
            rel = str(d.relative_to(FINAL_VAD_DIR))
            calibrated = rel in calibrated_subsets
            windows = []
            for ref_path in sorted((d / "rttms").glob("*.rttm")):
                wid = ref_path.stem
                wav = d / "wavs" / f"{wid}.wav"
                info = sf.info(str(wav))
                dur = round(info.frames / info.samplerate, 3)
                pred_path = PREDS_ROOT / rel / "preds" / f"{wid}.rttm"
                pred = load_rttm_segments(pred_path)
                res = diar_eval.evaluate(str(ref_path), str(pred_path),
                                         windows=[(0.0, dur)], collar=0.25)
                mapping = res.windows[0].mapping if res.windows else {}
                pred = [(a, b, mapping.get(spk, spk)) for a, b, spk in pred]
                b = bench.get((rel, wid), {})
                windows.append({
                    "id": wid, "dur": dur, "calibrated": calibrated,
                    "der": b.get("der"), "miss": b.get("miss", 0),
                    "fa": b.get("fa", 0), "conf": b.get("conf", 0),
                    "ref": [[a, b2, s] for a, b2, s in load_rttm_segments(ref_path)],
                    "pred": [[a, b2, s] for a, b2, s in pred],
                    "peaks": wave_peaks(wav),
                })
            page = (PAGE_TEMPLATE
                    .replace("__TITLE__", html.escape(
                        f"{rel} 最终评测集 ({len(windows)} 窗)"))
                    .replace("__DATA__", json.dumps(windows, ensure_ascii=False)))
            out = d / "view.html"
            out.write_text(page)
            print(f"[view] {rel}: {len(windows)} windows -> {out}", flush=True)


if __name__ == "__main__":
    main()
