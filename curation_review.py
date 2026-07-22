#!/usr/bin/env python3
"""Generate the human review / annotation-correction page for each eval set.

Reads out/curation/manifest.json plus the exported per-window clips and writes
out/curation/<dataset>/review.html. Each card shows the audio player, the v1
prediction lanes (read-only, colored by the window-optimal mapping to the
reference) and editable reference lanes (drag to move, drag edges to resize,
double-click a lane to add a segment, select + Delete to remove, rename or add
speakers, mark the whole window keep/discard, free-text note).

Edits persist in localStorage; the export button downloads
<dataset>_edits.json for apply_curation_edits.py.

Audio is referenced by relative path (wavs/<id>.wav), not embedded — open the
page via `python -m http.server` in out/curation/<dataset>/ or directly from
the filesystem.
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SD_EVAL = Path("/workspace/sd_evaluation")
if str(SD_EVAL) not in sys.path:
    sys.path.insert(0, str(SD_EVAL))

import diar_eval
from curate_eval_set import OUT_DIR, MANIFEST_PATH, load_rttm_segments, window_id

BIN_TITLES = {"der<10": "DER < 10%", "der10-20": "DER 10–20%",
              "der20-30": "DER 20–30%", "der>=30": "DER ≥ 30%"}
PEAKS_HZ = 100  # waveform bins per second embedded in the page (10 ms)


def wave_peaks(wav_path, hz=PEAKS_HZ):
    """Per-bin min/max peaks as base64 int8, normalised to the window peak."""
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    step = max(round(sr / hz), 1)
    n = int(np.ceil(len(data) / step))
    data = np.pad(data, (0, n * step - len(data)))
    d = data.reshape(n, step)
    scale = np.abs(d).max() or 1.0
    out = np.empty(2 * n, dtype=np.int8)
    out[0::2] = np.clip(np.round(d.min(axis=1) / scale * 127), -127, 127)
    out[1::2] = np.clip(np.round(d.max(axis=1) / scale * 127), -127, 127)
    return base64.b64encode(out.tobytes()).decode()


def build_window_payload(ds_dir, w):
    wid = window_id(w)
    dur = round(w["t1"] - w["t0"], 3)
    ref = load_rttm_segments(ds_dir / "rttms" / f"{wid}.rttm")
    pred = load_rttm_segments(ds_dir / "preds" / f"{wid}.rttm")
    # Window-optimal mapping so prediction lanes share the reference colors.
    res = diar_eval.evaluate(str(ds_dir / "rttms" / f"{wid}.rttm"),
                             str(ds_dir / "preds" / f"{wid}.rttm"),
                             windows=[(0.0, dur)], collar=0.25)
    mapping = res.windows[0].mapping if res.windows else {}
    pred = [(a, b, mapping.get(spk, spk)) for a, b, spk in pred]
    return {"id": wid, "dur": dur, "der": w["der"], "miss": w["miss"],
            "fa": w["fa"], "conf": w["conf"], "ref_sec": w["ref_sec"],
            "nspk": w["n_ref_speakers"],
            "ref": [[a, b, s] for a, b, s in ref],
            "pred": [[a, b, s] for a, b, s in pred],
            "peaks": wave_peaks(ds_dir / "wavs" / f"{wid}.wav"),
            "peaks_hz": PEAKS_HZ}


PAGE_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background:#14161a; color:#dde3ea; font:14px/1.5 system-ui,sans-serif; margin:0; padding:20px 28px 120px; }
h1 { font-size:20px; } h2 { font-size:16px; margin:28px 0 10px; color:#9fb4ff; }
.topbar { position:sticky; top:0; z-index:50; background:#14161acc; backdrop-filter:blur(6px);
  padding:8px 0; display:flex; gap:14px; align-items:center; border-bottom:1px solid #2a2e36; }
button { background:#2b3442; color:#dde3ea; border:1px solid #3d4756; border-radius:6px;
  padding:5px 12px; cursor:pointer; font-size:13px; }
button:hover { background:#38445a; }
.card { background:#1b1e24; border:1px solid #2a2e36; border-radius:10px; padding:14px 16px; margin:14px 0; }
.card.discarded { opacity:.45; }
.card.edited { border-color:#c9a227; }
.hdr { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin-bottom:8px; }
.hdr .wid { font-weight:600; font-family:ui-monospace,monospace; }
.stat { color:#8b93a1; font-size:12.5px; }
.hdr input[type=text] { flex:1; min-width:180px; background:#12151a; border:1px solid #2a2e36;
  color:#dde3ea; border-radius:6px; padding:4px 8px; font-size:13px; }
audio { width:100%; height:32px; margin:4px 0 8px; }
.scrollwrap { overflow-x:auto; overflow-y:hidden; }
.lanes { position:relative; user-select:none; min-width:100%; }
.ruler { position:relative; height:16px; margin-left:110px; cursor:pointer; }
.ruler span { position:absolute; top:0; font-size:10px; color:#6b7280; transform:translateX(-50%); }
.lane { display:flex; align-items:center; height:26px; margin:2px 0; }
.lane.wave { height:48px; }
.lane.wave .track { height:44px; cursor:pointer; }
.lane.wave canvas { position:absolute; inset:0; width:100%; height:100%; }
/* Width includes the 6px padding (border-box): tracks must start exactly at
   110px to line up with the ruler margin and the playhead offset. */
.lane .lbl { width:110px; flex:none; font-size:12px; color:#aab2c0; text-align:right;
  padding-right:6px; font-family:ui-monospace,monospace; overflow:hidden; white-space:nowrap;
  position:sticky; left:0; z-index:3; background:#1b1e24; }
.lane .lbl input { width:100%; background:#12151a; border:1px solid #2a2e36; color:#aab2c0;
  border-radius:4px; font-size:11px; padding:1px 4px; text-align:right; }
.lane .track { position:relative; flex:1; height:22px; background:#12151a; border-radius:4px; }
.lane.pred .track { height:14px; }
.seg { position:absolute; top:1px; bottom:1px; border-radius:3px; opacity:.85; }
.lane.ref .seg { cursor:grab; }
.seg.sel { outline:2px solid #fff; opacity:1; }
.seg .h { position:absolute; top:0; bottom:0; width:7px; cursor:ew-resize; }
.seg .h.l { left:-2px; } .seg .h.r { right:-2px; }
.playhead { position:absolute; top:0; bottom:0; width:1px; background:#ff5c5c; left:110px; pointer-events:none; z-index:5; }
.rowbtns { margin-top:6px; display:flex; gap:8px; align-items:center; }
.tag { font-size:11px; padding:1px 8px; border-radius:10px; background:#2b3442; }
.tag.discard { background:#5c2b2b; }
.hint { color:#6b7280; font-size:12px; margin-top:6px; }
</style>
</head>
<body>
<div class="topbar">
  <h1 style="margin:0">__TITLE__</h1>
  <span id="progress" class="stat"></span>
  <button onclick="exportEdits()">导出修改 JSON</button>
  <button onclick="resetAll()">全部重置</button>
  <span class="stat">操作:拖片段移动 / 拖边缘伸缩 / 双击轨道空白新增 / 点选后 Delete 删除 / 点刻度尺跳播</span>
</div>
<div id="cards"></div>
<script>
const DATASET = "__DATASET__";
const WINDOWS = __DATA__;
const LSKEY = "curation_" + DATASET;
const COLORS = ["#5b8ff9","#5ad8a6","#f6bd16","#e8684a","#6dc8ec","#9270ca","#ff9d4d","#78d3f8"];

let store = JSON.parse(localStorage.getItem(LSKEY) || "{}");
function save() { localStorage.setItem(LSKEY, JSON.stringify(store)); updProgress(); }
function stateOf(w) {
  if (!store[w.id]) store[w.id] = { action:"keep", segments: w.ref.map(s=>s.slice()), note:"", edited:false };
  return store[w.id];
}
function colorFor(card, spk) {
  if (!card.cmap.has(spk)) card.cmap.set(spk, COLORS[card.cmap.size % COLORS.length]);
  return card.cmap.get(spk);
}

const cards = [];
function render() {
  const root = document.getElementById("cards");
  root.innerHTML = "";
  let curBin = null;
  for (const w of WINDOWS) {
    if (w.bin !== curBin) { curBin = w.bin;
      const h = document.createElement("h2"); h.textContent = w.binTitle; root.appendChild(h); }
    root.appendChild(buildCard(w));
  }
  updProgress();
}

function buildCard(w) {
  const st = stateOf(w);
  const el = document.createElement("div");
  el.className = "card"; el.id = "card_" + w.id;
  const card = { w, el, cmap: new Map(), sel: null, zoom: 1,
                 peaks: decodePeaks(w.peaks) };
  cards.push(card);
  el.innerHTML = `
    <div class="hdr">
      <span class="wid">${w.id}</span>
      <span class="stat">DER ${w.der.toFixed(1)}% · miss ${w.miss.toFixed(1)}s · fa ${w.fa.toFixed(1)}s · conf ${w.conf.toFixed(1)}s · ${w.nspk} spk · ref ${w.ref_sec.toFixed(0)}s</span>
      <span class="tag" data-role="tag"></span>
      <button data-role="discard"></button>
      <input type="text" placeholder="备注..." value="${st.note.replace(/"/g,'&quot;')}">
    </div>
    <audio controls preload="metadata" src="wavs/${w.id}.wav"></audio>
    <div class="scrollwrap">
    <div class="lanes">
      <div class="ruler"></div>
      <div class="lane wave"><div class="lbl">波形</div>
        <div class="track"><canvas></canvas></div></div>
      <div data-role="pred"></div>
      <div data-role="ref"></div>
      <div class="playhead"></div>
    </div>
    </div>
    <div class="rowbtns">
      <button data-role="addspk">+ 说话人</button>
      <button data-role="revert">还原本窗</button>
      <button data-role="zoomout">缩放 −</button>
      <span data-role="zoomlbl" class="stat">1×</span>
      <button data-role="zoomin">缩放 +</button>
      <span class="hint">上排细条 = v1 预测(只读参照,按窗口内最优映射与参考同色)</span>
    </div>`;
  el.querySelector("input[type=text]").oninput = e => { st.note = e.target.value; markEdited(card); };
  el.querySelector("[data-role=discard]").onclick = () => {
    st.action = st.action === "keep" ? "discard" : "keep"; save(); paintCard(card); };
  el.querySelector("[data-role=addspk]").onclick = () => {
    let i = 1; const names = new Set(st.segments.map(s=>s[2]));
    while (names.has("spk_new" + i)) i++;
    st.segments.push([0, 2, "spk_new" + i]); markEdited(card); paintCard(card); };
  el.querySelector("[data-role=revert]").onclick = () => {
    store[w.id] = { action:"keep", segments: w.ref.map(s=>s.slice()), note:"", edited:false };
    save(); paintCard(card); };
  const audio = el.querySelector("audio");
  const seekTo = t => {
    // Seeking before metadata is loaded is silently dropped by browsers.
    if (audio.readyState >= 1) { audio.currentTime = t; audio.play(); }
    else {
      audio.addEventListener("loadedmetadata",
        () => { audio.currentTime = t; audio.play(); }, { once: true });
      audio.load();
    }
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
  paintCard(card);
  return el;
}

function markEdited(card) {
  const st = stateOf(card.w); st.edited = true; save();
  card.el.classList.add("edited");
}

function paintCard(card) {
  const { w, el } = card, st = stateOf(w);
  el.classList.toggle("discarded", st.action === "discard");
  el.classList.toggle("edited", st.edited);
  const tag = el.querySelector("[data-role=tag]");
  tag.textContent = st.action === "discard" ? "已丢弃" : (st.edited ? "已修改" : "原始");
  tag.className = "tag" + (st.action === "discard" ? " discard" : "");
  el.querySelector("[data-role=discard]").textContent = st.action === "discard" ? "恢复" : "丢弃本窗";
  paintRuler(card);
  drawWave(card);
  paintLanes(card, el.querySelector("[data-role=pred]"), groupBy(w.pred), false);
  paintLanes(card, el.querySelector("[data-role=ref]"), groupBy(st.segments), true);
}

function decodePeaks(b64) {
  const raw = atob(b64);
  const a = new Int8Array(raw.length);
  for (let i = 0; i < raw.length; i++) a[i] = (raw.charCodeAt(i) << 24) >> 24;
  return a;  // interleaved [min, max] per bin
}

function paintRuler(card) {
  const { w, el } = card;
  const ruler = el.querySelector(".ruler");
  ruler.innerHTML = "";
  const px = ruler.clientWidth || 1100 * card.zoom;
  // Finest step that keeps labels >=46px apart.
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

function groupBy(segs) {
  const m = new Map();
  for (const s of segs) { if (!m.has(s[2])) m.set(s[2], []); m.get(s[2]).push(s); }
  return m;
}

function paintLanes(card, root, bySpk, editable) {
  const { w } = card, st = stateOf(w);
  root.innerHTML = "";
  const names = [...bySpk.keys()].sort();
  if (editable && !names.length) names.push("(空)");
  for (const spk of names) {
    const lane = document.createElement("div");
    lane.className = "lane " + (editable ? "ref" : "pred");
    const lbl = document.createElement("div"); lbl.className = "lbl";
    if (editable && spk !== "(空)") {
      const inp = document.createElement("input"); inp.value = spk;
      inp.onchange = () => {
        for (const s of st.segments) if (s[2] === spk) s[2] = inp.value;
        markEdited(card); paintCard(card); };
      lbl.appendChild(inp);
    } else lbl.textContent = spk;
    lane.appendChild(lbl);
    const track = document.createElement("div"); track.className = "track";
    lane.appendChild(track);
    for (const seg of bySpk.get(spk) || []) {
      const d = document.createElement("div"); d.className = "seg";
      d.style.background = colorFor(card, spk);
      positionSeg(d, seg, w.dur);
      if (editable) attachEditing(card, d, seg, track);
      track.appendChild(d);
    }
    if (editable && spk !== "(空)") track.ondblclick = ev => {
      if (ev.target !== track) return;
      const r = track.getBoundingClientRect();
      const t = (ev.clientX - r.left) / r.width * w.dur;
      st.segments.push([round3(Math.max(0, t - 1)), round3(Math.min(w.dur, t + 1)), spk]);
      markEdited(card); paintCard(card); };
    root.appendChild(lane);
  }
}

function positionSeg(d, seg, dur) {
  d.style.left = (seg[0] / dur * 100) + "%";
  d.style.width = (Math.max(seg[1] - seg[0], 0.05) / dur * 100) + "%";
}
function round3(x) { return Math.round(x * 1000) / 1000; }

function attachEditing(card, d, seg, track) {
  const { w } = card, st = stateOf(w);
  for (const side of ["l", "r"]) {
    const h = document.createElement("div"); h.className = "h " + side;
    h.onmousedown = ev => startDrag(ev, side); d.appendChild(h);
  }
  d.onmousedown = ev => { if (ev.target === d) startDrag(ev, "m"); };
  d.onclick = ev => { ev.stopPropagation(); selectSeg(card, d, seg); };
  function startDrag(ev, mode) {
    ev.preventDefault(); ev.stopPropagation();
    selectSeg(card, d, seg);
    const r = track.getBoundingClientRect();
    const t0 = seg[0], t1 = seg[1], x0 = ev.clientX;
    const onmove = e => {
      const dt = (e.clientX - x0) / r.width * w.dur;
      if (mode === "l") seg[0] = round3(Math.min(Math.max(0, t0 + dt), seg[1] - 0.05));
      else if (mode === "r") seg[1] = round3(Math.max(Math.min(w.dur, t1 + dt), seg[0] + 0.05));
      else { const len = t1 - t0;
        seg[0] = round3(Math.min(Math.max(0, t0 + dt), w.dur - len)); seg[1] = round3(seg[0] + len); }
      positionSeg(d, seg, w.dur);
    };
    const onup = () => { document.removeEventListener("mousemove", onmove);
      document.removeEventListener("mouseup", onup); markEdited(card); };
    document.addEventListener("mousemove", onmove);
    document.addEventListener("mouseup", onup);
  }
}

function selectSeg(card, d, seg) {
  for (const c of cards) if (c.sel) { c.sel.d.classList.remove("sel"); c.sel = null; }
  d.classList.add("sel"); card.sel = { d, seg };
}

document.addEventListener("keydown", ev => {
  if (ev.key !== "Delete" && ev.key !== "Backspace") return;
  if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  for (const c of cards) if (c.sel) {
    const st = stateOf(c.w);
    const i = st.segments.indexOf(c.sel.seg);
    if (i >= 0) st.segments.splice(i, 1);
    c.sel = null; markEdited(c); paintCard(c); ev.preventDefault();
  }
});

function updProgress() {
  const total = WINDOWS.length;
  let edited = 0, discarded = 0;
  for (const w of WINDOWS) { const s = store[w.id];
    if (s && s.edited) edited++; if (s && s.action === "discard") discarded++; }
  document.getElementById("progress").textContent =
    `${total} 窗 · 已修改 ${edited} · 已丢弃 ${discarded}`;
}

function exportEdits() {
  const out = {};
  for (const w of WINDOWS) {
    const s = stateOf(w);
    out[w.id] = { action: s.action, edited: s.edited, note: s.note,
                  segments: s.segments.map(x => x.slice()) };
  }
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = DATASET.replaceAll("/", "_") + "_edits.json";
  a.click();
}
function resetAll() {
  if (!confirm("清空本数据集所有本地修改?")) return;
  store = {}; save(); cards.length = 0; render();
}
render();
// Ruler/waveform need real pixel widths, which only exist after layout.
requestAnimationFrame(() => { for (const c of cards) { paintRuler(c); drawWave(c); } });
</script>
</body>
</html>
"""


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for ds, bins in manifest["datasets"].items():
        ds_dir = OUT_DIR / ds
        windows = []
        for label in ["der<10", "der10-20", "der20-30", "der>=30"]:
            for w in bins.get(label, []):
                payload = build_window_payload(ds_dir, w)
                payload["bin"] = label
                payload["binTitle"] = BIN_TITLES[label]
                windows.append(payload)
        page = (PAGE_TEMPLATE
                .replace("__TITLE__", html.escape(f"{ds} 听审校准 ({len(windows)} 窗)"))
                .replace("__DATASET__", ds)
                .replace("__DATA__", json.dumps(windows, ensure_ascii=False)))
        out = ds_dir / "review.html"
        out.write_text(page)
        print(f"[review] {ds}: {len(windows)} windows -> {out}", flush=True)


if __name__ == "__main__":
    main()
