#!/usr/bin/env python3
"""Generate the human review page for VAD boundary calibration.

For every directory under out/curation/final that has wavs/, rttms/ and
rttms_vad/ (the Silero-VAD calibrated references), writes vad_review.html.
The editable lanes are initialized from the VAD-calibrated segments
(rttms_vad); the curated human reference (rttms) is shown above them as a
read-only comparison lane in the same colors. Editing interactions are the
same as review.html (drag to move, drag edges to resize, double-click to add,
Delete to remove, per-window revert-to-VAD / restore-original buttons).

Edits persist in localStorage; the export button downloads
<dir>_vad_edits.json for apply_vad_edits.py.
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

from curate_eval_set import load_rttm_segments
from curation_review import wave_peaks

FINAL_DIR = SD_EVAL / "out/curation/final"


def window_dirs(dataset_dir: Path):
    """Yield every directory that directly contains wavs/ and rttms/."""
    if (dataset_dir / "wavs").is_dir():
        yield dataset_dir
        return
    for sub in sorted(dataset_dir.iterdir()):
        if (sub / "wavs").is_dir():
            yield sub

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
.card.edited { border-color:#c9a227; }
.card.done { border-color:#3f8f5f; }
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
.lane .lbl { width:110px; flex:none; font-size:12px; color:#aab2c0;
  display:flex; align-items:center; justify-content:flex-end; gap:2px;
  padding-right:6px; font-family:ui-monospace,monospace; overflow:hidden; white-space:nowrap;
  position:sticky; left:0; z-index:3; background:#1b1e24; }
.lane .lbl .pfx { color:#6b7280; flex:none; }
.lane .lbl input { flex:1; min-width:0; background:#12151a; border:1px solid #2a2e36; color:#aab2c0;
  border-radius:4px; font-size:11px; padding:1px 4px; text-align:right; }
.lane .track { position:relative; flex:1; height:22px; background:#12151a; border-radius:4px; }
.lane.pred .seg { opacity:.55; }
.seg { position:absolute; top:1px; bottom:1px; border-radius:3px; opacity:.85; }
.lane.ref .seg { cursor:grab; }
.seg.sel { outline:2px solid #fff; opacity:1; }
.seg .h { position:absolute; top:0; bottom:0; width:7px; cursor:ew-resize; }
.seg .h.l { left:-2px; } .seg .h.r { right:-2px; }
.playhead { position:absolute; top:0; bottom:0; width:1px; background:#ff5c5c; left:110px; pointer-events:none; z-index:5; }
.rowbtns { margin-top:6px; display:flex; gap:8px; align-items:center; }
.tag { font-size:11px; padding:1px 8px; border-radius:10px; background:#2b3442; }
.tag.done { background:#2b5c3c; }
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
const LSKEY = "vadcal_" + DATASET;
const RO_PREFIX = "ref:", ED_PREFIX = "vad:";
const COLORS = ["#5b8ff9","#5ad8a6","#f6bd16","#e8684a","#6dc8ec","#9270ca","#ff9d4d","#78d3f8"];

let store = JSON.parse(localStorage.getItem(LSKEY) || "{}");
function save() { localStorage.setItem(LSKEY, JSON.stringify(store)); updProgress(); }
function stateOf(w) {
  if (!store[w.id]) store[w.id] = { segments: w.vad.map(s=>s.slice()), note:"", edited:false, done:false };
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
  for (const w of WINDOWS) root.appendChild(buildCard(w));
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
      <span class="stat">原始 ${w.ref.length} 段 → VAD ${w.vad.length} 段 · 无语音保留 ${w.n_flag} 段</span>
      <span class="tag" data-role="tag"></span>
      <button data-role="done"></button>
      <input type="text" placeholder="备注..." value="${st.note.replace(/"/g,'&quot;')}">
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
      <button data-role="addspk">+ 说话人</button>
      <button data-role="revert">还原为 VAD 结果</button>
      <button data-role="orig">还原为原始标注</button>
      <button data-role="zoomout">缩放 −</button>
      <span data-role="zoomlbl" class="stat">1×</span>
      <button data-role="zoomin">缩放 +</button>
      <span class="hint">ref: = 人工审查后的原始标注(只读,半透明) · vad: = VAD 校准结果(可编辑) · 同名说话人相邻同色,未匹配的排在下方</span>
    </div>`;
  el.querySelector("input[type=text]").oninput = e => { st.note = e.target.value; markEdited(card); };
  el.querySelector("[data-role=done]").onclick = () => {
    st.done = !st.done; save(); paintCard(card); };
  el.querySelector("[data-role=addspk]").onclick = () => {
    let i = 1; const names = new Set(st.segments.map(s=>s[2]));
    while (names.has("spk_new" + i)) i++;
    st.segments.push([0, 2, "spk_new" + i]); markEdited(card); paintCard(card); };
  el.querySelector("[data-role=revert]").onclick = () => {
    store[w.id] = { segments: w.vad.map(s=>s.slice()), note: st.note, edited:false, done:st.done };
    save(); paintCard(card); };
  el.querySelector("[data-role=orig]").onclick = () => {
    store[w.id] = { segments: w.ref.map(s=>s.slice()), note: st.note, edited:true, done:st.done };
    save(); paintCard(card); };
  const audio = el.querySelector("audio");
  // Native seeking needs HTTP Range support, which simple static servers
  // (e.g. python -m http.server) lack; fetch the wav into a blob URL once
  // so all subsequent seeks are local and always work.
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
  paintCard(card);
  return el;
}

function markEdited(card) {
  const st = stateOf(card.w); st.edited = true; save();
  card.el.classList.add("edited");
}

function paintCard(card) {
  const { w, el } = card, st = stateOf(w);
  el.classList.toggle("edited", st.edited);
  el.classList.toggle("done", st.done);
  const tag = el.querySelector("[data-role=tag]");
  tag.textContent = st.done ? "已确认" : (st.edited ? "已修改" : "VAD 默认");
  tag.className = "tag" + (st.done ? " done" : "");
  el.querySelector("[data-role=done]").textContent = st.done ? "取消确认" : "确认本窗";
  paintRuler(card);
  drawWave(card);
  paintAllLanes(card);
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

function groupBy(segs) {
  const m = new Map();
  for (const s of segs) { if (!m.has(s[2])) m.set(s[2], []); m.get(s[2]).push(s); }
  return m;
}

// Matched speakers (same name on both sides) render as adjacent pairs in the
// same color: read-only lane first, editable lane right below. Speakers that
// exist on only one side follow after all matched pairs.
function paintAllLanes(card) {
  const { w, el } = card, st = stateOf(w);
  const ro = groupBy(w.ref);
  const ed = groupBy(st.segments);
  const matched = [...ed.keys()].filter(s => ro.has(s)).sort();
  const edOnly = [...ed.keys()].filter(s => !ro.has(s)).sort();
  const roOnly = [...ro.keys()].filter(s => !ed.has(s)).sort();
  const rows = [];
  for (const s of matched) {
    rows.push({ spk: s, segs: ro.get(s), editable: false });
    rows.push({ spk: s, segs: ed.get(s), editable: true });
  }
  for (const s of edOnly) rows.push({ spk: s, segs: ed.get(s), editable: true });
  for (const s of roOnly) rows.push({ spk: s, segs: ro.get(s), editable: false });
  if (!ed.size) rows.push({ spk: "(空)", segs: [], editable: true });
  const root = el.querySelector("[data-role=lanes]");
  root.innerHTML = "";
  for (const r of rows) root.appendChild(buildLane(card, r));
}

function buildLane(card, { spk, segs, editable }) {
  const { w } = card, st = stateOf(w);
  const lane = document.createElement("div");
  lane.className = "lane " + (editable ? "ref" : "pred");
  const lbl = document.createElement("div"); lbl.className = "lbl";
  const pfx = document.createElement("span"); pfx.className = "pfx";
  pfx.textContent = editable ? ED_PREFIX : RO_PREFIX;
  lbl.appendChild(pfx);
  if (editable && spk !== "(空)") {
    const inp = document.createElement("input"); inp.value = spk;
    inp.onchange = () => {
      for (const s of st.segments) if (s[2] === spk) s[2] = inp.value;
      markEdited(card); paintCard(card); };
    lbl.appendChild(inp);
  } else {
    const t = document.createElement("span"); t.textContent = spk;
    lbl.appendChild(t);
  }
  lane.appendChild(lbl);
  const track = document.createElement("div"); track.className = "track";
  lane.appendChild(track);
  for (const seg of segs) {
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
  return lane;
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
  let edited = 0, done = 0;
  for (const w of WINDOWS) { const s = store[w.id];
    if (s && s.edited) edited++; if (s && s.done) done++; }
  document.getElementById("progress").textContent =
    `${total} 窗 · 已修改 ${edited} · 已确认 ${done}`;
}

function exportEdits() {
  const out = {};
  for (const w of WINDOWS) {
    const s = stateOf(w);
    out[w.id] = { edited: s.edited, done: s.done, note: s.note,
                  segments: s.segments.map(x => x.slice()) };
  }
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = DATASET.replaceAll("/", "_") + "_vad_edits.json";
  a.click();
}
function resetAll() {
  if (!confirm("清空本数据集所有本地修改?")) return;
  store = {}; save(); cards.length = 0; render();
}
render();
requestAnimationFrame(() => { for (const c of cards) { paintRuler(c); drawWave(c); } });
</script>
</body>
</html>
"""


def flag_counts():
    """(dir, wid) -> number of no-speech segments kept by the calibrator."""
    counts = {}
    changes_path = FINAL_DIR / "vad_calibration_changes.jsonl"
    for line in open(changes_path):
        c = json.loads(line)
        if c["flag"]:
            key = (c["dir"], c["wid"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    flags = flag_counts()
    for ds in sorted(d.name for d in FINAL_DIR.iterdir() if d.is_dir()):
        for d in window_dirs(FINAL_DIR / ds):
            if not (d / "rttms_vad").is_dir():
                continue
            rel = str(d.relative_to(FINAL_DIR))
            windows = []
            for vad_rttm in sorted((d / "rttms_vad").glob("*.rttm")):
                wid = vad_rttm.stem
                wav = d / "wavs" / f"{wid}.wav"
                info = sf.info(str(wav))
                windows.append({
                    "id": wid,
                    "dur": round(info.frames / info.samplerate, 3),
                    "ref": [[a, b, s] for a, b, s in
                            load_rttm_segments(d / "rttms" / f"{wid}.rttm")],
                    "vad": [[a, b, s] for a, b, s in load_rttm_segments(vad_rttm)],
                    "n_flag": flags.get((rel, wid), 0),
                    "peaks": wave_peaks(wav),
                })
            page = (PAGE_TEMPLATE
                    .replace("__TITLE__", html.escape(
                        f"{rel} VAD 边界校准 ({len(windows)} 窗)"))
                    .replace("__DATASET__", rel)
                    .replace("__DATA__", json.dumps(windows, ensure_ascii=False)))
            out = d / "vad_review.html"
            out.write_text(page)
            print(f"[vad-review] {rel}: {len(windows)} windows -> {out}", flush=True)


if __name__ == "__main__":
    main()
