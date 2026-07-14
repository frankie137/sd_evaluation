#!/usr/bin/env python3
"""Render a single diarization sample: model prediction vs ground truth.

Produces a self-contained HTML with:
  - waveform overview (whole file) + a zoomable / scrollable detail view,
  - ground-truth speaker lanes and predicted speaker lanes on a shared timeline,
    color-matched via the optimal (overlap-maximizing) speaker mapping,
  - a stats panel (inference timing/RTF + DER scored by pyannote.metrics),
  - embedded audio for click-to-seek spot checking.
"""

import argparse
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

PEAKS_PER_SEC = 25
PALETTE = ["#58a6ff", "#f78166", "#56d364", "#d2a8ff", "#e3b341", "#ff7b72"]


def parse_rttm(path):
    segs = []
    for line in open(path):
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        try:
            start = float(p[3]); dur = float(p[4])
        except ValueError:
            continue
        segs.append((start, start + dur, p[7]))
    return segs


def compute_peaks(wav, dur):
    data, sr = sf.read(str(wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    n_bins = int(dur * PEAKS_PER_SEC)
    peaks = np.zeros(n_bins, dtype=np.float32)
    edges = np.linspace(0, len(data), n_bins + 1).astype(int)
    for i in range(n_bins):
        chunk = data[edges[i]:edges[i + 1]]
        if len(chunk):
            peaks[i] = np.abs(chunk).max()
    mx = peaks.max()
    if mx > 0:
        peaks /= mx
    return (peaks * 100).astype(int).tolist()


def embed_audio(wav, tmpdir):
    mp3 = os.path.join(tmpdir, "a.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(wav),
         "-ac", "1", "-ar", "16000", "-b:a", "32k", mp3],
        check=True,
    )
    with open(mp3, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _ann(path, uri):
    a = Annotation(uri=uri)
    for line in open(path):
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        s, d = float(p[3]), float(p[4])
        a[Segment(s, s + d)] = p[7]
    return a


def score_pa(gt_path, pred_path, uri):
    """Authoritative DER via pyannote.metrics under standard collar settings."""
    ref, hyp = _ann(gt_path, uri), _ann(pred_path, uri)

    def one(collar, skip):
        # collar is per-side (md-eval convention); pyannote takes the total window.
        c = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip)(ref, hyp, detailed=True)
        t = c["total"]
        return {
            "der": round(c["diarization error rate"] * 100, 2),
            "miss_pct": round(c["missed detection"] / t * 100, 2),
            "fa_pct": round(c["false alarm"] / t * 100, 2),
            "conf_pct": round(c["confusion"] / t * 100, 2),
            "ref_s": round(t, 1),
        }

    # optimal speaker mapping (hyp_label -> ref_label) + matched-pair overlap
    hyp2ref = DiarizationErrorRate(collar=2 * 0.25).optimal_mapping(ref, hyp)
    ref2hyp = {r: h for h, r in hyp2ref.items()}
    overlaps = {}
    for h, r in hyp2ref.items():
        inter = hyp.label_timeline(h).crop(ref.label_timeline(r), mode="intersection")
        overlaps[r] = round(inter.duration(), 1)

    return {
        "strict": one(0.0, False),     # no collar, overlap scored
        "std": one(0.25, False),       # ±0.25s per side (md-eval -c 0.25), overlap scored
        "pa025": one(0.125, False),    # ±0.125s per side (legacy pyannote-0.25 total)
        "jer": round(JaccardErrorRate(collar=2 * 0.25)(ref, hyp) * 100, 2),
        "ref_labels": sorted(ref.labels()),
        "hyp_labels": sorted(hyp.labels(),
                             key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x),
        "ref2hyp": ref2hyp,
        "overlaps": overlaps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="/workspace/speaker_diarization_benchmark/Alimeeting/Far/wavs/R8005_M8009.wav")
    ap.add_argument("--gt", default="/workspace/speaker_diarization_benchmark/Alimeeting/Far/rttms/R8005_M8009.rttm")
    ap.add_argument("--pred", default="/workspace/sortformer_diar/out/R8005_M8009.pred.rttm")
    ap.add_argument("--meta", default="/workspace/sortformer_diar/out/R8005_M8009.meta.json")
    ap.add_argument("--out", default="/workspace/sortformer_diar/out/R8005_M8009.compare.html")
    args = ap.parse_args()

    audio = Path(args.audio)
    info = sf.info(str(audio))
    dur = info.duration

    gt_segs = parse_rttm(args.gt)
    pred_segs = parse_rttm(args.pred)

    pa = score_pa(args.gt, args.pred, audio.stem)  # pyannote.metrics: DER + mapping
    gt_speakers = pa["ref_labels"]
    pred_speakers = pa["hyp_labels"]
    ref2hyp = pa["ref2hyp"]

    # colors: each GT speaker a color; its matched pred shares it
    gt_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(gt_speakers)}
    pred_color, pred_order, used = {}, [], set()
    for gs in gt_speakers:
        p = ref2hyp.get(gs)
        if p is not None:
            pred_color[p] = gt_color[gs]
            pred_order.append(p); used.add(p)
    extra_c = len(gt_speakers)
    for p in pred_speakers:
        if p not in used:
            pred_color[p] = PALETTE[extra_c % len(PALETTE)]; extra_c += 1
            pred_order.append(p)

    mapping_rows = [{"gt": gs, "pred": ref2hyp.get(gs) or "—",
                     "overlap_s": pa["overlaps"].get(gs, 0)}
                    for gs in gt_speakers]

    meta = json.loads(Path(args.meta).read_text()) if Path(args.meta).exists() else {}

    with tempfile.TemporaryDirectory() as td:
        peaks = compute_peaks(audio, dur)
        audio_b64 = embed_audio(audio, td)

    payload = {
        "uri": audio.stem,
        "file": audio.name,
        "duration": round(dur, 3),
        "model": meta.get("model", ""),
        "latency": meta.get("latency_preset", ""),
        "infer_seconds": meta.get("infer_seconds", None),
        "rtf": meta.get("rtf", None),
        "gt_lanes": [{"id": s, "color": gt_color[s]} for s in gt_speakers],
        "pred_lanes": [{"id": s, "color": pred_color[s]} for s in pred_order],
        "gt_segments": [{"s": round(a, 3), "e": round(b, 3), "spk": c} for a, b, c in gt_segs],
        "pred_segments": [{"s": round(a, 3), "e": round(b, 3), "spk": c} for a, b, c in pred_segs],
        "mapping": mapping_rows,
        "der_strict": pa["strict"],
        "der_std": pa["std"],
        "der_pa025": pa["pa025"],
        "jer": pa["jer"],
        "scorer": "pyannote.metrics 4.1",
        "peaks": peaks,
        "peaks_per_sec": PEAKS_PER_SEC,
        "audio": audio_b64,
    }

    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    Path(args.out).write_text(html, encoding="utf-8")
    size = Path(args.out).stat().st_size / 1e6
    print(f"[pyannote.metrics] DER strict(no collar)={pa['strict']['der']}%  "
          f"std(md-eval -c0.25)={pa['std']['der']}%  pa-collar0.25={pa['pa025']['der']}%  JER={pa['jer']}%")
    print(f"  std  miss={pa['std']['miss_pct']}%  fa={pa['std']['fa_pct']}%  conf={pa['std']['conf_pct']}%  ref={pa['std']['ref_s']}s")
    print("Mapping:")
    for r in mapping_rows:
        print(f"  {r['gt']}  <->  {r['pred']}   (overlap {r['overlap_s']}s)")
    print(f"\nWrote {args.out} ({size:.1f} MB)")


HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diarization: 预测 vs Ground Truth</title>
<style>
:root{--bg:#0f1419;--panel:#1a2129;--panel2:#222c37;--text:#e6edf3;--muted:#8b98a5;--accent:#58a6ff;--grid:#2d3742;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
header{padding:16px 24px;border-bottom:1px solid var(--grid)}
header h1{margin:0;font-size:18px}
header .sub{color:var(--muted);font-size:13px;margin-top:4px}
.wrap{padding:18px 24px 60px;max-width:1400px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:16px 18px;margin-bottom:18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:6px}
.stat{background:var(--panel2);border-radius:8px;padding:10px 12px}
.stat .k{color:var(--muted);font-size:11.5px}
.stat .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.stat .v small{font-size:12px;color:var(--muted);font-weight:400}
.der-big .v{color:#ffa657}
table{border-collapse:collapse;font-size:13px;margin-top:6px}
td,th{padding:5px 12px;text-align:left;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:500}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;margin-right:6px}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:10px 0 4px;font-size:12.5px;color:var(--muted)}
.controls{display:flex;align-items:center;gap:14px;margin-top:10px;flex-wrap:wrap}
.controls button{background:var(--accent);color:#06101d;border:none;border-radius:6px;padding:7px 16px;font-size:13px;font-weight:600;cursor:pointer}
.controls .time{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}
.controls select{background:var(--panel2);color:var(--text);border:1px solid var(--grid);border-radius:5px;padding:5px 8px;font-size:12.5px}
.section-title{font-size:13px;color:var(--muted);margin:4px 0 8px}
.detailwrap{overflow-x:auto;overflow-y:hidden;border-radius:6px;background:var(--panel2)}
canvas{display:block}
.overview canvas{width:100%;background:var(--panel2);border-radius:6px;cursor:pointer}
audio{display:none}
.hint{color:var(--muted);font-size:12px;margin-top:6px}
</style></head>
<body>
<header>
<h1>说话人分离 · 模型预测 vs Ground Truth</h1>
<div class="sub" id="sub"></div>
</header>
<div class="wrap">
  <div class="card" id="statcard"></div>
  <div class="card">
    <div class="section-title">总览（整段，点击跳转播放）· 上=Ground Truth，下=模型预测</div>
    <div class="overview"><canvas id="ov"></canvas></div>
    <div class="legend" id="legend"></div>
    <div class="controls">
      <button id="play">▶ 播放</button>
      <span class="time" id="time">0:00 / 0:00</span>
      <span>速度 <select id="rate"><option value="1">1×</option><option value="1.5">1.5×</option><option value="2">2×</option><option value="0.75">0.75×</option></select></span>
    </div>
  </div>
  <div class="card">
    <div class="section-title">细节视图（可缩放 / 横向滚动，播放时自动跟随）</div>
    <div class="controls" style="margin-top:0;margin-bottom:10px">
      缩放 <select id="zoom"><option value="2">2 px/s</option><option value="4" selected>4 px/s</option><option value="8">8 px/s</option><option value="16">16 px/s</option></select>
    </div>
    <div class="detailwrap" id="detailwrap"><canvas id="dt"></canvas></div>
    <div class="hint">说话人颜色：预测说话人已按最优重叠映射，与对应的 GT 说话人配成同色；颜色对得上即说明该时刻预测正确。</div>
  </div>
</div>
<script>
const D = __PAYLOAD__;
const fmt=t=>{t=Math.max(0,t);const m=Math.floor(t/60),s=t%60;return m+":"+s.toFixed(0).padStart(2,"0")};

// ---------- stats ----------
document.getElementById("sub").textContent =
  `${D.file} · 时长 ${fmt(D.duration)} · 模型 ${D.model}`;
const sc=document.getElementById("statcard");
const di=(k,v)=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`;
sc.innerHTML = `<div class="stats">
  ${di("推理耗时", (D.infer_seconds??"–")+`<small> s / ${fmt(D.duration)}</small>`)}
  ${di("RTF", (D.rtf??"–")+`<small> ≈${D.rtf?(1/D.rtf).toFixed(0):"–"}× 实时</small>`)}
  ${di("说话人 GT/预测", D.gt_lanes.length+" / "+D.pred_lanes.length)}
  <div class="stat der-big"><div class="k">DER（collar 0.25 ≈ md-eval, 含重叠）</div><div class="v">${D.der_std.der}%</div></div>
  <div class="stat der-big"><div class="k">DER（无 collar, 严格）</div><div class="v">${D.der_strict.der}%</div></div>
  ${di("DER（pyannote collar0.25）", D.der_pa025.der+"%")}
  ${di("JER", D.jer+"%")}
</div>
<table>
  <tr><th>误差分解（collar 0.25, 含重叠）</th><th>Miss 漏检</th><th>FA 误检</th><th>Conf 混淆</th><th>计分语音</th></tr>
  <tr><td>占参考语音比例</td><td>${D.der_std.miss_pct}%</td><td>${D.der_std.fa_pct}%</td><td>${D.der_std.conf_pct}%</td><td>${D.der_std.ref_s}s</td></tr>
</table>
<div style="color:var(--muted);font-size:12px;margin-top:8px">评分工具：${D.scorer} · 标准 DER 实现（最优说话人映射 + collar + overlap）。collar 0.25 行对应 md-eval <code>-c 0.25</code> 口径，可与 Alimeeting/M2MeT 榜单对比。</div>
<table style="margin-top:12px">
  <tr><th>GT 说话人</th><th>↔ 预测说话人（最优映射）</th><th>重叠时长</th></tr>
  ${D.mapping.map(r=>`<tr><td><span class="sw" style="background:${(D.gt_lanes.find(g=>g.id===r.gt)||{}).color}"></span>${r.gt}</td><td>${r.pred==="—"?"—":`<span class="sw" style="background:${(D.pred_lanes.find(p=>p.id===r.pred)||{}).color}"></span>${r.pred}`}</td><td>${r.overlap_s}s</td></tr>`).join("")}
</table>`;

// legend
document.getElementById("legend").innerHTML =
  "<b style='color:var(--text)'>GT:</b> " +
  D.gt_lanes.map(l=>`<span><span class="sw" style="background:${l.color}"></span>${l.id}</span>`).join("") +
  " &nbsp;&nbsp; <b style='color:var(--text)'>预测:</b> " +
  D.pred_lanes.map(l=>`<span><span class="sw" style="background:${l.color}"></span>${l.id}</span>`).join("");

// ---------- shared draw routine ----------
const LANE_H=20, LANE_GAP=3, GROUP_GAP=16, WAVE_H=70, AXIS_H=20;
const nGT=D.gt_lanes.length, nPR=D.pred_lanes.length;
const gtTop=WAVE_H+10;
const prTop=gtTop + nGT*(LANE_H+LANE_GAP) + GROUP_GAP;
const totalH = prTop + nPR*(LANE_H+LANE_GAP) + AXIS_H;
const gtIdx={}, prIdx={};
D.gt_lanes.forEach((l,i)=>gtIdx[l.id]=i);
D.pred_lanes.forEach((l,i)=>prIdx[l.id]=i);

function drawTimeline(canvas, W, x0, x1, pxStart){
  // x0,x1: visible time range; W css width; pxStart: scroll offset in px (for axis labels)
  const ctx=canvas.getContext("2d");
  const dpr = (W>15000)?1:(window.devicePixelRatio||1);
  canvas.width=W*dpr; canvas.height=totalH*dpr; canvas.style.width=W+"px"; canvas.style.height=totalH+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,totalH);
  const dur=D.duration;
  const X=t=>t/dur*W;

  // waveform
  const peaks=D.peaks, n=peaks.length, mid=WAVE_H/2;
  ctx.fillStyle="#3d4b5a";
  const bw=Math.max(1,W/n);
  for(let i=0;i<n;i++){const h=peaks[i]/100*(WAVE_H/2-2);const x=i/n*W;ctx.fillRect(x,mid-h,bw,h*2);}

  // group backgrounds + lane labels
  ctx.font="11px sans-serif"; ctx.textBaseline="middle";
  function laneBG(top,n){for(let i=0;i<n;i++){const y=top+i*(LANE_H+LANE_GAP);ctx.fillStyle="rgba(255,255,255,0.03)";ctx.fillRect(0,y,W,LANE_H);}}
  laneBG(gtTop,nGT); laneBG(prTop,nPR);

  // group titles (sticky-ish at left edge of view)
  ctx.fillStyle="var(--muted)";

  // segments
  function drawSegs(segs,topMap,idxMap,colorMap){
    segs.forEach(seg=>{
      if(seg.e<x0||seg.s>x1) return;
      const i=idxMap[seg.spk]; if(i==null)return;
      const y=topMap+i*(LANE_H+LANE_GAP);
      const xa=X(seg.s), xb=X(seg.e);
      ctx.fillStyle=colorMap[seg.spk];
      ctx.globalAlpha=.92;
      ctx.fillRect(xa,y+1.5,Math.max(1,xb-xa),LANE_H-3);
      ctx.globalAlpha=1;
    });
  }
  const gtColor={}, prColor={};
  D.gt_lanes.forEach(l=>gtColor[l.id]=l.color);
  D.pred_lanes.forEach(l=>prColor[l.id]=l.color);
  drawSegs(D.gt_segments,gtTop,gtIdx,gtColor);
  drawSegs(D.pred_segments,prTop,prIdx,prColor);

  // axis
  const axisY=prTop+nPR*(LANE_H+LANE_GAP)+2;
  ctx.strokeStyle="#2d3742"; ctx.fillStyle="#8b98a5"; ctx.font="10px sans-serif"; ctx.textAlign="center";
  const pps=W/dur;
  let step= pps>12?10: pps>4?30: pps>1.5?60:120;
  for(let t=0;t<=dur;t+=step){const x=X(t);ctx.beginPath();ctx.moveTo(x,axisY);ctx.lineTo(x,axisY+5);ctx.stroke();ctx.fillText(fmt(t),x,axisY+13);}
  ctx.textAlign="left";

  // group labels
  ctx.fillStyle="#7d8b99"; ctx.font="bold 10px sans-serif";
  // playhead
  drawPlayhead(ctx,X,axisY);
  canvas._X=X; canvas._axisY=axisY;
}
let curTime=0;
function drawPlayhead(ctx,X,axisY){
  const x=X(curTime);
  ctx.strokeStyle="#ff4d4d";ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,axisY);ctx.stroke();ctx.lineWidth=1;
}

// ---------- overview ----------
const ov=document.getElementById("ov");
function drawOv(){const W=ov.clientWidth||ov.parentElement.clientWidth;drawTimeline(ov,W,0,D.duration,0);}

// ---------- detail ----------
const dt=document.getElementById("dt");
const dwrap=document.getElementById("detailwrap");
let pxPerSec=4;
function drawDt(){const W=Math.max(dwrap.clientWidth, Math.round(D.duration*pxPerSec));drawTimeline(dt,W,0,D.duration,0);}

// ---------- audio ----------
const audio=new Audio("data:audio/mp3;base64,"+D.audio);
const playBtn=document.getElementById("play"), timeEl=document.getElementById("time"), rateSel=document.getElementById("rate"), zoomSel=document.getElementById("zoom");
function seekFromCanvas(canvas,e){const r=canvas.getBoundingClientRect();const W=canvas.clientWidth;const t=(e.clientX-r.left)/W*D.duration;audio.currentTime=Math.max(0,Math.min(D.duration,t));}
ov.addEventListener("click",e=>{seekFromCanvas(ov,e);scrollDetailToPlayhead(true);});
dt.addEventListener("click",e=>seekFromCanvas(dt,e));
playBtn.onclick=()=>audio.paused?audio.play():audio.pause();
audio.addEventListener("play",()=>{playBtn.textContent="⏸ 暂停";loop();});
audio.addEventListener("pause",()=>{playBtn.textContent="▶ 播放";});
audio.addEventListener("ended",()=>playBtn.textContent="▶ 播放");
rateSel.onchange=()=>audio.playbackRate=parseFloat(rateSel.value);
zoomSel.onchange=()=>{pxPerSec=parseFloat(zoomSel.value);drawDt();scrollDetailToPlayhead(true);};

function scrollDetailToPlayhead(center){
  const W=dt.clientWidth; const x=curTime/D.duration*W;
  if(center){dwrap.scrollLeft=x-dwrap.clientWidth/2;}
  else{const m=dwrap.clientWidth*0.15;if(x<dwrap.scrollLeft+m||x>dwrap.scrollLeft+dwrap.clientWidth-m)dwrap.scrollLeft=x-dwrap.clientWidth/2;}
}
let raf;
function loop(){
  curTime=audio.currentTime;
  timeEl.textContent=fmt(curTime)+" / "+fmt(D.duration);
  drawOv(); drawDt(); scrollDetailToPlayhead(false);
  if(!audio.paused) raf=requestAnimationFrame(loop);
}
audio.addEventListener("seeked",()=>{curTime=audio.currentTime;timeEl.textContent=fmt(curTime)+" / "+fmt(D.duration);drawOv();drawDt();});

window.addEventListener("resize",()=>{drawOv();drawDt();});
timeEl.textContent="0:00 / "+fmt(D.duration);
drawOv(); drawDt();
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
