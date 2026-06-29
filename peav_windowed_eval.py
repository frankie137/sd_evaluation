#!/usr/bin/env python3
"""Windowed evaluation + visualization for the PEAV Sortformer model (45 s windows).

The PEAV model is trained on 45 s windows and has NO cross-window speaker
association (speaker ids are only locally consistent inside a window). So we:

  1. cut the audio into 45 s non-overlapping windows (matching validation_ds),
  2. run the model independently on each window -> per-window speaker-activity
     probabilities,
  3. binarize the probs to per-window predicted segments using the SAME
     post-processing as tasks/dia/metrics.py (onset/offset = 0.5),
  4. score with diar_eval.evaluate() in per-window mode -- each window gets its
     OWN optimal speaker mapping and the raw error/reference durations are pooled
     into a single DER (this is exactly the right protocol for a model without
     cross-window speaker consistency),
  5. render a self-contained HTML showing, per window, predicted vs ground-truth
     speaker lanes (color-matched by that window's optimal mapping) plus the
     per-window DER / miss / FA / confusion.

Example:
    python peav_windowed_eval.py \
        --checkpoint /workspace/peaf_conformer_40M_head/checkpoints/step=25000-val_der=0.112199.pt \
        --config-file /workspace/peaf_conformer_40M_head/config.yaml \
        --wav  /workspace/sd_full_benchmark/VoxConverse/wavs/aepyx.wav \
        --rttm /workspace/sd_full_benchmark/VoxConverse/rttms/aepyx.rttm
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
import time
from pathlib import Path

# --- make both repos importable ------------------------------------------------
ALM_ROOT = Path("/workspace/ALM")
SD_EVAL = Path("/workspace/sd_evaluation")
for p in (ALM_ROOT, SD_EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf
from pyannote.core import Annotation, Segment

import diar_eval
from tasks.dia.factory import build_model
from tasks.dia.metrics import _DER_VAD_PARAMS, get_der_frame_length_sec, ts_vad_post_processing
from training.config import load_config, resolve_config_file

# Dark-theme palette (kept consistent with sd_evaluation/visualize_compare.py).
PALETTE = ["#58a6ff", "#f78166", "#56d364", "#d2a8ff", "#e3b341", "#ff7b72",
           "#79c0ff", "#ffa657", "#7ee787", "#ff9bce"]
UNMATCHED = "#6e7681"

# old checkpoint prefix -> new model prefix (from scripts/dia_eval_checkpoint.py)
KEY_MAP = [
    ("dac_vae.", "preprocessor.dac_vae."),
    ("data_proj.", "preprocessor.data_proj."),
    ("sortformer_modules.", "head.sortformer_modules."),
    ("transformer_encoder.", "head.transformer_encoder."),
]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def remap_legacy_state_dict(old_sd):
    new_sd = {}
    for key, value in old_sd.items():
        new_key = key
        for old_prefix, new_prefix in KEY_MAP:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        new_sd[new_key] = value
    return new_sd


def load_peav_model(config_file, checkpoint_path, device):
    cfg = load_config(resolve_config_file(config_file), [])
    model = build_model(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        old_sd = ckpt["state_dict"]
    elif "model" in ckpt:
        old_sd = ckpt["model"]
    else:
        old_sd = ckpt
    new_sd = remap_legacy_state_dict(old_sd)

    model_keys = set(model.state_dict().keys())
    load_sd = {k: v for k, v in new_sd.items() if k in model_keys}
    model.load_state_dict(load_sd, strict=False)
    print(f"[model] loaded {len(load_sd)}/{len(model_keys)} params", flush=True)

    model.eval().to(device)
    return model, cfg


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def load_audio(path, target_sr):
    """Load a wav as mono float32 at target_sr."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(np.ascontiguousarray(data))
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav, sr


def probs_to_segments(prob_matrix, num_valid_frames, frame_length_sec):
    """[T, n_spk] sigmoid probs -> list of (start, end, spk_idx) in window-local sec.

    Uses the exact same binarization/post-processing as
    tasks/dia/metrics.frames_to_annotation (onset/offset = 0.5).
    """
    pm = prob_matrix[:num_valid_frames]
    n_spk = pm.shape[-1]
    out = []
    for spk in range(n_spk):
        col = pm[:, spk].detach().cpu().float()
        seg = ts_vad_post_processing(
            col,
            cfg_vad_params=OmegaConf.create(OmegaConf.to_container(_DER_VAD_PARAMS, resolve=True)),
            unit_10ms_frame_count=1,
            bypass_postprocessing=False,
            frame_length_sec=frame_length_sec,
        )
        for s, e in seg.tolist():
            if e > s:
                out.append((float(s), float(e), spk))
    return out


# --------------------------------------------------------------------------- #
# Windowed inference
# --------------------------------------------------------------------------- #
def run_windowed_inference(model, cfg, wav, sr, device, window_sec, stride_sec):
    frame_len_sec = get_der_frame_length_sec(cfg)
    total_sec = wav.shape[0] / sr

    windows = []
    t = 0.0
    while t < total_sec - 1e-3:
        windows.append((round(t, 3), round(min(t + window_sec, total_sec), 3)))
        t += stride_sec

    per_window_hyp = []   # Annotation per window, in ABSOLUTE time
    per_window_segs = []  # list[(start, end, spk_idx)] absolute, for the viz
    autocast_enabled = device == "cuda"

    t0_infer = time.time()
    for i, (ws, we) in enumerate(windows):
        s0, s1 = int(ws * sr), int(we * sr)
        chunk = wav[s0:s1].unsqueeze(0).to(device)
        length = torch.tensor([chunk.shape[1]], device=device, dtype=torch.long)

        with torch.no_grad(), torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            out = model(audio_signal=chunk, audio_signal_length=length)

        preds = out["preds"][0].float()                 # [T, n_spk]
        valid = int(out["lengths"][0].item())
        valid = min(valid, preds.shape[0])
        win_len = we - ws

        segs_rel = probs_to_segments(preds, valid, frame_len_sec)
        ann = Annotation(uri=f"w{i}")
        segs_abs = []
        for s, e, spk in segs_rel:
            a = ws + s
            b = ws + min(e, win_len)            # clip to the real window extent
            if b > a:
                ann[Segment(a, b)] = f"spk{spk}"
                segs_abs.append((round(a, 3), round(b, 3), spk))
        per_window_hyp.append(ann)
        per_window_segs.append(segs_abs)

    infer_s = time.time() - t0_infer
    print(f"[infer] {len(windows)} windows in {infer_s:.1f}s "
          f"(audio {total_sec:.1f}s, RTF={infer_s/max(total_sec,1e-9):.4f})", flush=True)
    return windows, per_window_hyp, per_window_segs, total_sec, infer_s


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _fmt(x):
    return f"{x:.2f}"


def crop_annotation(ann, t0, t1):
    """Return [(start, end, label), ...] of `ann` intersected with [t0, t1]."""
    cropped = ann.crop(Segment(t0, t1), mode="intersection")
    return [(seg.start, seg.end, lab) for seg, _, lab in cropped.itertracks(yield_label=True)]


def render_lane_svg(t0, t1, gt_segs, pred_segs, gt_speakers, gt_color, mapping,
                    pps=24.0, lane_h=20, lane_gap=5):
    """One window: GT lanes (stable, global) above, predicted lanes below.

    Predicted lanes are colored by this window's optimal hyp->ref mapping so a
    color clash between a predicted bar and the GT lane above it = confusion;
    a predicted bar over silence = false alarm; an empty predicted slot under a
    GT bar = missed detection.
    """
    LG = 78            # left gutter for lane labels
    PAD_R = 12
    AX_H = 20          # top axis height
    win_len = max(t1 - t0, 1e-3)
    plot_w = win_len * pps
    width = LG + plot_w + PAD_R

    # predicted speakers present in this window, ordered
    pred_ids = sorted({spk for _, _, spk in pred_segs})

    n_lanes = len(gt_speakers) + len(pred_ids)
    group_gap = 10
    height = AX_H + len(gt_speakers) * (lane_h + lane_gap) + group_gap \
        + len(pred_ids) * (lane_h + lane_gap) + 8

    def x_of(t):
        return LG + (t - t0) * pps

    parts = [f'<svg class="lane-svg" width="{width:.0f}" height="{height:.0f}" '
             f'viewBox="0 0 {width:.0f} {height:.0f}" '
             f'data-t0="{t0}" data-t1="{t1}" data-lg="{LG}" data-pps="{pps}" '
             f'style="cursor:crosshair" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']

    # time grid + labels (every 5 s, absolute)
    tick = 5.0
    first = (int(t0 // tick)) * tick
    tt = first
    while tt <= t1 + 1e-6:
        if tt >= t0 - 1e-6:
            x = x_of(tt)
            parts.append(f'<line x1="{x:.1f}" y1="{AX_H}" x2="{x:.1f}" y2="{height-4}" '
                         f'stroke="#21262d" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="13" fill="#6e7681" font-size="10" '
                         f'text-anchor="middle">{tt:.0f}s</text>')
        tt += tick

    y = AX_H + 4

    def draw_lane(label, label_color, segs, color_fn):
        nonlocal y
        # lane background
        parts.append(f'<rect x="{LG}" y="{y}" width="{plot_w:.1f}" height="{lane_h}" '
                     f'rx="3" fill="#0d1117" stroke="#1b2129" stroke-width="1"/>')
        parts.append(f'<text x="{LG-8}" y="{y+lane_h*0.7:.0f}" fill="{label_color}" '
                     f'font-size="11" text-anchor="end">{html.escape(label)}</text>')
        for s, e, key in segs:
            x = x_of(max(s, t0))
            w = max(x_of(min(e, t1)) - x, 1.0)
            parts.append(f'<rect x="{x:.1f}" y="{y+2}" width="{w:.1f}" height="{lane_h-4}" '
                         f'rx="2" fill="{color_fn(key)}"/>')
        y += lane_h + lane_gap

    # GT group
    parts.append(f'<text x="4" y="{y+lane_h*0.7:.0f}" fill="#8b949e" font-size="10">GT</text>')
    for s in gt_speakers:
        lane_segs = [(a, b, lab) for a, b, lab in gt_segs if lab == s]
        draw_lane(s, gt_color[s], lane_segs, lambda lab: gt_color.get(lab, UNMATCHED))

    y += group_gap

    # Pred group
    if pred_ids:
        parts.append(f'<text x="4" y="{y+lane_h*0.7:.0f}" fill="#8b949e" font-size="10">Pred</text>')
    for spk in pred_ids:
        mapped = mapping.get(f"spk{spk}")
        col = gt_color.get(mapped, UNMATCHED)
        tag = f"P{spk}" + (f" →{mapped}" if mapped else " →(none)")
        lane_segs = [(a, b, spk) for a, b, sp in pred_segs if sp == spk]
        draw_lane(tag, col, lane_segs, lambda _k, c=col: c)

    parts.append(f'<line class="hov" x1="-9" y1="{AX_H}" x2="-9" y2="{height-4}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="3,2" style="display:none"/>')
    parts.append(f'<line class="ph" x1="-9" y1="{AX_H}" x2="-9" y2="{height-4}" '
                 f'stroke="#ff7b72" stroke-width="1.5" style="display:none"/>')
    parts.append(f'<text class="hovt" x="-9" y="11" fill="#e6edf3" font-size="10" '
                 f'text-anchor="middle" style="display:none"></text>')
    parts.append("</svg>")
    return "".join(parts)


def render_overview_svg(total_sec, ref, per_window_segs, windows, window_results,
                        gt_speakers, gt_color, width=1180, lane_h=16):
    """Whole-file overview: GT lanes over all speakers + a single predicted lane
    strip per window (colored by each window's mapping), with window boundaries."""
    LG, PAD_R, AX_H = 78, 12, 18
    pps = (width - LG - PAD_R) / max(total_sec, 1e-3)

    def x_of(t):
        return LG + t * pps

    n_lanes = len(gt_speakers) + 1
    gap = 5
    height = AX_H + n_lanes * (lane_h + gap) + 26

    parts = [f'<svg class="lane-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'data-t0="0" data-t1="{total_sec}" data-lg="{LG}" data-pps="{pps}" '
             f'style="cursor:crosshair" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']

    # window boundaries + 30s ticks
    for (ws, we) in windows:
        x = x_of(ws)
        parts.append(f'<line x1="{x:.1f}" y1="{AX_H}" x2="{x:.1f}" y2="{height-4}" '
                     f'stroke="#30363d" stroke-width="1" stroke-dasharray="3,3"/>')
    tt = 0.0
    while tt <= total_sec:
        x = x_of(tt)
        parts.append(f'<text x="{x:.1f}" y="12" fill="#6e7681" font-size="9" '
                     f'text-anchor="middle">{tt:.0f}</text>')
        tt += 30.0

    y = AX_H + 2
    # GT lanes
    parts.append(f'<text x="4" y="{y+lane_h*0.7:.0f}" fill="#8b949e" font-size="9">GT</text>')
    for s in gt_speakers:
        parts.append(f'<rect x="{LG}" y="{y}" width="{x_of(total_sec)-LG:.1f}" height="{lane_h}" '
                     f'rx="2" fill="#0d1117" stroke="#1b2129"/>')
        parts.append(f'<text x="{LG-8}" y="{y+lane_h*0.7:.0f}" fill="{gt_color[s]}" '
                     f'font-size="10" text-anchor="end">{html.escape(s)}</text>')
        for seg, _, lab in ref.itertracks(yield_label=True):
            if lab != s:
                continue
            x = x_of(seg.start)
            w = max(x_of(seg.end) - x, 0.7)
            parts.append(f'<rect x="{x:.1f}" y="{y+2}" width="{w:.1f}" height="{lane_h-4}" '
                         f'fill="{gt_color[s]}"/>')
        y += lane_h + gap

    # single predicted strip (per-window mapped colors)
    parts.append(f'<text x="4" y="{y+lane_h*0.7:.0f}" fill="#8b949e" font-size="9">Pred</text>')
    parts.append(f'<rect x="{LG}" y="{y}" width="{x_of(total_sec)-LG:.1f}" height="{lane_h}" '
                 f'rx="2" fill="#0d1117" stroke="#1b2129"/>')
    for segs_abs, wr in zip(per_window_segs, window_results):
        mapping = wr.mapping if wr else {}
        for a, b, spk in segs_abs:
            mapped = mapping.get(f"spk{spk}")
            col = gt_color.get(mapped, UNMATCHED)
            x = x_of(a)
            w = max(x_of(b) - x, 0.7)
            parts.append(f'<rect x="{x:.1f}" y="{y+2}" width="{w:.1f}" height="{lane_h-4}" fill="{col}"/>')
    y += lane_h + gap

    parts.append(f'<line class="hov" x1="-9" y1="{AX_H}" x2="-9" y2="{height-4}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="3,2" style="display:none"/>')
    parts.append(f'<line class="ph" x1="-9" y1="{AX_H}" x2="-9" y2="{height-4}" '
                 f'stroke="#ff7b72" stroke-width="1.5" style="display:none"/>')
    parts.append(f'<text class="hovt" x="-9" y="11" fill="#e6edf3" font-size="10" '
                 f'text-anchor="middle" style="display:none"></text>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#010409;--panel:#0d1117;--panel2:#161b22;--text:#e6edf3;--muted:#8b949e;--grid:#30363d;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;padding:22px}
h1{font-size:19px;margin:0 0 2px}
.sub{color:var(--muted);font-size:13px;margin-bottom:16px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.stat{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:9px 14px;min-width:96px}
.stat .k{color:var(--muted);font-size:11px}
.stat .v{font-size:19px;font-weight:600;margin-top:2px}
.stat.big .v{color:#ffa657}
.panel{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:14px 16px;margin:12px 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--muted);margin:6px 0 2px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
.win-card{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:10px 14px;margin:12px 0;cursor:pointer}
.win-card:hover{border-color:var(--accent)}
.win-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;margin-bottom:6px}
.win-head .title{font-size:14px;font-weight:600}
.win-head .der{font-size:13px;color:#ffa657;font-variant-numeric:tabular-nums}
.win-head .small{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.svgwrap{overflow-x:auto}
table{border-collapse:collapse;font-size:12.5px;margin-top:6px}
th,td{padding:3px 10px;text-align:right;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:500}
td:first-child,th:first-child{text-align:left}
audio{width:100%;margin-top:6px}
.note{color:var(--muted);font-size:12px;margin-top:8px;line-height:1.55}
code{background:var(--panel2);padding:1px 5px;border-radius:4px}
"""

SEEK_JS = """
const audio=document.getElementById('player');
const svgs=[...document.querySelectorAll('svg.lane-svg')];
function xToTime(svg,clientX){
  const r=svg.getBoundingClientRect();
  const scale=svg.viewBox.baseVal.width/r.width;
  const lg=parseFloat(svg.dataset.lg), pps=parseFloat(svg.dataset.pps);
  const t0=parseFloat(svg.dataset.t0), t1=parseFloat(svg.dataset.t1);
  const x=(clientX-r.left)*scale;
  return Math.min(t1, Math.max(t0, t0+(x-lg)/pps));
}
function xOf(svg,t){
  const lg=parseFloat(svg.dataset.lg), pps=parseFloat(svg.dataset.pps), t0=parseFloat(svg.dataset.t0);
  return lg+(t-t0)*pps;
}
svgs.forEach(svg=>{
  const hov=svg.querySelector('.hov'), hovt=svg.querySelector('.hovt');
  svg.addEventListener('mousemove',e=>{
    const t=xToTime(svg,e.clientX), x=xOf(svg,t);
    hov.setAttribute('x1',x); hov.setAttribute('x2',x); hov.style.display='';
    hovt.setAttribute('x',x); hovt.textContent=t.toFixed(2)+'s'; hovt.style.display='';
  });
  svg.addEventListener('mouseleave',()=>{hov.style.display='none'; hovt.style.display='none';});
  // click anywhere on the timeline -> seek to that exact time and play
  svg.addEventListener('click',e=>{
    if(!audio) return;
    audio.currentTime=xToTime(svg,e.clientX);
    audio.play();
  });
});
// red playhead synced to playback position, shown in whichever svg spans currentTime
function updatePlayhead(){
  if(!audio) return;
  const ct=audio.currentTime;
  svgs.forEach(svg=>{
    const t0=parseFloat(svg.dataset.t0), t1=parseFloat(svg.dataset.t1);
    const ph=svg.querySelector('.ph');
    if(ct>=t0 && ct<=t1){ const x=xOf(svg,ct); ph.setAttribute('x1',x); ph.setAttribute('x2',x); ph.style.display=''; }
    else ph.style.display='none';
  });
}
if(audio){ audio.addEventListener('timeupdate',updatePlayhead); audio.addEventListener('seeked',updatePlayhead); }
"""


def build_html(*, title, sub, res, windows, per_window_segs, ref, gt_speakers,
               gt_color, total_sec, audio_b64, audio_mime, extra_stats):
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             f"<style>{CSS}</style></head><body>"]
    parts.append(f"<h1>{html.escape(title)}</h1><div class='sub'>{html.escape(sub)}</div>")

    # stats
    s = [("Pooled DER", f"{res.der:.2f}%", "big"),
         ("Miss", f"{res.miss_pct:.2f}%", ""),
         ("FA", f"{res.fa_pct:.2f}%", ""),
         ("Conf", f"{res.conf_pct:.2f}%", ""),
         ("#windows", str(res.n_windows), ""),
         ("collar", f"{res.collar}", ""),
         ("skip_overlap", str(res.skip_overlap), "")]
    s += extra_stats
    parts.append("<div class='stats'>")
    for k, v, cls in s:
        parts.append(f"<div class='stat {cls}'><div class='k'>{html.escape(k)}</div>"
                     f"<div class='v'>{html.escape(str(v))}</div></div>")
    parts.append("</div>")

    # legend (global GT speaker colors)
    parts.append("<div class='legend'><b style='color:var(--text)'>GT speakers:</b>")
    for sp in gt_speakers:
        parts.append(f"<span><span class='sw' style='background:{gt_color[sp]}'></span>{html.escape(sp)}</span>")
    parts.append("</div>")

    # audio
    if audio_b64:
        parts.append(f"<audio id='player' controls preload='none' "
                     f"src='data:{audio_mime};base64,{audio_b64}'></audio>")

    # overview
    parts.append("<div class='panel'><div class='sub' style='margin:0 0 8px'>"
                 "Whole-file overview &mdash; predicted strip is colored by each window's "
                 "optimal mapping (dashed lines = window boundaries)</div><div class='svgwrap'>")
    parts.append(render_overview_svg(total_sec, ref, per_window_segs, windows,
                                     res.windows, gt_speakers, gt_color))
    parts.append("</div></div>")

    # per-window cards
    for i, ((ws, we), wr, segs_abs) in enumerate(zip(windows, res.windows, per_window_segs)):
        gt_segs = crop_annotation(ref, ws, we)
        der_txt = "n/a" if wr.der is None else f"{wr.der:.2f}%"
        svg = render_lane_svg(ws, we, gt_segs, segs_abs, gt_speakers, gt_color,
                              wr.mapping if wr else {})
        parts.append(f"<div class='win-card' data-start='{ws:.3f}'>")
        parts.append(
            f"<div class='win-head'><span class='title'>Window {i}: "
            f"{ws:.1f}–{we:.1f}s</span>"
            f"<span class='der'>DER {der_txt}</span>"
            f"<span class='small'>miss {_fmt(wr.miss)}s &middot; FA {_fmt(wr.false_alarm)}s "
            f"&middot; conf {_fmt(wr.confusion)}s &middot; ref {_fmt(wr.reference)}s</span></div>")
        # mapping line
        if wr and wr.mapping:
            maps = " &nbsp; ".join(
                f"<span class='sw' style='background:{gt_color.get(v, UNMATCHED)}'></span>"
                f"P{k.replace('spk','')}→{html.escape(v)}"
                for k, v in sorted(wr.mapping.items()))
            parts.append(f"<div class='legend'>map: {maps}</div>")
        parts.append(f"<div class='svgwrap'>{svg}</div></div>")

    parts.append(
        "<div class='note'>Each window is scored independently by "
        "<code>diar_eval.evaluate()</code> (per-window optimal speaker mapping), then "
        "miss/FA/confusion seconds are pooled into a single DER "
        "(<code>pooled_DER = &sum; error / &sum; reference</code>). Predicted lanes are "
        "colored by the window's hyp&rarr;ref mapping, so a predicted bar whose color "
        "differs from the GT speaker active beneath it is a <b>confusion</b>; a predicted "
        "bar over silence is a <b>false alarm</b>; a GT bar with no matching predicted bar "
        "is a <b>miss</b>. Speaker ids are <b>not</b> linked across windows by design "
        "(the model has no cross-window association).<br>"
        "<b>Tip:</b> click anywhere on a timeline to play from that exact moment; "
        "hover shows the time; the red line tracks playback.</div>")

    parts.append(f"<script>{SEEK_JS}</script></body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default="/workspace/peaf_conformer_40M_head/checkpoints/step=25000-val_der=0.112199.pt")
    ap.add_argument("--config-file", default="/workspace/peaf_conformer_40M_head/config.yaml")
    ap.add_argument("--wav", default="/workspace/sd_full_benchmark/VoxConverse/wavs/aepyx.wav")
    ap.add_argument("--rttm", default="/workspace/sd_full_benchmark/VoxConverse/rttms/aepyx.rttm")
    ap.add_argument("--out", default="/workspace/sd_evaluation/out/peav_windowed/aepyx.html")
    ap.add_argument("--collar", type=float, default=0.25,
                    help="DER collar (s); VoxConverse benchmark uses 0.25")
    ap.add_argument("--skip-overlap", action="store_true",
                    help="exclude overlapped speech from scoring")
    ap.add_argument("--window-sec", type=float, default=None,
                    help="window length; default = validation_ds.session_len_sec")
    ap.add_argument("--stride-sec", type=float, default=None,
                    help="window hop; default = non-overlapping (= window-sec)")
    ap.add_argument("--no-audio", action="store_true", help="do not embed the wav")
    ap.add_argument("--audio-bitrate", default="64k", help="mp3 bitrate for embedded audio")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)

    model, cfg = load_peav_model(args.config_file, args.checkpoint, device)

    val_ds = cfg.model.validation_ds
    window_sec = args.window_sec if args.window_sec is not None else float(val_ds.session_len_sec)
    if args.stride_sec is not None:
        stride_sec = args.stride_sec
    else:
        stride_sec = float(val_ds.get("eval_window_stride_sec", window_sec))
    sr = int(cfg.model.sample_rate)
    print(f"[cfg] window={window_sec}s stride={stride_sec}s sample_rate={sr} "
          f"frame_len={get_der_frame_length_sec(cfg)}s", flush=True)

    wav, orig_sr = load_audio(args.wav, sr)
    print(f"[audio] {Path(args.wav).name} orig_sr={orig_sr} -> {sr}, "
          f"dur={wav.shape[0]/sr:.1f}s", flush=True)

    windows, per_window_hyp, per_window_segs, total_sec, infer_s = run_windowed_inference(
        model, cfg, wav, sr, device, window_sec, stride_sec)

    # ---- DER via the standardized pipeline (diar_eval, per-window mode) ----
    ref = diar_eval.load_rttm(args.rttm, uri=Path(args.rttm).stem)
    res = diar_eval.evaluate(ref, per_window_hyp, windows=windows,
                             collar=args.collar, skip_overlap=args.skip_overlap)

    # ---- console summary ----
    print("\n" + "=" * 72)
    print(f"POOLED DER = {res.der:.2f}%   (miss {res.miss_pct:.2f} / "
          f"FA {res.fa_pct:.2f} / conf {res.conf_pct:.2f})   "
          f"collar={res.collar} skip_overlap={res.skip_overlap}")
    print("-" * 72)
    print(f"{'win':>3} {'span(s)':>14} {'DER%':>7} {'miss':>7} {'FA':>7} {'conf':>7} {'ref_s':>7}")
    for i, wr in enumerate(res.windows):
        der = "  n/a" if wr.der is None else f"{wr.der:6.2f}"
        print(f"{i:>3} {f'{wr.start:.1f}-{wr.end:.1f}':>14} {der} "
              f"{wr.miss:7.2f} {wr.false_alarm:7.2f} {wr.confusion:7.2f} {wr.reference:7.2f}")
    print("=" * 72 + "\n")

    # ---- colors ----
    gt_speakers = sorted(ref.labels())
    gt_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(gt_speakers)}

    # ---- audio embed (compressed to mp3 to keep the page light) ----
    audio_b64, audio_mime = "", "audio/mpeg"
    if not args.no_audio:
        import os
        import subprocess
        import tempfile
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", args.wav,
                 "-ac", "1", "-b:a", args.audio_bitrate, tmp_mp3],
                check=True)
            raw = Path(tmp_mp3).read_bytes()
            audio_b64 = base64.b64encode(raw).decode("ascii")
            print(f"[viz] embedding mp3 @{args.audio_bitrate} ({len(raw)/1e6:.1f} MB)", flush=True)
        except Exception as e:
            print(f"[viz] ffmpeg failed ({e}); embedding raw wav", flush=True)
            audio_mime = "audio/wav"
            audio_b64 = base64.b64encode(Path(args.wav).read_bytes()).decode("ascii")
        finally:
            try:
                os.remove(tmp_mp3)
            except OSError:
                pass

    extra = [("audio", f"{total_sec:.0f}s", ""),
             ("GT speakers", str(len(gt_speakers)), ""),
             ("infer RTF", f"{infer_s/max(total_sec,1e-9):.4f}", "")]
    sub = (f"{Path(args.wav).name} &middot; PEAV Sortformer (45 s windows, no cross-window "
           f"association) &middot; scored with diar_eval per-window pooled DER")
    page = build_html(title=f"PEAV windowed diarization — {Path(args.wav).stem}",
                      sub=sub, res=res, windows=windows, per_window_segs=per_window_segs,
                      ref=ref, gt_speakers=gt_speakers, gt_color=gt_color,
                      total_sec=total_sec, audio_b64=audio_b64, audio_mime=audio_mime,
                      extra_stats=extra)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[out] {out}  ({out.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
