#!/usr/bin/env python3
"""Single-file streaming Sortformer (4spk v2.1) diarization, following the model card.

https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1

Loads the model, sets the recommended high-latency streaming config, runs
`diarize()` on one mono-16k WAV, and writes the prediction as an RTTM plus the
raw per-frame speaker-activity probabilities (.npy) for visualization.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_segment_line(line):
    """A predicted segment is the string 'start end speaker_idx' (space-sep)."""
    parts = line.split()
    start = float(parts[0])
    end = float(parts[1])
    spk = parts[2]  # e.g. 'speaker_0'
    return start, end, spk


def write_rttm(segments, uri, out_path):
    with open(out_path, "w") as f:
        for start, end, spk in segments:
            dur = end - start
            f.write(
                f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audio",
        default="/workspace/speaker_diarization_benchmark/Alimeeting/Far/wavs/R8005_M8009.wav",
    )
    ap.add_argument("--out-dir", default="/workspace/sortformer_diar/out")
    ap.add_argument(
        "--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1"
    )
    ap.add_argument(
        "--latency",
        choices=["high", "low"],
        default="high",
        help="streaming latency preset (high=30.4s, best accuracy)",
    )
    args = ap.parse_args()

    audio = Path(args.audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    uri = audio.stem

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] device={device} model={args.model}", flush=True)

    from nemo.collections.asr.models import SortformerEncLabelModel

    t0 = time.time()
    diar_model = SortformerEncLabelModel.from_pretrained(args.model)
    diar_model.eval()
    diar_model = diar_model.to(device)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    # --- streaming config (values in 80ms frames), per model card ---
    if args.latency == "high":
        # Very high latency: 30.4s, best accuracy
        diar_model.sortformer_modules.chunk_len = 340
        diar_model.sortformer_modules.chunk_right_context = 40
        diar_model.sortformer_modules.fifo_len = 40
        diar_model.sortformer_modules.spkcache_update_period = 300
        diar_model.sortformer_modules.spkcache_len = 188
    else:
        # Low latency: 1.04s
        diar_model.sortformer_modules.chunk_len = 6
        diar_model.sortformer_modules.chunk_right_context = 7
        diar_model.sortformer_modules.fifo_len = 188
        diar_model.sortformer_modules.spkcache_update_period = 144
        diar_model.sortformer_modules.spkcache_len = 188
    diar_model.sortformer_modules._check_streaming_parameters()
    print(
        f"[cfg] latency={args.latency} "
        f"chunk_len={diar_model.sortformer_modules.chunk_len} "
        f"right_ctx={diar_model.sortformer_modules.chunk_right_context} "
        f"fifo_len={diar_model.sortformer_modules.fifo_len} "
        f"spkcache_len={diar_model.sortformer_modules.spkcache_len}",
        flush=True,
    )

    import soundfile as sf

    info = sf.info(str(audio))
    print(
        f"[audio] {audio.name} sr={info.samplerate} ch={info.channels} "
        f"dur={info.duration:.1f}s",
        flush=True,
    )

    # --- inference ---
    t0 = time.time()
    predicted_segments, predicted_probs = diar_model.diarize(
        audio=str(audio),
        batch_size=1,
        include_tensor_outputs=True,
        verbose=True,
    )
    infer_s = time.time() - t0
    rtf = infer_s / info.duration
    print(
        f"[infer] done in {infer_s:.1f}s  (RTF={rtf:.4f}, "
        f"{1/rtf:.1f}x realtime)",
        flush=True,
    )

    seg_lines = predicted_segments[0]
    probs = predicted_probs[0]
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().float().numpy()
    probs = np.asarray(probs)

    segments = [parse_segment_line(s) for s in seg_lines]
    spk_set = sorted({s[2] for s in segments})
    total_speech = sum(e - s for s, e, _ in segments)

    rttm_path = out_dir / f"{uri}.pred.rttm"
    write_rttm(segments, uri, rttm_path)
    np.save(out_dir / f"{uri}.probs.npy", probs)

    meta = {
        "uri": uri,
        "audio": str(audio),
        "model": args.model,
        "latency_preset": args.latency,
        "audio_duration_s": round(info.duration, 3),
        "infer_seconds": round(infer_s, 2),
        "rtf": round(rtf, 5),
        "num_segments": len(segments),
        "pred_speakers": spk_set,
        "num_pred_speakers": len(spk_set),
        "total_pred_speech_s": round(total_speech, 2),
        "probs_shape": list(probs.shape),
        "frame_sec": 0.08,
    }
    (out_dir / f"{uri}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print(f"\nFirst 8 predicted segments:", flush=True)
    for s in seg_lines[:8]:
        print("   ", s, flush=True)
    print(f"\n[out] {rttm_path}", flush=True)
    print(f"[out] {out_dir / (uri + '.probs.npy')}", flush=True)
    print(f"[out] {out_dir / (uri + '.meta.json')}", flush=True)


if __name__ == "__main__":
    main()
