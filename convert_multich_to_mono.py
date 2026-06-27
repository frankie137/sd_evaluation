#!/usr/bin/env python3
"""Convert every multi-channel WAV under the benchmark to mono (channel 0), in place.

Method: take channel 0 only (SDM condition), 16 kHz, PCM_16 -- matching the
AISHELL-4 conversion convention already used in this benchmark. Mono files are
left untouched. Each conversion writes to a temp file in the same directory and
is then atomically renamed over the original.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf

ROOT = Path("/workspace/speaker_diarization_benchmark")


def main():
    wavs = sorted(ROOT.rglob("*.wav"))
    multich = []
    for w in wavs:
        try:
            info = sf.info(str(w))
        except Exception as e:
            print(f"[WARN] cannot read {w}: {e}", flush=True)
            continue
        if info.channels > 1:
            multich.append((w, info.channels, info.samplerate))

    print(f"Found {len(multich)} multi-channel file(s) to convert.\n", flush=True)
    ok = 0
    for i, (w, ch, sr) in enumerate(multich, 1):
        print(f"[{i}/{len(multich)}] {w.name}  ({ch}ch/{sr}Hz) -> mono ch0 ...",
              flush=True)
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=str(w.parent))
        import os
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(w),
                 "-filter:a", "pan=mono|c0=c0",
                 "-ar", "16000", "-c:a", "pcm_s16le",
                 tmp],
                check=True,
            )
            out = sf.info(tmp)
            if out.channels != 1:
                raise RuntimeError(f"result has {out.channels} channels, expected 1")
            shutil.move(tmp, str(w))  # atomic within same filesystem
            print(f"        OK -> {out.channels}ch/{out.samplerate}Hz "
                  f"{out.duration:.0f}s", flush=True)
            ok += 1
        except Exception as e:
            print(f"        FAILED: {e}", flush=True)
            Path(tmp).unlink(missing_ok=True)

    print(f"\nDone. Converted {ok}/{len(multich)} file(s).", flush=True)
    sys.exit(0 if ok == len(multich) else 1)


if __name__ == "__main__":
    main()
