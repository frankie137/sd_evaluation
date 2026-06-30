#!/usr/bin/env python3
"""Compare official AMI Mix-Headset vs the user's custom IHM mix (aggregate DER),
Sortformer + model-card post-processing, scored against the forced_align reference."""
import sys, os, glob, tempfile, pathlib
sys.path.insert(0, "/workspace/sd_evaluation")
import benchmark as B, diar_eval

DEST = "/workspace/sd_full_benchmark/_ami_mix_official"
OLD = "/workspace/sd_full_benchmark/AMI_forced_align"
CUSTOM_PRED = "/workspace/sortformer_diar/out/preds/AMI_forced_align/IHM"  # cached (postproc)
keys = sorted(os.path.splitext(os.path.basename(w))[0] for w in glob.glob(DEST + "/*.wav"))
print(f"comparing {len(keys)} files: {keys}", flush=True)

model = B.load_model()

def pooled(pairs):
    m = f = c = r = 0.0
    for ref, hyp in pairs:
        res = diar_eval.evaluate_collection([(ref, hyp)], collar=0.0, skip_overlap=False)
        w = res.windows[0]
        m += w.miss; f += w.false_alarm; c += w.confusion; r += w.reference
    return dict(der=round((m+f+c)/r*100, 2), miss=round(m/r*100, 2),
                fa=round(f/r*100, 2), conf=round(c/r*100, 2))

# official: fresh inference
off_pairs, per_file = [], []
for k in keys:
    ref = f"{OLD}/IHM/rttms/{k}.rttm"
    segs = model.diarize(audio=[f"{DEST}/{k}.wav"], batch_size=1, verbose=False,
                         postprocessing_yaml=B.POSTPROC_YAML)[0]
    tf = tempfile.NamedTemporaryFile(suffix=".rttm", delete=False).name
    B.write_rttm(segs, k, pathlib.Path(tf))
    off_pairs.append((ref, tf))
    r1 = diar_eval.evaluate_collection([(ref, tf)], collar=0.0, skip_overlap=False)
    per_file.append((k, r1.der))

# custom: from cache (same keys)
cu_pairs = [(f"{OLD}/IHM/rttms/{k}.rttm", f"{CUSTOM_PRED}/{k}.rttm")
            for k in keys if os.path.exists(f"{CUSTOM_PRED}/{k}.rttm")]

off = pooled(off_pairs)
cu = pooled(cu_pairs)
print("\n=== per-file official DER ===", flush=True)
for k, d in per_file:
    print(f"  {k}: {d}", flush=True)
print(f"\n=== AGGREGATE (15 files, same forced_align reference, collar=0, +overlap) ===", flush=True)
print(f"  official Mix-Headset : {off}", flush=True)
print(f"  custom self-mix      : {cu}", flush=True)
print(f"  (model card AMI IHM = 15.90, over 16 files)", flush=True)
for _, tf in off_pairs:
    try: os.remove(tf)
    except OSError: pass
