# Tier 2 training runbook

Tier 2 has no runtime fallback. Production scoring requires all four trained,
calibrated ONNX artifacts: XLS-R, WavLM, RawNet3, and AASIST.

## Where to train

Use Google Colab Pro (A100 preferred, L4/T4 acceptable for the base models) or a
RunPod/Lambda GPU instance. A local CPU is not appropriate for the SSL models.
Store data and checkpoints in Google Drive or a private bucket; never commit voice
data or weights to Git.

## What to train

1. Train `wav2vec2_xlsr` on ASVspoof 2019 LA train/dev plus your consented Indian
   language genuine/spoof samples. Start from `facebook/wav2vec2-xls-r-300m`.
2. Train `wavlm_large` with the same split from `microsoft/wavlm-large`.
3. Train RawNet3 from its maintained research implementation against the same
   protocol. Keep raw waveform at 16 kHz.
4. Train AASIST from its maintained research implementation, with the same labels
   and codec/noise augmentation.
5. Hold out whole spoof-generator families (and ASVspoof 2021/2024 where licensed)
   for evaluation. Do not report random-split accuracy as robustness.

Use a balanced genuine/spoof sampler and augment only training data with telephone
band limiting, µ-law/Opus-like compression, resampling, packet loss, room noise and
gain changes. Measure EER, min-tDCF and p95 segment latency per model, then measure
the fused ensemble on unseen generator families.

## Commands for the two SSL members

```bash
pip install -e ".[ml,tier2]"

python -m voice_detection.train_tier2_ssl \
  --backbone facebook/wav2vec2-xls-r-300m \
  --genuine-dir data/genuine/train --spoof-dir data/spoof/train \
  --epochs 8 --batch-size 8 --out checkpoints/wav2vec2_xlsr.pt

python -m voice_detection.train_tier2_ssl \
  --backbone microsoft/wavlm-large \
  --genuine-dir data/genuine/train --spoof-dir data/spoof/train \
  --epochs 6 --batch-size 4 --out checkpoints/wavlm_large.pt

python -m voice_detection.export_tier2_ssl --checkpoint checkpoints/wav2vec2_xlsr.pt --out checkpoints/wav2vec2_xlsr.onnx
python -m voice_detection.export_tier2_ssl --checkpoint checkpoints/wavlm_large.pt --out checkpoints/wavlm_large.onnx
```

For RawNet3 and AASIST, use the maintained upstream model implementations rather
than substituting a look-alike network. Export each fine-tuned model so it accepts
`float32 [1, samples]` waveform input and emits a single spoof logit. Copy resulting
ONNX files to `checkpoints/rawnet3.onnx` and `checkpoints/aasist.onnx`.

## Calibration and serving

Fit fusion weights/temperature only on a validation set untouched by individual
model training. Populate `config/tier2-manifest.example.json` with the resulting
paths and coefficients, copy it to a private location, then:

```bash
export TIER2_MANIFEST=/secure/models/tier2-manifest.json
python -m uvicorn voice_detection.api:app --app-dir src
```

Startup/inference refuses to produce a Tier 2 score if any required artifact is
missing. This is intentional: it prevents demo heuristics from masquerading as a
trained anti-spoof system.
