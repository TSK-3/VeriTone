# Voice Clone Detection MVP

Privacy-first reference service for streaming, segment-level synthetic-speech risk scoring.

It implements the PRD's orchestration contract now: each submitted speech segment is
processed by a low-latency Tier 1 scorer and a parallel Tier 2 ensemble; the result
contains an explainable feature summary, optional cross-session consistency result,
and a threshold-based alert. Raw audio is decoded and discarded in the same request;
the in-memory audit store retains only the derived result.

> The scoring adapters are deliberately deterministic heuristics, not trained anti-spoofing
> models. Replace them with benchmarked model checkpoints before production use.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn voice_detection.api:app --reload
```

Open `http://127.0.0.1:8000` for the live-call dashboard. It submits WAV bytes directly
to the API (rather than via multipart upload) so audio is not spooled to disk.

Submit a 16-bit mono WAV segment (1–3 seconds recommended):

```powershell
curl -X POST "http://127.0.0.1:8000/v1/calls/demo/segments?start_s=0" -H "Content-Type: audio/wav" --data-binary "@segment.wav"
```

Use `GET /v1/calls/{call_id}/audit` to retrieve derived results only. The OpenAPI UI
is available at `/docs`.

## Tier 1 and call confidence

Tier 1 is intentionally a fast per-segment scorer. Its production adapter should be
a shallow causal/dilated 1D CNN over MFCC frames or waveform samples: causal/dilated
convolution blocks → global pooling → a small classification head. Long-range call
context belongs in the separate per-call aggregator, not an LSTM/GRU. Alerts require
three recent segments by default; their running score is recency weighted and gives
Tier 2 more influence as its confidence rises.

### Training the Tier 1 CNN

Install the ML extra, train with waveform tensors shaped `[batch, samples]` and
`BCEWithLogitsLoss`, then save either a plain state dictionary or
`{"model_state_dict": state_dict}`. The CNN computes causal log-mel features itself;
input must be 16 kHz mono PCM. Set `TIER1_CHECKPOINT` to the saved checkpoint path
before starting the server and the checkpoint-backed CNN automatically replaces the
development heuristic.

```powershell
pip install -e ".[ml]"
$env:TIER1_CHECKPOINT = "C:\models\tier1_cnn.pt"
uvicorn voice_detection.api:app --app-dir src --reload
```

`Tier1CausalCNN.parameter_count` verifies the model stays below the 2M-parameter
edge budget. Benchmark p95 latency on the target edge hardware with a 1.5-second
window before deployment.

## Next implementation milestones

1. Replace `HeuristicTier1Scorer` and `HeuristicTier2Scorer` with ONNX/PyTorch model adapters.
2. Add a VAD-backed websocket/media adapter that emits 1–3s speech segments.
3. Connect a durable, encrypted feature-only audit store and consented speaker references.
4. Benchmark EER and latency with ASVspoof and VoIP/codec augmentations.
