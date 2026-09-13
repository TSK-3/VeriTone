# Voice Clone Detection

Privacy-first reference service for streaming, segment-level synthetic-speech risk scoring.
Built for **SIH 2026 (SIH26104)** — AI-powered voice cloning / impersonation detection.

Each submitted speech segment is scored by a low-latency **Tier 1** causal CNN and a
parallel **Tier 2** ensemble in the same request. The response carries an explainable
feature summary, an optional cross-session consistency result, and a threshold-based
alert. Raw audio is decoded and discarded in memory; the audit store keeps derived
results only — never waveforms.

> **Status:** Tier 1 is a trained checkpoint (`checkpoints/tier1_mlaad.pt`,
> 597k params, grows out of the MVP heuristic). Tier 2 requires four trained ONNX
> models and will refuse scoring if any required artifact is unavailable.

---

## 1. How it works

```
WAV segment (1–3 s, 16-bit PCM) ──POST /v1/calls/{id}/segments──▶ decode (in-memory)
        ├─▶ Tier 1: Tier1CausalCNN checkpoint ──▶ score + latency
        └─▶ Tier 2: 4 trained ONNX models + calibrated fusion ──▶ score, confidence, contributions
                              │
                    consistency check (optional speaker_similarity)
                              │
                    combined risk = 0.35·T1 + 0.65·T2 (+0.15 if inconsistent)
                              │
            RunningRiskAggregator (per call, recency-weighted, Tier 2 gains
            weight with its confidence) ──▶ alert after ≥3 segments @ ≥0.70
                              │
                    audit record (scores + metadata only)
```

- **Tier 1 model** (`src/voice_detection/tier1_cnn.py`): causal log-mel frontend with
  per-utterance normalization → `Conv(64,7) → Conv(128,5)+pool → dilated
  Conv(256,3)×3` → global pooling → classification head. No recurrence by design —
  long-range call context lives in `RunningRiskAggregator`, not in the model.
  Input is a fixed 1.5 s window (24,000 samples @ 16 kHz); longer segments are
  averaged over 50%-overlap sliding windows. Training adds SpecAugment-lite and
  waveform augmentation (speed, gain, noise, polarity, shift).
- **Tier 2** (`src/voice_detection/tier2_ensemble.py`): four mandatory anti-spoof
  models (`wav2vec2_xlsr`, `wavlm_large`, `rawnet3`, `aasist`) plus calibrated,
  quality-aware fusion. It has no heuristic fallback.
- **Privacy**: `audio.py:decode_wav` never touches disk; `models.py:audit_record`
  contains no audio or embeddings; `?feature_only_logging=true` drops even the
  feature breakdown from the stored record.

### Measured performance (current checkpoint, full set)

| Split | Files (gen/spoof) | Accuracy @0.5 | EER | Mean latency | p95 |
|---|---|---|---|---|---|
| MLAAD train sample (`data/`) | 12,470 (6,070 / 6,400) | **0.72** | **0.283** | 7.1 ms | 9.5 ms |
| Held-out unseen (`data_unseen/`, German + unseen TTS) | 752 (375 / 377) | **0.64** | **0.353** | 9.4 ms | 13.3 ms |

Center-crop evaluation over the complete splits (one 1.5 s window per file,
threshold 0.5), run with `scripts/benchmark_tier1.py`. Score means:
genuine 0.336 vs spoof 0.668 on the train split (+0.333 separation) and
0.452 vs 0.682 on the unseen split (+0.230). Re-run the script after retraining;
use `--mode sliding` for the authoritative serving-path numbers on a sample.
Model: **597,057 params** (under the 2M edge budget), single 1.5 s window scored
in ~7–10 ms on CPU (this machine; re-benchmark p95 on target edge hardware).

---

## 2. Repository structure

```
├── src/voice_detection/      # the service (pip package)
│   ├── api.py                # FastAPI routes + Twilio WebSocket + dashboard mount
│   ├── service.py            # Tier 1 / Tier 2 orchestration
│   ├── tier1_cnn.py          # trainable causal CNN architecture
│   ├── tier1_adapter.py      # TIER1_CHECKPOINT loader (checkpoint > heuristic)
│   ├── train_tier1.py        # training script (CPU or CUDA)
│   ├── aggregation.py        # per-call running-risk aggregator
│   ├── alerts.py             # trigger words + TwilioClient (SMS, calls, status)
│   ├── twilio_stream.py      # Media Streams WS: µ-law → 16 kHz → VAD → scoring
│   ├── twilio_play.py        # cloned-voice scam audio + TwiML playback helpers
│   ├── audit_store.py        # durable Fernet-encrypted audit db (SQLite) + erasure
│   ├── speaker_refs.py       # consented speaker references (spectral embedding)
│   ├── audio.py / models.py  # audio primitives (decode/resample/RMS), PRD schema
├── scripts/                  # benchmark_tier1.py (full-set accuracy/EER/latency)
│                             # + download_mlaad_tiny.py (dataset fetcher)
├── notebooks/colab_train_tier1.ipynb  # GPU training notebook (Colab T4)
├── web/                      # live-call dashboard (index.html, app.js, styles.css)
├── tests/                    # pytest contract tests
├── checkpoints/              # *.pt live here (git-ignored, .gitkeep keeps the dir)
├── data/ data_unseen/        # training / held-out WAVs (git-ignored)
├── audit/                    # encrypted audit db (git-ignored, created on first run)
├── Dockerfile / docker-compose.yml / .dockerignore / .env.example
├── voice-clone-detection-prd.md    # full PRD
└── pyproject.toml
```

Checkpoints, datasets, and `*.pt` files are intentionally **not** committed
(see `.gitignore`). The container and the server both run without a checkpoint —
Tier 1 simply falls back to heuristics until you mount one.

---

## 3. Quickstart (local)

Requires Python ≥ 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # API + tests
pip install -e ".[ml]"           # + torch/numpy for the CNN checkpoint

# With a trained checkpoint (recommended):
$env:TIER1_CHECKPOINT = "checkpoints\tier1_mlaad.pt"
uvicorn voice_detection.api:app --reload
```

Open `http://127.0.0.1:8000` for the live-call dashboard, or
`http://127.0.0.1:8000/docs` for OpenAPI.

Submit a 16-bit mono WAV segment (1–3 s recommended). Audio goes in the raw
request body so it is never spooled to disk:

```powershell
curl -X POST "http://127.0.0.1:8000/v1/calls/demo/segments?start_s=0" `
  -H "Content-Type: audio/wav" --data-binary "@segment.wav"

# With a speaker reference + privacy mode:
curl -X POST "http://127.0.0.1:8000/v1/calls/demo/segments?start_s=3&speaker_similarity=0.41&feature_only_logging=true" `
  -H "Content-Type: audio/wav" --data-binary "@segment2.wav"

# Derived results only, never audio:
curl "http://127.0.0.1:8000/v1/calls/demo/audit"
```

---

## 4. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TIER1_CHECKPOINT` | *(unset → heuristic)* | Path to `{"model_state_dict": …, "config": …}` checkpoint. When set, the CNN replaces the Tier 1 heuristic. |
| `PORT` | `8000` | Used by the Docker `CMD`; uvicorn flag locally. |
| `AUDIT_STORE` | `on` | `off` disables the durable audit db (in-memory only). |
| `AUDIT_STORE_PATH` | `audit/veritone-audit.db` | SQLite location for the Fernet-encrypted audit store (volume-mounted in Docker). |
| `AUDIT_STORE_KEY` | *(auto → `<db>.key`)* | Fernet key; if unset a key is generated beside the db and reused. |

Alerting behaviour (`RunningRiskAggregator`: window 5, threshold 0.7,
min 3 evidence segments) is code-configured in `api.py` / `aggregation.py`.

---

## 5. API reference

- `GET /health` → `{"status": "ok"}` (also the Docker healthcheck).
- `POST /v1/calls/{call_id}/segments?start_s=0&speaker_similarity=&feature_only_logging=`
  — `Content-Type: audio/wav` (or `audio/x-wav`, `application/octet-stream`),
  raw WAV bytes in the body. Returns the audit record below.
  Errors: `415` non-WAV content type, `422` invalid WAV / bad `speaker_similarity`.
- `GET /v1/calls/{call_id}/audit` → list of derived records for the call.
- `GET /v1/audit?session_id=&limit=` → decrypted records from the durable
  encrypted store; `DELETE /v1/audit/{session_id}` → right-to-erasure.
- `POST /v1/speakers/{speaker_id}/reference?consent=true` — enrol a consented
  speaker reference (raw WAV body; `403` without consent). `GET /v1/speakers/{id}`
  → status; `DELETE /v1/speakers/{id}` → erasure. Sessions created with a
  `speaker_id` check every segment against the reference live
  (`consistency_check.similarity_score`).
- `GET /` → dashboard. `GET /docs` → OpenAPI UI.

Example segment response (abridged):

```json
{
  "segment_timestamp_range": [0.0, 1.5],
  "tier1": {"score": 0.76, "label": "synthetic", "latency_ms": 4},
  "tier2": {"score": 0.85, "label": "synthetic", "confidence": 0.91,
            "encoder_contributions": {"wav2vec2_xlsr": 0.81, "wavlm_large": 0.88, "rawnet3": 0.86}},
  "combined_risk_score": 0.82,
  "running_risk_score": 0.79,
  "evidence_segments": 3,
  "consistency_check": {"ran": true, "similarity_score": 0.41, "flag": "inconsistent"},
  "feature_breakdown": {"prosody_irregularity": "high", "spectral_artifacts": "high",
                        "breathing_pattern": "absent", "background_noise_consistency": "inconsistent"},
  "alert": true,
  "recommended_action": "Request secondary verification before proceeding."
}
```

Alerts require ≥ 3 segments by default — single-segment scores are evidence,
never verdicts.

---

## 6. Training

All commands assume `pip install -e ".[ml]"` and run from the repo root.
`train_tier1.py` auto-selects CUDA when available; `--num-workers 2` on Colab.

```powershell
# 1) Smoke test, no dataset (synthetic genuine/synthetic clips):
$env:PYTHONPATH = "src"
python -m voice_detection.train_tier1 --synthetic 800 --epochs 8 --out checkpoints/tier1_cnn.pt

# 2) Real data (directories of 16-bit PCM WAVs):
python -m voice_detection.train_tier1 --genuine-dir data/genuine --spoof-dir data/spoof `
  --epochs 20 --out checkpoints/tier1_cnn.pt

# 3) MLAAD (no registration). Tiny slice first, then full:
pip install -e ".[data]"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
python scripts/download_mlaad_tiny.py            # ~1.3k + ~1.3k + 400 unseen
python scripts/download_mlaad_tiny.py --full     # ~6k + ~6.4k + ~750 unseen
python -m voice_detection.train_tier1 --genuine-dir data/genuine --spoof-dir data/spoof `
  --epochs 20 --lr 0.001 --batch-size 128 --num-workers 2 --out checkpoints/tier1_mlaad.pt
```

Training details: waveform tensors `[batch, 24000]`, `BCEWithLogitsLoss` +
AdamW + cosine schedule, best-checkpoint selection on val accuracy, final report
includes center-crop **and** sliding-window (inference-style) accuracy/EER plus
a CPU latency probe. `Tier1CausalCNN.parameter_count` asserts the 2M edge budget.

### GPU on Google Colab (recommended for the full 12k set)

1. `Runtime → Change runtime type → T4 GPU`.
2. Bundle your **current working tree** (checkpoints in git are stale the moment
   the arch changes — never train from an old checkout):
   ```powershell
   python -c "import zipfile; from pathlib import Path
   files = list(Path('src/voice_detection').glob('*.py')) + [Path('scripts/download_mlaad_tiny.py'), Path('pyproject.toml')]
   [zipfile.ZipFile('sih-code.zip','w',zipfile.ZIP_DEFLATED).write(f, f.as_posix()) for f in files]"
   ```
3. In Colab: `File → Upload notebook` → `notebooks/colab_train_tier1.ipynb`,
   run cells top-to-bottom (GPU check → upload `sih-code.zip` → deps →
   `--full` download → smoke test → 20-epoch train → download checkpoint).
4. Copy the downloaded file back to `checkpoints/tier1_mlaad.pt` and set
   `TIER1_CHECKPOINT` before starting the server.

---

## 7. Docker (recommended for demos and deployment)

The image is CPU-first (Torch CPU wheels, ~300 MB saved vs CUDA wheels) and runs
as a non-root user with a `/health` healthcheck. GPU training stays in Colab —
the container serves the trained checkpoint on CPU.

```powershell
# Build + run with Compose (mounts ./checkpoints read-only):
Copy-Item .env.example .env
docker compose up --build
# → http://localhost:8000 (dashboard), /docs, /health

# Or plain Docker:
docker build -t voice-clone-detection .
docker run -p 8000:8000 -v "${PWD}/checkpoints:/app/checkpoints:ro" `
  -e TIER1_CHECKPOINT=/app/checkpoints/tier1_mlaad.pt voice-clone-detection
```

Notes:

- Without `tier1_mlaad.pt` in `./checkpoints`, the container still starts —
  Tier 1 serves heuristic scores until you mount the file. No rebuild needed.
- `docker-compose.yml` reads `PORT` from `.env` (`${PORT:-8000}`).
- For a GPU *serving* image, swap the Torch install lines for CUDA wheels and
  run with `--gpus`; training images are intentionally out of scope here.

---

## 8. Testing

```powershell
pip install -e ".[dev]"
python -m pytest tests -q
```

37 tests pin the PRD contract and the live-call path: audit records carry the
three encoder slots and no audio bytes, invalid `speaker_similarity` is rejected,
alerts require aggregated evidence across segments, the Twilio µ-law → VAD →
scoring → trigger-SMS pipeline runs end-to-end, the shared audio primitives
(WAV decode, resampling, frame RMS) are covered, and the durable encrypted audit
store + consented speaker references round-trip with erasure enforced.

For a live end-to-end check (server running on :8901):

```powershell
$env:TIER1_CHECKPOINT = "checkpoints\tier1_mlaad.pt"
Start-Process .venv\Scripts\python.exe -ArgumentList '-m','uvicorn','voice_detection.api:app','--port','8901'
python scripts\smoke_test_audit.py   # enrol → score → consistency → erasure
```

---

## 9. Demo scenario (SIH judging)

Simulated live support/collections call: one leg is a real speaker, the other a
cloned voice pushing an action (refund / payment approval). Stream 1–3 s WAV
segments per leg through `POST /v1/calls/{leg}/segments`, watch the dashboard's
running risk, and expect the pre-transaction alert on the cloned leg once
evidence reaches 3 segments — with feature breakdown and consistency mismatch
as supporting evidence. The dashboard's demo button renders a sample alert
without audio.

---

## 10. Roadmap

1. **Done** — Tier 2 ensemble: `Tier2ProductionEnsemble` (ONNX + manifest, strict)
   replaces the old heuristic; local runs use the labelled `DemoTier2Ensemble`.
2. **Done** — VAD-backed Twilio Media Streams adapter emitting 1–3 s speech
   segments (`twilio_stream.py`).
3. **Done** — durable Fernet-encrypted audit store (`audit_store.py`, SQLite +
   right-to-erasure) and consented speaker references (`speaker_refs.py`) with
   live consistency checking per segment.
4. **Done** — full-set benchmark on the complete MLAAD splits (12,470 + 752
   files): accuracy 0.72 / EER 0.283 on the train sample, 0.64 / EER 0.353 on
   the held-out unseen split; §1 table refreshed. Next: the same benchmark on
   ASVspoof + VoIP/codec augmentations with the next trained checkpoint.
5. Multi-worker deployment — the registry/session store are in-process by
   design; move to Redis or a DB when scaling beyond one process.

### Tier 2 production ensemble path

Tier 2 now has four isolated scorer slots (`wav2vec2_xlsr`, `wavlm_large`,
`rawnet3`, `aasist`) and an independent prosody signal. Its fusion is quality-aware,
calibrated with a manifest temperature/bias, and explicitly lowers confidence when
models disagree or audio is weak/clipped. This prevents a brittle single score from
being presented as a confident verdict.

For production, export each fine-tuned anti-spoof model as ONNX with its own waveform
preprocessing embedded, then point `TIER2_MANIFEST` to a manifest with model paths.
All four trained models are mandatory. The service refuses Tier 2 scoring if the
manifest or any model is missing; there is no heuristic/development fallback.

```powershell
pip install -e ".[tier2]"
$env:TIER2_MANIFEST = "C:\models\tier2-manifest.json"
```

---

## 11. Session API and prevention workflow

Swagger documents the integration surface at `/docs`:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/sessions` | Start a monitored call with scenario, channel, language hint and privacy mode. |
| `POST /v1/sessions/{id}/audio` | Submit an in-memory WAV segment. |
| `POST /v1/sessions/{id}/context` | Add caller reputation, transaction value and fraud indicators during a call. |
| `GET /v1/sessions/{id}/score` | Read the continuous 0–100 contextual risk score and prevention action. |
| `GET /v1/sessions/{id}/verdict` | Finish a session and retrieve its verdict. |
| `WS /v1/streams/twilio` | Twilio Media Streams connection surface. |

The rules engine returns `pass`, `warn`, `step_up_verification`, `escalate`, or
`block`, with a human-readable recommendation. Current configured thresholds
are support calls: 70, high-value transfer approvals: 40, and privileged access
requests: 30.

The Twilio WebSocket is an adapter boundary: it accepts Twilio events and bounds
raw-frame retention, but still needs a vendor-specific µ-law-to-16kHz-PCM worker
and Twilio credentials/configuration to produce live model scores. No account
configuration is stored in this repository.
