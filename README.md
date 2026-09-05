# VeriTone

VeriTone is a privacy-first reference service for detecting synthetic or cloned speech
in streaming call segments. It accepts short WAV segments, combines a fast Tier 1
scorer with a parallel Tier 2 ensemble, and returns an explainable risk result with
optional speaker-consistency checks and threshold-based alerts.

The current scoring adapters are deterministic heuristics intended for development and
integration testing. They are not a production anti-spoofing model. Replace them with
benchmarked model adapters before making security or identity decisions.

## Features

- Low-latency, segment-level risk scoring.
- Tier 1 and Tier 2 results with explainable feature and model contributions.
- Recency-weighted call-level aggregation; alerts require multiple segments by default.
- Optional cross-session speaker similarity checks.
- Raw audio is decoded in memory and is not written to the audit store.
- FastAPI service, OpenAPI documentation, and a lightweight browser dashboard.

## Requirements

- Python 3.11 or newer
- `pip`
- A 16-bit mono WAV file for manual API testing

The project supports Windows, macOS, and Linux. The commands below use the standard
Python virtual environment module and can be run from a shell in the repository root.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/TSK-3/VeriTone.git
cd VeriTone
```

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the application and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, either run `Set-ExecutionPolicy -Scope Process
Bypass` for the current shell or invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Run the service

With the virtual environment active, start the development server:

```bash
python -m uvicorn voice_detection.api:app --reload
```

Open the dashboard at <http://127.0.0.1:8000>. Interactive OpenAPI documentation is
available at <http://127.0.0.1:8000/docs>.

To check that the service is running:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Score an audio segment

Submit a 1–3 second, 16-bit mono WAV segment directly as the request body. No
multipart upload is required:

```bash
curl -X POST "http://127.0.0.1:8000/v1/calls/demo/segments?start_s=0" \
  -H "Content-Type: audio/wav" \
  --data-binary "@segment.wav"
```

PowerShell users can use:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/calls/demo/segments?start_s=0" `
  -Method Post `
  -ContentType "audio/wav" `
  -InFile ".\segment.wav"
```

Optional query parameters:

- `start_s`: segment start time in seconds; defaults to `0`.
- `speaker_similarity`: a value from `0` to `1` for a known speaker reference.
- `feature_only_logging=true`: omit the feature breakdown from the returned result.

Retrieve the derived audit records for a call:

```bash
curl "http://127.0.0.1:8000/v1/calls/demo/audit"
```

The in-memory audit store is cleared whenever the process restarts. It stores derived
scores and metadata only, never waveform bytes.

## Run tests

Install the development extra as described above, then run:

```bash
python -m pytest
```

## Optional Tier 1 CNN checkpoint

The ML extra enables the optional PyTorch Tier 1 adapter:

```bash
python -m pip install -e ".[ml]"
```

Set `TIER1_CHECKPOINT` to a checkpoint containing either a plain PyTorch state
dictionary or a `{"model_state_dict": state_dict}` object, then start the server:

**Windows PowerShell**

```powershell
$env:TIER1_CHECKPOINT = "C:\models\tier1_cnn.pt"
python -m uvicorn voice_detection.api:app --reload
```

**macOS/Linux**

```bash
export TIER1_CHECKPOINT=/models/tier1_cnn.pt
python -m uvicorn voice_detection.api:app --reload
```

The checkpoint model expects 16 kHz mono PCM input. Validate accuracy, p95 latency,
and the model's parameter budget on the target hardware before deployment.

## Project structure

```text
src/voice_detection/
  api.py             FastAPI routes and dashboard hosting
  audio.py           WAV decoding and audio validation
  service.py         Tier 1/Tier 2 scoring orchestration
  tier1_adapter.py   Development and checkpoint-backed Tier 1 adapters
  tier1_cnn.py       Optional causal CNN implementation
  aggregation.py     Per-call running risk and alert aggregation
  models.py          Typed result and audit data models
web/                 Browser dashboard
tests/               Service contract tests
```

## Production considerations

This repository is an MVP and should not be used as the sole control for
authentication, payments, access control, or other high-impact decisions. Before
production use:

1. Replace heuristic scorers with evaluated anti-spoofing models.
2. Add authentication, authorization, rate limiting, and request-size limits.
3. Move audit records to a durable, encrypted, access-controlled feature-only store.
4. Add monitoring, model/version tracking, and retention controls.
5. Benchmark with representative codecs, devices, languages, and datasets such as
   ASVspoof.

## License

No license has been selected for this repository yet.
