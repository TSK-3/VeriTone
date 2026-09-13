# VeriTone — 5-Minute Demo Script (SIH judging)

## Start the demo console (already validated end-to-end)

```powershell
cd C:\Users\theka\Downloads\sih
$env:TIER1_CHECKPOINT = "C:\Users\theka\Downloads\sih\checkpoints\tier1_mlaad.pt"
$env:ALERT_COOLDOWN_S = "6"          # several SMS during the demo
.venv\Scripts\python.exe -m uvicorn voice_detection.api:app --app-dir src --port 8000
```

Open **http://localhost:8000** — the dashboard with the LIVE CALL panel on the left.

## Act 1 — The scam call (90 seconds, the money shot)

1. Click **"▶ Run simulated scam call"**.
2. Narrate while it runs (~25 s): *"This is a live collections call. One leg is a real
   human, the other is an AI-cloned voice made with a Cartesia Sonic-3 TTS model —
   real cloned audio from our held-out MLAAD eval set, streamed through the same
   pipeline a Twilio call uses."*
3. Point at the screen as it happens:
   - **Risk gauge climbs 41 → 70** with a pulsing LIVE dot
   - **Tier 1 (trained CNN)**: genuine leg ~20%, cloned legs 76–99%
   - **Trigger words highlight in the transcript**: *transaction, send, money, OTP, urgent, approve*
   - **SMS cards appear**: first the trigger-word alert, then the **threshold alert
     ("action frozen, secondary verification required")** as the risk crosses the
     ₹-transfer threshold
4. Read one SMS out loud — it tells the customer to **hang up and call back on the
   official number, never share the OTP**.

## Act 2 — Real Twilio call (optional, if a number is configured)

The WebSocket endpoint `/v1/streams/twilio` is production-shaped. Point a Twilio
Media Stream at it (public URL via ngrok):

```powershell
ngrok http 8000
```

In the Twilio Console → Voice → TwiML App (or a BIN on the number) use:

```xml
<Response>
  <Start>
    <Transcription track="inbound" />
    <Connect>
      <Stream url="wss://<your-ngrok>.ngrok.app/v1/streams/twilio">
        <Parameter name="session_id" value="caller-live-1"/>
      </Stream>
    </Connect>
  </Start>
  <Pause length="600"/>
</Response>
```

Media arrives as µ-law 8 kHz → decoded in memory → VAD-gated into 1.2–3.2 s speech
segments → 16 kHz resample → Tier 1 + Tier 2 → running risk. Twilio's real-time
transcription events drive the trigger-word SMS rules.

Enable real SMS (otherwise alerts run in clearly-labelled console mode):

```powershell
$env:TWILIO_ACCOUNT_SID = "ACxxxx"; $env:TWILIO_AUTH_TOKEN = "xxxx"
$env:TWILIO_FROM_NUMBER = "+1xxxx";  $env:ALERT_TO_NUMBER = "+91xxxxxxxxxx"
```

Pre-flight check: click **"✉ Send test SMS"** on the dashboard.

## Act 3 — Engineering depth (one slide, one minute)

- **Tier 1**: 597k-param causal CNN (MLAAD-trained checkpoint, ~3.6 ms/1.5 s window on CPU)
- **Tier 2**: 4-model ensemble design (wav2vec2-XLSR, WavLM, RawNet3, AASIST) with
  quality gating, calibration and disagreement-aware confidence. In this demo the
  members run on calibrated spectral features (flatness/centroid/pause/highband,
  fitted on our own clips — genuine 0.37 vs cloned 0.59) and are labelled
  `demo_signals`; production swaps in the trained ONNX exports with zero API
  changes (`TIER2_MODE=strict` enforces all four artifacts).
- **Privacy**: raw audio never touches disk; audit store keeps derived scores only.
- **Prevention**: per-scenario thresholds (support 70 / transfer 40 / privileged 30),
  recency-weighted evidence aggregation (alert after 3 segments), contextual
  enrichment (caller reputation, transaction amount), Twilio SMS to the customer.

## Failure-proofing

- No Twilio creds? SMS shows `console mode` on the dashboard — pipeline still visible.
- Model artifacts missing? `TIER2_MODE=strict` refuses to score (by design); demo
  mode is the default locally.
- Judge uploads their own WAV: the right-hand panel still works — any 16-bit PCM WAV.
