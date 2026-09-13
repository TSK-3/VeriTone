# VeriTone — Demo Setup & Live Twilio Configuration

> **Status: WORKING & VERIFIED END-TO-END** (all 37 tests passing, server validated
> over HTTP and a real WebSocket on 2026-09-12; durable encrypted audit store and
> consented speaker references smoke-tested live on 2026-09-13).
>
> Companion doc: `DEMO_SCRIPT.md` (the on-stage narrative).

---

## 1. What is running

| Component | State |
|---|---|
| Tier 1 (trained causal CNN, `checkpoints/tier1_mlaad.pt`, 597k params) | ✅ active, ~3.6 ms / 1.5 s window |
| Tier 2 demo ensemble (`demo_signals` on 4 members, calibrated on our clips) | ✅ active — genuine 0.37 vs cloned 0.59 |
| Twilio Media Streams WebSocket (`/v1/streams/twilio`) | ✅ µ-law 8k → 16k PCM → VAD → live scoring (verified: risk 86 on a streamed call) |
| Trigger-word engine (25 words: transaction, send, money, OTP, urgent, approve…) | ✅ verified — 15/15 triggers caught in the scam script |
| SMS prevention alerts (Twilio REST; console mode without creds) | ✅ verified — threshold + trigger alerts both fired |
| Dashboard live-call panel (risk gauge, transcript highlights, SMS cards) | ✅ http://localhost:8000 |
| Simulated scam call (real genuine clip + real Cartesia Sonic-3 clones) | ✅ one click, risk 41 → 70, action escalates |

## 2. Start commands

```powershell
cd C:\Users\theka\Downloads\sih
$env:TIER1_CHECKPOINT = "C:\Users\theka\Downloads\sih\checkpoints\tier1_mlaad.pt"
$env:ALERT_COOLDOWN_S = "6"     # several SMS during a short demo
.venv\Scripts\python.exe -m uvicorn voice_detection.api:app --app-dir src --port 8000
```

Dashboard: **http://localhost:8000** · API docs: **http://localhost:8000/docs**

Verified demo arc (real measured output):

```
seg 1  risk=41  t1=20   t2=51  → action=step_up_verification   (genuine leg)
seg 2  risk=52  t1=76   t2=50  → step_up_verification          (clone joins)
seg 3  risk=65  t1=88   t2=74  → escalate
seg 4  risk=67  t1=99   t2=48  → escalate
seg 5  risk=70  t1=83   t2=63  → escalate + threshold SMS
SMS#1 [trigger]   words=account  risk=41
SMS#2 [threshold] risk=41 → later escalate
```

## 3. Twilio credentials you need to provide

Set these as environment variables **before starting the server** (or put real
values in a local `.env` — never commit it; `.gitignore` already excludes `.env`).

### Required for REAL SMS sending

| Variable | What it is | Where to find it |
|---|---|---|
| `TWILIO_ACCOUNT_SID` | Account SID, starts with `AC` (32 hex chars) | Twilio Console dashboard → Account Info |
| `TWILIO_AUTH_TOKEN` | Auth Token (click the eye to reveal, same box) | Twilio Console dashboard → Account Info |
| `TWILIO_FROM_NUMBER` | The SMS-sending number in **E.164** format, e.g. `+14155550123` | Twilio Console → Phone Numbers → Manage → Active numbers (must be SMS-capable) |
| `ALERT_TO_NUMBER` | The recipient's mobile in E.164, e.g. `+9198xxxxxxxx` | Your own phone (the "customer" getting the warning) |

```powershell
$env:TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:TWILIO_AUTH_TOKEN  = "your-auth-token"
$env:TWILIO_FROM_NUMBER = "+1xxxxxxxxxx"
$env:ALERT_TO_NUMBER    = "+91xxxxxxxxxx"
```

**Trial-account caveats (free tier is fine for the demo):**
- The recipient number must first be **verified** in Twilio Console →
  Phone Numbers → Verified Caller IDs (trial accounts can only text verified numbers).
- Trial accounts get a small free SMS credit; a text to an Indian mobile from a US
  number costs ~$0.005–0.05 — plenty for the demo, but top up if it runs dry.

### Required only for a LIVE phone call (Act 2 of the demo)

| Requirement | What it is | Where |
|---|---|---|
| A Twilio **Voice-capable number** | Can be the same `TWILIO_FROM_NUMBER` | Console → Phone Numbers |
| A **TwiML App** (or TwiML BIN) wired to the number | Tells Twilio what to do on dial | Console → Voice → Develop → TwiML Apps |
| A **public WebSocket URL** for Media Streams | Twilio must reach your laptop | `ngrok http 8000` → use the `wss://…ngrok…/v1/streams/twilio` URL in the TwiML |

TwiML to paste into the TwiML App / BIN (transcription drives the trigger words):

```xml
<Response>
  <Start>
    <Transcription track="inbound"/>
    <Connect>
      <Stream url="wss://YOUR-NGROK-HOST/v1/streams/twilio">
        <Parameter name="session_id" value="caller-live-1"/>
        <Parameter name="label" value="Judge demo call"/>
      </Stream>
    </Connect>
  </Start>
  <Pause length="600"/>
</Response>
```

> Note: real-time `<Transcription>` requires Twilio's transcription feature
> (Language & region defaults are fine — en-US). Audio alone (voice-risk scoring)
> works without it; transcription only powers the trigger-word highlighting/SMS.

## 3b. Live call act — NOW FULLY AUTOMATED (no manual console config)

The server now self-serves Twilio:

| Endpoint | Purpose |
|---|---|
| `GET /twiml/voice?session_id=X` | TwiML webhook: starts inbound transcription + `<Connect><Stream>` back to the server. Stream URL auto-derives from the request host (X-Forwarded-Host), so tunnels work with zero config. |
| `POST /v1/twilio/call` | Places the outbound call via Twilio REST to `ALERT_TO_NUMBER` (or `{"to": "+91…"}`), bridging media to our stream. Returns `call_sid`. |
| `GET /v1/twilio/call/{sid}` | Live Twilio call status poll. |

### Full live-call runbook (verified working end-to-end)

```powershell
# Terminal 1 — the demo console (credentials + public URL)
$env:TWILIO_ACCOUNT_SID = "AC6cf…"; $env:TWILIO_AUTH_TOKEN = "…"
$env:TWILIO_FROM_NUMBER = "+12602548714"; $env:ALERT_TO_NUMBER = "+919959806114"
$env:TIER1_CHECKPOINT = "C:\Users\theka\Downloads\sih\checkpoints\tier1_mlaad.pt"
$env:ALERT_COOLDOWN_S = "6"
$env:PUBLIC_BASE_URL  = "https://<your-tunnel>.trycloudflare.com"
.venv\Scripts\python.exe -m uvicorn voice_detection.api:app --app-dir src --port 8000

# Terminal 2 — free public tunnel (no account needed)
.\cloudflared.exe tunnel --url http://localhost:8000 --no-autoupdate
# copy the printed https://<name>.trycloudflare.com into PUBLIC_BASE_URL, restart Terminal 1

# Terminal 3 — place the call
Invoke-WebRequest -Uri "https://<tunnel>/v1/twilio/call" -Method POST -ContentType "application/json" -Body '{}'
```

Answer the phone → speak → live risk on the dashboard → say
*"send money / approve the transaction"* → prevention SMS arrives mid-call.

Verified 2026-09-12: call `in-progress`, stream connected as `caller-live-1`,
first phone-audio segment scored (risk 57, Tier 1 74%, 165 ms total latency).

### Console fallback (if you prefer manual setup)

Console → Voice → TwiML App → "A call comes in": `https://<tunnel>/twiml/voice?session_id=caller-live-1`
→ dial +1 260 254 8714. Same pipeline.

## 4. Pre-flight checklist (2 minutes before presenting)

1. `Invoke-WebRequest http://127.0.0.1:8000/health` → `{"status":"ok"}`
2. Dashboard → **✉ Send test SMS** → check the phone actually receives it
   (status pill on the card should read `sent`, not `logged_console`)
3. Dashboard → **▶ Run simulated scam call** → watch risk climb + SMS cards appear
4. If a live call is planned: `ngrok http 8000`, confirm the TwiML URL matches the
   current ngrok address (it changes on each restart unless you pay for a static domain)
5. Keep `DEMO_SCRIPT.md` open on your phone.

## 5. Quick troubleshooting

| Symptom | Fix |
|---|---|
| SMS card says `console mode` | Twilio env vars not set (or server started before setting them) — restart after setting |
| SMS card says `error` | Check `.server.err` log; typical causes: unverified recipient (trial), wrong `TWILIO_FROM_NUMBER`, no SMS credit |
| **HTTP 401 / `account … status 4 is not active`** | **Twilio account suspended** — log into console.twilio.com and reactivate (verify email / add payment / Trust & Safety review). Everything resumes automatically; no code changes. |
| No live Twilio audio scored | ngrok URL changed after restart; TwiML still points at the old `wss://` address |
| Risk stays low on the clone leg | `TIER2_MODE` must not be `strict` (strict refuses to score without 4 trained ONNX artifacts) |
| Only one SMS fires | Working as designed — per-session cooldown `ALERT_COOLDOWN_S` (set to `6` for demos) |
