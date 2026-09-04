# PRD: Voice Cloning / Synthetic Speech Detection System

**Project:** SIH26104 — AI-powered Voice Cloning/Impersonation Detection
**Context:** SIH 2026 PS + Razorpay AI Buildathon (Open Innovation category)
**Doc owner:** T Siddhartha Karthik
**Status:** Draft for engineering handoff

---

## 1. Problem Statement

Voice cloning and synthetic speech generation have become accessible enough to be used in social-engineering attacks — impersonating a person's voice to authorize actions, bypass voice-based verification, or manipulate support/collections calls. This system continuously analyzes a **live voice stream** (telephony/VoIP/collaboration platform) and classifies ongoing speech as **genuine human speech** or **synthetic/cloned speech** in near real time, delivering an actionable risk score and alert *before* a sensitive action (e.g. fund transfer approval) is taken — with an interpretable confidence breakdown and zero raw-audio persistence.

## 2. Goals

- **Primary:** Maximize detection accuracy against both known and unseen (novel) voice-cloning/TTS attack methods, without sacrificing near-real-time responsiveness during a live call.
- **Secondary:** Provide an interpretable output (not a pure black box) so a human reviewer can understand *why* a segment was flagged.
- **Tertiary:** Preserve privacy — raw audio is never stored; only derived scores/features persist in the audit trail, with edge/on-device and feature-only-logging options where required.

## 3. Non-Goals

- Speaker identification/verification (confirming *who* is speaking) — out of scope. This system only answers real-vs-synthetic. (Cross-session consistency checks in Section 4.7 use speaker *similarity*, not identification, and don't require a full speaker-ID system.)
- Sub-segment (sub-1s) granularity — detection operates on short speech segments (1–3s, VAD-gated), not individual frames or samples.

## 4. System Architecture

### 4.0 Architecture: Two Parallel Tiers — Continuous Streaming + Full Ensemble

The SIH26104 problem statement requires **continuous, near-real-time analysis of live/streaming audio**, with alerts delivered *before* a sensitive action (e.g. fund transfer approval) is taken. It also requires maximum detection accuracy against known and unseen attacks. Both requirements are met by running **two tiers in parallel on every call segment**, rather than one gating the other:

- **Tier 1 — Real-Time Streaming Detection (continuous, low-latency):** a lightweight model scores short speech segments (e.g. 1–3s, VAD-gated) as they complete, producing a running risk score throughout the call with minimal lag. This is the tier the alerting layer watches continuously, satisfying the PS's "continuous risk scoring engine" and "before sensitive action is taken" requirements.
- **Tier 2 — Full Ensemble Deep Analysis (runs on every segment, async, slightly higher latency):** the complete architecture — wav2vec2-XLSR + WavLM-large + RawNet3 + AASIST-style graph attention back-ends + learned score fusion (Sections 4.2–4.4) — runs on the same segments in parallel, not conditionally. Its higher-confidence, interpretable score arrives a beat behind Tier 1 and either confirms/upgrades the running risk score or triggers escalation if it disagrees with Tier 1.

Both tiers write to the same running risk profile for the call; the alerting layer can act on Tier 1 immediately and revise/escalate when Tier 2's result lands. This keeps the full-accuracy ensemble as the system's primary analytical engine — not a fallback — while still satisfying "near real time" through Tier 1's continuous low-latency scoring.

### 4.1 High-Level Pipeline

```
Live audio stream (telephony / VoIP / collaboration platform)
    │
    ▼
VAD / segmentation (chunks into short speech segments, e.g. 1–3s)
    │
    ├─────────────────────────────┬─────────────────────────────────┐
    ▼                             ▼                                 │
┌───────────────────────────┐ ┌─────────────────────────────────┐   │
│ TIER 1 — Real-Time         │ │ TIER 2 — Full Ensemble           │   │
│ Streaming Detection         │ │ Deep Analysis                    │   │
│ (continuous, low-latency)   │ │ (runs on every segment, async)   │   │
│                             │ │                                   │   │
│ Lightweight model scores    │ │ wav2vec2-XLSR │ WavLM-large │    │   │
│ each segment as it          │ │ RawNet3 (each parallel)         │   │
│ completes                   │ │      │              │            │   │
│                             │ │      ▼              ▼            │   │
│ → running risk score,       │ │ AASIST-style graph attention     │   │
│   updated continuously      │ │ back-end (per encoder)           │   │
│   through the call          │ │      │                            │   │
│                             │ │      ▼                            │   │
│                             │ │ Learned score fusion layer       │   │
│                             │ │      │                            │   │
│                             │ │      ▼                            │   │
│                             │ │ High-confidence final score      │   │
│                             │ └─────────────────────────────────┘   │
└───────────────────────────┘                                        │
    │                                             │                   │
    ▼                                             ▼                   ▼
Running per-call risk profile (both tiers write to the same profile;
Tier 1 updates arrive first, Tier 2 confirms/escalates)
    │
    ├──► Cross-session consistency check (compares against historical
    │     genuine samples for this speaker/contact, where available)
    │
    ▼
Combined with classical feature extractor (parallel, display-only:
jitter/shimmer, spectral flatness, breath-pause ratio, formant naturalness)
    │
    ▼
{score, confidence, tier, per-encoder contributions, consistency check
 result, interpretable feature summary, timestamp}
    │
    ├──► Alerting layer (threshold-based, pre-transaction warning)
    └──► Audit trail (no raw audio stored)
```

### 4.2 Front-End Encoders — Tier 2 Full Ensemble (run in parallel on every segment)

| Encoder | Type | Purpose |
|---|---|---|
| wav2vec 2.0 (XLS-R, multilingual) | Pretrained SSL | Strong general phonetic representation; broad language coverage |
| WavLM (large) | Pretrained SSL | Better speaker/prosody-level cues (partially trained for speaker tasks) |
| RawNet3 | Raw-waveform CNN, trained end-to-end | Captures low-level signal artifacts (vocoder seams, phase discontinuities) that SSL models trained for semantic content can miss |

**Rationale:** Different encoders are sensitive to different artifact classes. No single encoder catches everything a cloning method can leave behind — ensembling closes coverage gaps.

### 4.3 Back-End: Graph Attention Network (AASIST-style)

- Replace flat pooling + MLP classification with a graph attention architecture applied to each encoder's output.
- Models **relational** artifacts across time and frequency (e.g., inconsistency between pitch contour and spectral envelope) rather than treating the embedding as a single point.
- Each of the three encoders gets its own AASIST-style back-end head.
- Pretrained AASIST checkpoints exist publicly and can be used as a starting point rather than training from scratch.

### 4.4 Score Fusion

- Each encoder+back-end pair outputs its own real/synthetic score.
- Combine via a **learned fusion layer** (small logistic regression or attention-weighted average) — not a simple average — so the system learns which encoder to trust more for which artifact type.
- Fusion layer is trained jointly/after the individual back-ends are trained.

### 4.5 Interpretability Layer (parallel, non-decisional)

- Independently compute classical, human-readable features on the same audio, purely for display — **not** part of the model's decision path:
  - Jitter / shimmer (pitch/amplitude micro-variation)
  - Spectral flatness
  - Silence / breath-pause ratio
  - Formant naturalness
- Present alongside the fused score as a "signal breakdown" (e.g., "prosody irregularity: high, spectral artifacts: high, breathing pattern: absent") so the output is defensible/explainable without compromising the accuracy-optimized decision pipeline.

### 4.6 Privacy / Data Handling

- Raw waveform exists only in-memory for the duration of processing.
- Nothing is persisted except the output object: `{score, confidence, per-encoder contributions, feature breakdown, timestamp}`.
- No raw audio, no intermediate embeddings, written to disk or logs.
- Per PS: support on-device/edge inference where deployment allows, to avoid central storage of sensitive audio — Tier 1's lightweight model is the natural candidate for edge deployment given its lower compute footprint; Tier 2's full ensemble is more realistically run server-side but still discards audio post-inference.
- Per PS: feature-only logging option — where even the derived feature breakdown is considered sensitive, the audit trail can be configured to log only the score/label/timestamp, dropping the feature breakdown.

### 4.7 Cross-Session Consistency Check (per PS: "Cross-session consistency checks")

- Where historical genuine samples exist for a given speaker/contact (e.g. a known executive's previously verified calls), maintain a lightweight speaker-similarity embedding (not full identification) computed from those samples.
- On a new call, compare the current segment's embedding against the stored reference to flag anomalies — a mismatch adds to the risk signal independent of the synthetic-speech score, catching cases where the voice is synthetic *or* simply doesn't match who it claims to be.
- This is a similarity/anomaly check, not a speaker-ID system — no claim of positively identifying the speaker, only flagging inconsistency against a known-good reference where one exists. Optional per call (only runs when a reference sample is available and consented to).

## 5. Training Strategy

### 5.1 Datasets

- ASVspoof 2019 + 2021 + 2024 combined (spans multiple years/attack-generation families → broader generalization).
- Optional: supplement with self-collected real recordings + outputs from a few open-source TTS/voice-cloning tools for demo-scenario-specific tuning.

### 5.2 Generalization to Unseen Attacks

- Hold out entire spoofing **algorithm families** during training (not just held-out samples of known families) and evaluate on them — this simulates real-world exposure to cloning methods not seen in training, which is the realistic threat model and the main failure mode of systems that overfit to known TTS/VC algorithms.

### 5.3 Robustness / Augmentation

- Codec compression (phone/VoIP-realistic)
- Resampling artifacts
- Background noise injection
- Rationale: real attack audio arrives via phone lines/VoIP; models trained only on clean lab audio degrade badly on compressed real-world calls.

## 6. Output Schema

```json
{
  "segment_timestamp_range": ["start_s", "end_s"],
  "tier1": {
    "score": 0.0,
    "label": "genuine | synthetic",
    "latency_ms": 0
  },
  "tier2": {
    "score": 0.0,
    "label": "genuine | synthetic",
    "confidence": 0.0,
    "encoder_contributions": {
      "wav2vec2_xlsr": 0.0,
      "wavlm_large": 0.0,
      "rawnet3": 0.0
    },
    "latency_ms": 0
  },
  "combined_risk_score": 0.0,
  "consistency_check": {
    "ran": true,
    "similarity_score": 0.0,
    "flag": "consistent | inconsistent | no_reference_available"
  },
  "feature_breakdown": {
    "prosody_irregularity": "low | medium | high",
    "spectral_artifacts": "low | medium | high",
    "breathing_pattern": "present | absent | irregular",
    "background_noise_consistency": "consistent | inconsistent"
  },
  "timestamp": "ISO-8601"
}
```

Notes:
- Both `tier1` and `tier2` populate on every segment; `tier1` arrives first (low latency), `tier2` follows shortly after and can revise `combined_risk_score`.
- `consistency_check` only produces a meaningful `flag` when a historical reference sample exists for the speaker/contact; otherwise `no_reference_available`.
- Each scored segment feeds the alerting layer independently, so a call's risk profile is a running sequence of these objects, not a single end-of-call verdict.

## 6.1 Alerting Layer (per PS: "Alerting and User Interaction Layer")

- Threshold-based alerting: configurable risk thresholds per scenario (e.g. lower threshold for high-value transaction approval calls, higher tolerance for routine support calls).
- Alert delivery: UI prompt to frontline staff during the call, with optional SMS/email/in-app notification for escalation.
- Pre-transaction warning: when risk crosses threshold before a sensitive action (fund transfer approval, confidential disclosure), system recommends secondary verification (call-back to a known number, MFA, escalation to supervisor) rather than blocking outright.
- Contextual enrichment (per PS): incorporate call metadata where available — call origin, known contact info, transaction context, historical fraud indicators — to adjust risk thresholds dynamically rather than relying on audio signal alone.

## 6.2 Integration Surface (per PS: "Platform and Integration APIs")

- REST/gRPC APIs and SDKs for integration with core banking systems, contact center platforms, enterprise communication tools, and telecom networks.
- Language-agnostic feature extraction with language-specific acoustic model variants to support multiple Indian languages/accents (PS requirement) — flagged as a required scope item for the SSL encoders and Tier 1 model, not just a nice-to-have.

## 7. Demo Scenario (for judging/evaluation)

A simulated live support/collections call: one leg is a real human speaker, the other is a cloned voice attempting to authorize an action (e.g., a refund or payment dispute). The system streams both through Tier 1 and Tier 2 continuously as the call plays, surfaces a running risk score in a UI, and fires a pre-transaction alert on the cloned leg before the "approval" step, with the interpretable feature breakdown and (where applicable) a consistency-check mismatch shown as supporting evidence.

## 8. Open Questions for Engineering

- Tier 1 model choice: which lightweight architecture (single small encoder vs. classical-feature classifier) hits the best accuracy/latency trade-off for continuous per-segment scoring — needs benchmarking.
- Fine-tune vs. freeze SSL encoder weights in Tier 2 (full fine-tune improves accuracy but costs significantly more compute/data).
- Whether to implement AASIST back-end from scratch or adapt existing public implementations/checkpoints.
- End-to-end latency budget for Tier 2 (three encoders + graph attention + fusion) running continuously on every segment — needs benchmarking to confirm it stays usefully "a beat behind" Tier 1 rather than falling arbitrarily far behind on long calls.
- Exact fusion layer architecture (logistic regression vs. attention-weighted) — decide empirically after individual back-ends are trained.
- How `combined_risk_score` reconciles Tier 1 and Tier 2 when they disagree (e.g. weighted by confidence, or Tier 2 always overrides once available).
- Compute/infra plan for running Tier 2's full ensemble continuously (not just on flagged segments) at call volume — this is a heavier infra commitment than the earlier flagged-only design and should be sized accordingly.

## 9. Success Metrics

- Equal Error Rate (EER) on held-out known attacks (standard ASVspoof metric).
- EER on held-out **unseen attack families** (generalization metric — the more important number for real-world credibility).
- Robustness delta: accuracy drop under codec/noise augmentation vs. clean audio.
