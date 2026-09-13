"""Play a prerecorded CLONED voice into a real Twilio call and catch it live.

``GET /v1/demo/scam-audio`` serves an 8 kHz µ-law WAV built from the repo's
English TTS-clone clips (MLAAD). ``GET /twiml/scam-call`` returns TwiML that
starts a bidirectional Media Stream + transcription and ``<Play>``s that audio
into the call on loop. The detector therefore hears its own cloned attacker:
VAD-gated segments are scored live and Twilio's transcription of the *outbound*
track drives the trigger-word SMS rules — the full prevention loop with zero
manual audio handling.

For richer AI voices: generate any TTS clip (ElevenLabs, Cartesia, …), save it as
``data/demo_scam/*.wav`` (16-bit PCM), and it is picked up automatically; clips
there take priority over the bundled MLAAD samples.
"""
from __future__ import annotations

import struct
from pathlib import Path

from .audio import AudioSegment, decode_wav, resample_linear, unpack_pcm16
from .twilio_stream import pcm16_to_mulaw

ROOT = Path(__file__).parents[2]
AUDIO_RATE = 8_000
_CACHE: dict[str, bytes] = {}


def _load_clips() -> list[AudioSegment]:
    """Preferred: data/demo_scam/*.wav (user-supplied TTS); fallback: MLAAD fakes."""
    clips: list[AudioSegment] = []
    custom = ROOT / "data" / "demo_scam"
    if custom.is_dir():
        for path in sorted(custom.glob("*.wav")):
            try:
                clip = decode_wav(path.read_bytes())
                if clip.duration_s <= 30:
                    clips.append(clip)
            except ValueError:
                continue
        if clips:
            return clips
    from .demo_scenario import _load_whole, _pick_file  # reuse the clean-clip selection
    for index in range(4):
        clips.append(_load_whole(_pick_file("spoof", index)))
    return clips


def _wav_bytes(pcm: bytes, *, mulaw: bool) -> bytes:
    """Wrap raw 8 kHz mono audio in a RIFF/WAVE header (µ-law or 16-bit PCM).

    PCM 16-bit is the most universally accepted <Play> format — some Twilio
    media parsers reject hand-rolled µ-law headers.
    """
    fmt = (7, 1, AUDIO_RATE, AUDIO_RATE, 1, 8) if mulaw else (1, 1, AUDIO_RATE, AUDIO_RATE * 2, 2, 16)
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, *fmt)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def build_scam_audio(encoding: str = "pcm16") -> bytes:
    """Concatenate cloned-voice clips into an 8 kHz WAV (cached per encoding)."""
    if encoding in _CACHE:
        return _CACHE[encoding]
    pcm8: list[int] = []
    gap = [0] * int(AUDIO_RATE * 0.6)  # natural pause between script lines
    for clip in _load_clips():
        mono8 = resample_linear(unpack_pcm16(clip.samples), clip.sample_rate, AUDIO_RATE)
        pcm8.extend(mono8[: AUDIO_RATE * 12])  # cap each line at 12 s
        pcm8.extend(gap)
    if encoding == "mulaw":
        _CACHE[encoding] = _wav_bytes(bytes(pcm16_to_mulaw(v) for v in pcm8), mulaw=True)
    else:  # "pcm16"
        _CACHE[encoding] = _wav_bytes(struct.pack(f"<{len(pcm8)}h", *pcm8), mulaw=False)
    return _CACHE[encoding]


def wav_bytes_for_twilio(encoding: str = "pcm16") -> tuple[bytes, str]:
    return build_scam_audio(encoding), "audio/wav"
