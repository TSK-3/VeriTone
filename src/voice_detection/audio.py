from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSegment:
    samples: bytes
    sample_rate: int
    duration_s: float


def decode_wav(payload: bytes) -> AudioSegment:
    """Decode supported WAV bytes without writing audio to disk."""
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            if width != 2:
                raise ValueError("only 16-bit PCM WAV audio is supported")
            if channels not in (1, 2):
                raise ValueError("only mono or stereo WAV audio is supported")
            samples = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("invalid WAV audio") from exc
    if channels == 2:
        values = struct.iter_unpack("<hh", samples)
        samples = b"".join(struct.pack("<h", (left + right) // 2) for left, right in values)
    if not samples:
        raise ValueError("audio contains no samples")
    return AudioSegment(samples=samples, sample_rate=rate, duration_s=len(samples) / (rate * 2))
