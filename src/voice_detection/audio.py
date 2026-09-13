"""Audio primitives: in-memory WAV decode, PCM16 packing, resampling, frame RMS.

All functions operate on raw PCM16 bytes / int samples only — nothing here
touches disk. ``decode_wav`` is the single entry point for submitted audio;
converted formats (8/24/32-bit, stereo) are normalized to 16-bit mono.
"""
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
    """Decode supported WAV bytes without writing audio to disk.

    Accepts 8/16/24/32-bit PCM mono or stereo; anything else is converted to
    16-bit in memory so user-recorded clips decode instead of erroring.
    """
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            if channels not in (1, 2):
                raise ValueError("only mono or stereo WAV audio is supported")
            if width == 2:
                samples = wav.readframes(frames)
            elif width == 1:  # 8-bit unsigned → 16-bit
                samples = bytes((byte - 128) * 256 for byte in wav.readframes(frames))
            elif width == 3:  # 24-bit → drop the lowest 8 bits
                raw = wav.readframes(frames)
                samples = b"".join(
                    struct.pack("<h", int.from_bytes(raw[i:i + 3], "little", signed=True) >> 8)
                    for i in range(0, len(raw) - 2, 3)
                )
            elif width == 4:  # 32-bit → scale down to 16-bit
                raw = wav.readframes(frames)
                samples = b"".join(
                    struct.pack("<h", max(-32768, min(32767, int.from_bytes(raw[i:i + 4], "little", signed=True) >> 16)))
                    for i in range(0, len(raw) - 3, 4)
                )
            else:
                raise ValueError("only 8/16/24/32-bit PCM WAV audio is supported")
    except (wave.Error, EOFError) as exc:
        raise ValueError("invalid WAV audio") from exc
    if channels == 2:
        values = struct.iter_unpack("<hh", samples)
        samples = b"".join(struct.pack("<h", (left + right) // 2) for left, right in values)
    if not samples:
        raise ValueError("audio contains no samples")
    return AudioSegment(samples=samples, sample_rate=rate, duration_s=len(samples) / (rate * 2))


def unpack_pcm16(samples: bytes) -> list[int]:
    """Decode little-endian PCM16 bytes into a list of samples."""
    return list(struct.unpack(f"<{len(samples) // 2}h", samples))


def pack_pcm16(values: list[int]) -> bytes:
    """Encode int samples as little-endian PCM16 bytes (endian-safe on any host)."""
    return struct.pack(f"<{len(values)}h", *values)


def resample_linear(values: list[int], src_rate: int, dst_rate: int) -> list[int]:
    """Linear-interpolation resample of int PCM samples (dependency-free)."""
    if src_rate == dst_rate or not values:
        return list(values)
    ratio = dst_rate / src_rate
    output: list[int] = []
    for i in range(int(len(values) * ratio)):
        position = i / ratio
        left = int(position)
        fraction = position - left
        a = values[left]
        b = values[left + 1] if left + 1 < len(values) else a
        output.append(int(a + (b - a) * fraction))
    return output


def frame_rms(values: list[int], sample_rate: int, frame_s: float = 0.02) -> list[float]:
    """RMS energy of consecutive ~frame_s windows (20 ms default)."""
    frame = max(1, int(sample_rate * frame_s))
    return [
        (sum(x * x for x in values[i:i + frame]) / max(1, len(values[i:i + frame]))) ** 0.5
        for i in range(0, len(values), frame)
    ]
