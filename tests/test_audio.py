"""Tests for the shared audio primitives (pack/unpack, resample, RMS, decode)."""
import io
import struct
import wave

import pytest

from voice_detection.audio import decode_wav, frame_rms, pack_pcm16, resample_linear, unpack_pcm16


def wav_bytes(samples: list[int], rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pack_pcm16(samples))
    return buffer.getvalue()


def test_pack_unpack_roundtrip() -> None:
    values = [0, 1, -1, 32767, -32768]
    assert unpack_pcm16(pack_pcm16(values)) == values


def test_resample_linear_identity_and_downsample() -> None:
    values = [100, 200, 300, 400]
    assert resample_linear(values, 16_000, 16_000) == values
    assert resample_linear(values, 16_000, 8_000) == [100, 300]


def test_decode_wav_measures_duration_and_rejects_garbage() -> None:
    audio = decode_wav(wav_bytes([100, -100] * 16_000))
    assert audio.sample_rate == 16_000 and audio.duration_s == 2.0
    with pytest.raises(ValueError):
        decode_wav(b"not a wav file")


def test_frame_rms_energy_levels() -> None:
    assert frame_rms([3000] * 16_000, 16_000)[0] == 3000
    assert frame_rms([0] * 16_000, 16_000)[0] == 0
    # 1 s of audio at 16 kHz → 50 frames of 20 ms (with a short final frame).
    assert len(frame_rms([100] * 16_000, 16_000)) == 50