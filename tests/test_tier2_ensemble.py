import struct

from voice_detection.audio import AudioSegment
import pytest

from voice_detection.tier2_ensemble import Tier2ProductionEnsemble, inspect_quality, prosody_signal


def audio(values: list[int]) -> AudioSegment:
    return AudioSegment(b"".join(struct.pack("<h", item) for item in values), 16_000, len(values) / 16_000)


def test_tier2_refuses_to_start_without_all_trained_models() -> None:
    with pytest.raises(RuntimeError, match="TIER2_MANIFEST is required"):
        Tier2ProductionEnsemble()


def test_low_quality_audio_does_not_receive_high_confidence() -> None:
    quality = inspect_quality(audio([0] * 12_000))
    assert quality.low_energy is True and quality.score == 0
    assert prosody_signal(audio([0] * 12_000), quality)["prosody_score"] >= 0
