import struct

from voice_detection.audio import AudioSegment
from voice_detection.aggregation import RunningRiskAggregator
from voice_detection.service import DetectionService
from voice_detection.tier2_ensemble import AudioQuality, EnsembleOutput


def pcm(values: list[int]) -> bytes:
    return b"".join(struct.pack("<h", item) for item in values)


class StubTier2:
    """Test double only; runtime Tier 2 never uses a fallback."""
    def score(self, _audio: AudioSegment) -> EnsembleOutput:
        return EnsembleOutput(.8, .9, {"wav2vec2_xlsr": .8, "wavlm_large": .8, "rawnet3": .8, "aasist": .8},
                              {"prosody_score": .7}, .01, AudioQuality(.9, True, False, 0, .1),
                              {"wav2vec2_xlsr": "onnx", "wavlm_large": "onnx", "rawnet3": "onnx", "aasist": "onnx"})


def detector() -> DetectionService:
    return DetectionService(tier2=StubTier2())


def test_analysis_exposes_prd_contract_without_features_when_disabled() -> None:
    result = detector().analyze(
        AudioSegment(pcm([50, -50] * 8000), 16000, 1.0), start_s=2.5, speaker_similarity=0.4, include_features=False
    )
    record = result.audit_record()
    assert record["segment_timestamp_range"] == (2.5, 3.5)
    assert set(record["tier2"]["encoder_contributions"]) == {"wav2vec2_xlsr", "wavlm_large", "rawnet3", "aasist"}
    assert record["consistency_check"]["flag"] == "inconsistent"
    assert record["feature_breakdown"] is None
    assert record["alert"] is False
    assert record["running_risk_score"] == 0
    assert "audio" not in str(record).lower()


def test_similarity_must_be_a_probability() -> None:
    audio = AudioSegment(pcm([1, -1] * 100), 16000, 0.0125)
    try:
        detector().analyze(audio, 0, speaker_similarity=1.2)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("invalid similarity must be rejected")


def test_alert_requires_aggregated_evidence() -> None:
    audio = AudioSegment(pcm([50, -50] * 8000), 16000, 1.0)
    aggregator = RunningRiskAggregator(alert_threshold=0.1, min_evidence=3)
    result = detector().analyze(audio, 0)
    assert aggregator.add(result)[2] is False
    assert aggregator.add(result)[2] is False
    assert aggregator.add(result)[2] is True
