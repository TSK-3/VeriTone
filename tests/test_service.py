import struct

from voice_detection.audio import AudioSegment
from voice_detection.aggregation import RunningRiskAggregator
from voice_detection.service import DetectionService


def pcm(values: list[int]) -> bytes:
    return b"".join(struct.pack("<h", item) for item in values)


def test_analysis_exposes_prd_contract_without_features_when_disabled() -> None:
    result =     DetectionService(alert_threshold=0, require_tier1_checkpoint=False).analyze(
        AudioSegment(pcm([50, -50] * 8000), 16000, 1.0), start_s=2.5, speaker_similarity=0.4, include_features=False
    )
    record = result.audit_record()
    assert record["segment_timestamp_range"] == (2.5, 3.5)
    assert set(record["tier2"]["encoder_contributions"]) == {"wav2vec2_xlsr", "wavlm_large", "rawnet3"}
    assert record["consistency_check"]["flag"] == "inconsistent"
    assert record["feature_breakdown"] is None
    assert record["alert"] is False
    assert record["running_risk_score"] == 0
    assert "audio" not in str(record).lower()


def test_similarity_must_be_a_probability() -> None:
    audio = AudioSegment(pcm([1, -1] * 100), 16000, 0.0125)
    try:
        DetectionService(require_tier1_checkpoint=False).analyze(audio, 0, speaker_similarity=1.2)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("invalid similarity must be rejected")


def test_alert_requires_aggregated_evidence() -> None:
    audio = AudioSegment(pcm([50, -50] * 8000), 16000, 1.0)
    aggregator = RunningRiskAggregator(alert_threshold=0.1, min_evidence=3)
    result = DetectionService(require_tier1_checkpoint=False).analyze(audio, 0)
    assert aggregator.add(result)[2] is False
    assert aggregator.add(result)[2] is False
    assert aggregator.add(result)[2] is True
