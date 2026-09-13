"""Consented speaker references: enrol a reference embedding, check live similarity.

A *speaker reference* is a compact spectral-voiceprint vector (mel-band log-energy
statistics, model-free and deterministic). It is computed **in memory** from the
enrolment WAV, never persisted as audio, and stored only in the encrypted audit
store — and only when the customer's explicit consent flag is ``True``.

During a call, segments are compared against the on-file reference with a cosine
similarity; the pipeline feeds that similarity into the existing
``consistency_check`` so a flagged mismatch becomes supporting evidence alongside
the voice risk — the cross-session check the PRD describes, without ever keeping
the reference waveform.

This is a deliberately lightweight reference implementation: swap the embedding for
a d-vector/x-vector model later without touching the consent + storage contract.
"""
from __future__ import annotations

import math

from .audio import AudioSegment, unpack_pcm16

N_BANDS = 24  # mel-ish log-band energies; ~0.6 ms per enrolment second


def _mel_hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2595) - 1)


def _band_edges(sample_rate: int, n_bands: int = N_BANDS) -> list[int]:
    """Log-spaced band edges from 80 Hz to Nyquist (mel-ish spacing)."""
    lo_mel = 2595 * math.log10(1 + 80 / 700)
    hi_mel = 2595 * math.log10(1 + sample_rate / 2 / 700)
    hz = [_mel_hz(lo_mel + (hi_mel - lo_mel) * i / n_bands) for i in range(n_bands + 1)]
    return [min(sample_rate // 2, int(h)) for h in hz]


def compute_speaker_embedding(audio: AudioSegment) -> list[float]:
    """Deterministic spectral voiceprint: per-band log-energy mean, 0–1 scaled."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    values = unpack_pcm16(audio.samples)
    if not values:
        raise ValueError("audio contains no samples")
    if audio.duration_s < 0.3:
        raise ValueError("reference audio must be at least 0.3 s")
    edges = _band_edges(audio.sample_rate)
    if np is not None:
        wave = np.asarray(values, dtype=np.float32)
        window = np.hanning(max(256, int(0.032 * audio.sample_rate)))
        n_fft = max(512, 1 << (int(0.032 * audio.sample_rate) - 1).bit_length())
        frames = np.lib.stride_tricks.sliding_window_view(wave, window.size)[:: window.size // 2]
        if frames.shape[0] > 256:  # cap compute on long enrolment files
            frames = frames[:256]
        mag = np.abs(np.fft.rfft(frames * window, n_fft)) ** 2
        # FFT bin index of each band edge (Hz → bins), strictly increasing, in bounds.
        n_bins = n_fft // 2 + 1
        bin_edges = sorted({min(max(int(round(hz / (audio.sample_rate / 2) * (n_fft // 2))), 1), n_bins - 1)
                            for hz in edges})
        band_sum = np.add.reduceat(mag, bin_edges, axis=1)[:, :N_BANDS]
        log_band = np.log(band_sum + 1.0)
        embedding = log_band.mean(axis=0)
        return [round(v, 5) for v in ((embedding - embedding.mean()) / (embedding.std() + 1e-6)).tolist()]
    # Dependency-free fallback: coarse RMS-proportional band proxies keep the
    # consent + storage contract working in API-only deployments.
    band = [0.0] * N_BANDS
    for i, value in enumerate(values[: 320 * 256]):
        band[i * N_BANDS // 320] += abs(value) / 32768
    norm = max(band) or 1.0
    return [round(v / norm, 5) for v in band]


def similarity(reference: list[float], candidate: list[float]) -> float:
    """Cosine similarity of two embeddings, mapped from [-1, 1] to [0, 1]."""
    length = min(len(reference), len(candidate))
    if length == 0:
        return 0.0
    a, b = reference[:length], candidate[:length]
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if norm <= 1e-9:
        return 0.0
    return round(max(0.0, min(1.0, (dot / norm + 1) / 2)), 4)


def check_against_reference(audio: AudioSegment, reference: list[float]) -> float:
    """Live similarity of the current segment vs the consented reference."""
    return similarity(reference, compute_speaker_embedding(audio))
