"""Tests for the texture generators: swarm, graincloud, crackle, and pluck."""

import numpy as np
import pytest

import sound_generator as generate
from tests.helpers import spectrum

SAMPLE_RATE = 8000


def test_swarm_single_voice_peaks_at_freq() -> None:
    samples = generate.generate_swarm(
        1.0,
        SAMPLE_RATE,
        freq=500.0,
        voices=1,
        wave_shape="sine",
        detune=0.0,
        drift=0.0,
        seed=1,
    )
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    assert freqs[np.argmax(mag)] == pytest.approx(500.0, abs=2.0)


def test_graincloud_zero_spread_peaks_at_pitch() -> None:
    samples = generate.generate_graincloud(
        1.0, SAMPLE_RATE, density=60.0, pitch=440.0, spread=0.0, seed=1
    )
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    # 60 ms Hanning grains have a wide spectral main lobe, so the argmax
    # wanders within it — assert the peak sits inside the lobe.
    assert freqs[np.argmax(mag)] == pytest.approx(440.0, abs=25.0)


def test_crackle_is_sparse_and_scales_with_density() -> None:
    sparse = generate.generate_crackle(2.0, SAMPLE_RATE, density=5.0, seed=1)
    dense = generate.generate_crackle(2.0, SAMPLE_RATE, density=200.0, seed=1)
    quiet_fraction = float(np.mean(np.abs(sparse) < 0.04))
    assert quiet_fraction > 0.5
    assert float(np.mean(np.abs(dense) < 0.04)) < quiet_fraction


def test_pluck_peaks_at_quantised_freq() -> None:
    samples = generate.generate_pluck(
        2.0, SAMPLE_RATE, freq=220.0, decay=0.8, interval=0.0, seed=1
    )
    tail = samples[len(samples) // 2 :]
    freqs, mag = spectrum(tail, SAMPLE_RATE)
    period = round(SAMPLE_RATE / 220.0 - 0.5)
    expected = SAMPLE_RATE / (period + 0.5)  # loop filter adds half a sample
    assert freqs[np.argmax(mag)] == pytest.approx(expected, abs=3.0)


def test_pluck_tail_decays() -> None:
    samples = generate.generate_pluck(
        2.0, SAMPLE_RATE, freq=220.0, decay=0.5, interval=0.0, seed=1
    )
    half = len(samples) // 2
    first_rms = float(np.sqrt(np.mean(samples[:half] ** 2)))
    second_rms = float(np.sqrt(np.mean(samples[half:] ** 2)))
    assert second_rms < first_rms * 0.5
