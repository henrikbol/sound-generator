"""Tests for the tone generators: noise, FM, and harmonics."""

import numpy as np
import pytest

import sound_generator as generate
from tests.helpers import band_power_ratio, spectrum

SAMPLE_RATE = 8000


def test_fm_zero_index_is_pure_sine() -> None:
    samples = generate.generate_fm(1.0, SAMPLE_RATE, carrier=440.0, index=0.0)
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    assert freqs[np.argmax(mag)] == pytest.approx(440.0, abs=1.0)
    significant = np.sum(mag > 0.1 * np.max(mag))
    assert significant == 1


def test_fm_index_creates_sidebands() -> None:
    samples = generate.generate_fm(
        1.0, SAMPLE_RATE, carrier=440.0, ratio=2.0, index=5.0
    )
    _, mag = spectrum(samples, SAMPLE_RATE)
    significant = np.sum(mag > 0.1 * np.max(mag))
    assert significant > 3


def test_harmonics_rolloff() -> None:
    samples = generate.generate_harmonics(
        1.0, SAMPLE_RATE, freq=100.0, count=4, decay=1.0
    )
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    fundamental = mag[np.argmin(np.abs(freqs - 100.0))]
    second = mag[np.argmin(np.abs(freqs - 200.0))]
    assert fundamental / second == pytest.approx(2.0, rel=0.1)


def test_harmonics_skips_partials_above_nyquist() -> None:
    samples = generate.generate_harmonics(1.0, SAMPLE_RATE, freq=3000.0, count=8)
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    significant = np.sum(mag > 0.1 * np.max(mag))
    assert significant == 1
    assert freqs[np.argmax(mag)] == pytest.approx(3000.0, abs=1.0)


def test_noise_colors_have_expected_spectral_tilt() -> None:
    ratios = {
        color: band_power_ratio(
            generate.generate_random_audio(1.0, 44100, color=color, seed=42), 44100
        )
        for color in generate.NOISE_COLORS
    }
    assert ratios["white"] < 3
    assert ratios["pink"] > 10
    assert ratios["brown"] > ratios["pink"]


@pytest.mark.parametrize("color", generate.NOISE_COLORS)
def test_seeded_noise_is_deterministic(color: str) -> None:
    first = generate.generate_random_audio(0.1, SAMPLE_RATE, color=color, seed=7)
    second = generate.generate_random_audio(0.1, SAMPLE_RATE, color=color, seed=7)
    other = generate.generate_random_audio(0.1, SAMPLE_RATE, color=color, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_random_audio_rejects_unknown_color() -> None:
    with pytest.raises(ValueError, match="noise color"):
        generate.generate_random_audio(0.1, SAMPLE_RATE, color="mauve")


def test_harmonics_stretch_raises_second_partial() -> None:
    samples = generate.generate_harmonics(
        1.0, SAMPLE_RATE, freq=200.0, count=2, decay=0.0, stretch=0.02, seed=1
    )
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    band = (freqs > 300) & (freqs < 500)
    peak_freq = freqs[band][np.argmax(mag[band])]
    expected = 200.0 * 2 * np.sqrt(1.0 + 0.02 * 4)
    assert peak_freq == pytest.approx(expected, abs=2.0)
    assert peak_freq > 403.0


def test_harmonics_odd_only_skips_even_partials() -> None:
    samples = generate.generate_harmonics(
        1.0, SAMPLE_RATE, freq=100.0, count=4, decay=1.0, odd_only=True, seed=1
    )
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    fundamental = mag[np.argmin(np.abs(freqs - 100.0))]
    second = mag[np.argmin(np.abs(freqs - 200.0))]
    third = mag[np.argmin(np.abs(freqs - 300.0))]
    assert second < 0.05 * fundamental
    assert third > 0.1 * fundamental


def test_harmonics_drift_respects_peak_bound() -> None:
    samples = generate.generate_harmonics(1.0, SAMPLE_RATE, drift=1.0, seed=1)
    assert np.max(np.abs(samples)) <= 0.8 + 1e-6
