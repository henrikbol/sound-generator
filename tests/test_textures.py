"""Tests for the texture generators: swarm, graincloud, crackle, and pluck."""

import numpy as np
import pytest

import sound_generator as generate
from tests.helpers import spectral_centroid, spectrum

SAMPLE_RATE = 8000


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2)))


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


def test_riser_gets_louder() -> None:
    samples = generate.generate_riser(2.0, SAMPLE_RATE, seed=1)
    quarter = len(samples) // 4
    assert _rms(samples[-quarter:]) > 2.0 * _rms(samples[:quarter])


def test_riser_gets_brighter() -> None:
    samples = generate.generate_riser(2.0, SAMPLE_RATE, noise_mix=1.0, seed=1)
    quarter = len(samples) // 4
    first = spectral_centroid(samples[:quarter], SAMPLE_RATE)
    last = spectral_centroid(samples[-quarter:], SAMPLE_RATE)
    assert last > first


def test_riser_pitch_sweeps_up() -> None:
    samples = generate.generate_riser(
        2.0,
        SAMPLE_RATE,
        freq=110.0,
        octaves=2.0,
        voices=1,
        detune=0.0,
        noise_mix=0.0,
        seed=1,
    )
    quarter = len(samples) // 4
    first_freqs, first_mag = spectrum(samples[:quarter], SAMPLE_RATE)
    last_freqs, last_mag = spectrum(samples[-quarter:], SAMPLE_RATE)
    first_peak = first_freqs[np.argmax(first_mag)]
    last_peak = last_freqs[np.argmax(last_mag)]
    # Swept saw over a quarter-clip window — assert direction and rough span,
    # not exact bins (windowed material needs wide tolerances).
    assert first_peak < 250.0
    assert last_peak > 280.0
    assert last_peak / first_peak > 1.8


def test_riser_down_fades_out() -> None:
    samples = generate.generate_riser(2.0, SAMPLE_RATE, direction="down", seed=1)
    quarter = len(samples) // 4
    assert _rms(samples[:quarter]) > 2.0 * _rms(samples[-quarter:])


def test_riser_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="riser direction"):
        generate.generate_riser(0.5, SAMPLE_RATE, direction="sideways")
