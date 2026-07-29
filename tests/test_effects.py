"""Tests for the effects chain and ADSR envelope."""

import numpy as np
import pytest

import sound_generator as generate
from tests.helpers import band_power_ratio, spectral_centroid, spectrum

SAMPLE_RATE = 8000


def test_adsr_defaults_are_identity() -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE)
    assert np.array_equal(generate.apply_adsr(samples, SAMPLE_RATE), samples)


def test_adsr_shapes_envelope() -> None:
    sample_rate = 1000
    samples = np.ones(sample_rate, dtype=np.float32)
    out = generate.apply_adsr(
        samples, sample_rate, attack=0.1, decay=0.1, sustain=0.5, release=0.2
    )
    assert len(out) == len(samples)
    assert out[0] == 0.0
    assert out[100] == pytest.approx(1.0)
    assert out[500] == pytest.approx(0.5)
    assert out[-1] == 0.0


def test_adsr_longer_than_clip_is_scaled_to_fit() -> None:
    sample_rate = 1000
    samples = np.ones(sample_rate, dtype=np.float32)
    out = generate.apply_adsr(
        samples, sample_rate, attack=1.0, decay=1.0, sustain=0.5, release=2.0
    )
    assert len(out) == len(samples)
    assert out[0] == 0.0
    assert out[-1] == 0.0


def test_effects_defaults_are_identity() -> None:
    samples = generate.generate_graincloud(0.5, SAMPLE_RATE, seed=1)
    assert np.array_equal(generate.apply_effects(samples, SAMPLE_RATE), samples)


def test_drive_raises_rms_within_peak_bound() -> None:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    sine = (0.8 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    driven = generate.apply_effects(sine, SAMPLE_RATE, drive=0.8)
    clean_rms = float(np.sqrt(np.mean(sine**2)))
    driven_rms = float(np.sqrt(np.mean(driven**2)))
    assert driven_rms > clean_rms + 0.05
    assert np.max(np.abs(driven)) <= 1.0 + 1e-6


def test_fold_adds_harmonics() -> None:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    sine = (0.9 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    folded = generate.apply_effects(sine, SAMPLE_RATE, fold=3.0)
    _, clean_mag = spectrum(sine, SAMPLE_RATE)
    _, folded_mag = spectrum(folded, SAMPLE_RATE)
    clean_partials = int(np.sum(clean_mag > 0.05 * np.max(clean_mag)))
    folded_partials = int(np.sum(folded_mag > 0.05 * np.max(folded_mag)))
    assert clean_partials == 1
    assert folded_partials > 3


def test_bitcrush_limits_unique_values() -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE)
    crushed = generate.apply_effects(samples, SAMPLE_RATE, crush_bits=3)
    assert len(np.unique(crushed)) <= 2**3 + 1


def test_rate_crush_holds_samples() -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE)
    held = generate.apply_effects(samples, SAMPLE_RATE, crush_rate=SAMPLE_RATE / 8)
    equal_neighbours = float(np.mean(np.diff(held) == 0.0))
    assert equal_neighbours > 0.7


def test_effects_preserve_stereo_shape() -> None:
    samples = generate.generate_swarm(0.5, SAMPLE_RATE, seed=1, stereo=True)
    out = generate.apply_effects(
        samples, SAMPLE_RATE, drive=0.3, fold=1.0, crush_bits=8, crush_rate=4000.0
    )
    assert out.shape == samples.shape
    assert out.dtype == np.float32


def test_filter_off_is_bitexact() -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE)
    with_params = generate.apply_effects(
        samples, SAMPLE_RATE, cutoff=500.0, resonance=8.0, cutoff_end=4000.0
    )
    assert np.array_equal(with_params, generate.apply_effects(samples, SAMPLE_RATE))


def test_lowpass_kills_highs() -> None:
    noise = generate.generate_random_audio(1.0, SAMPLE_RATE, seed=1)
    plain = band_power_ratio(noise, SAMPLE_RATE)
    filtered = generate.apply_effects(
        noise, SAMPLE_RATE, filter_type="lowpass", cutoff=300.0
    )
    assert band_power_ratio(filtered, SAMPLE_RATE) > 10 * plain


def test_highpass_kills_lows() -> None:
    noise = generate.generate_random_audio(1.0, SAMPLE_RATE, seed=1)
    plain = band_power_ratio(noise, SAMPLE_RATE)
    filtered = generate.apply_effects(
        noise, SAMPLE_RATE, filter_type="highpass", cutoff=2000.0
    )
    assert band_power_ratio(filtered, SAMPLE_RATE) < plain / 10


def test_bandpass_concentrates_energy() -> None:
    noise = generate.generate_random_audio(1.0, SAMPLE_RATE, seed=1)
    filtered = generate.apply_effects(
        noise, SAMPLE_RATE, filter_type="bandpass", cutoff=800.0, resonance=8.0
    )
    freqs, mag = spectrum(filtered, SAMPLE_RATE)
    assert freqs[np.argmax(mag)] == pytest.approx(800.0, abs=200.0)


def test_resonance_boosts_cutoff_region() -> None:
    noise = generate.generate_random_audio(1.0, SAMPLE_RATE, seed=1)
    flat = generate.apply_effects(
        noise, SAMPLE_RATE, filter_type="lowpass", cutoff=1000.0, resonance=0.707
    )
    resonant = generate.apply_effects(
        noise, SAMPLE_RATE, filter_type="lowpass", cutoff=1000.0, resonance=10.0
    )
    freqs, flat_mag = spectrum(flat, SAMPLE_RATE)
    _, res_mag = spectrum(resonant, SAMPLE_RATE)
    band = (freqs > 900) & (freqs < 1100)
    assert res_mag[band].mean() / flat_mag[band].mean() >= 2.0


def test_sweep_moves_spectral_centroid() -> None:
    noise = generate.generate_random_audio(2.0, SAMPLE_RATE, seed=1)
    swept = generate.apply_effects(
        noise, SAMPLE_RATE, filter_type="lowpass", cutoff=200.0, cutoff_end=3000.0
    )
    quarter = len(swept) // 4
    first = spectral_centroid(swept[:quarter], SAMPLE_RATE)
    last = spectral_centroid(swept[-quarter:], SAMPLE_RATE)
    assert last > 1.5 * first


def test_filter_preserves_stereo_shape_and_peak() -> None:
    samples = generate.generate_swarm(0.5, SAMPLE_RATE, seed=1, stereo=True)
    out = generate.apply_effects(
        samples, SAMPLE_RATE, filter_type="lowpass", cutoff=500.0, resonance=12.0
    )
    assert out.shape == samples.shape
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_filter_rejects_unknown_type() -> None:
    samples = generate.generate_fm(0.1, SAMPLE_RATE)
    with pytest.raises(ValueError, match="filter type"):
        generate.apply_effects(samples, SAMPLE_RATE, filter_type="notch")


def test_adsr_broadcasts_over_stereo() -> None:
    sample_rate = 1000
    samples = np.ones((sample_rate, 2), dtype=np.float32)
    out = generate.apply_adsr(
        samples, sample_rate, attack=0.1, decay=0.1, sustain=0.5, release=0.2
    )
    assert out.shape == samples.shape
    assert np.array_equal(out[0], [0.0, 0.0])
    assert out[500, 0] == pytest.approx(0.5)
    assert out[500, 1] == pytest.approx(0.5)
    assert np.array_equal(out[-1], [0.0, 0.0])
