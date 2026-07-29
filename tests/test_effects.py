"""Tests for the effects chain and ADSR envelope."""

import numpy as np
import pytest

import sound_generator as generate
from tests.helpers import spectrum

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
