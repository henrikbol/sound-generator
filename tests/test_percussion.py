"""Tests for the percussion generators: drum one-shots and modal resonators."""

import numpy as np
import pytest

import sound_generator as generate
from tests.helpers import band_power_ratio, spectral_centroid, spectrum

SAMPLE_RATE = 8000


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2)))


def _zero_crossing_rate(samples: np.ndarray) -> float:
    return float(np.mean(np.diff(np.sign(samples)) != 0))


def test_kick_pitch_descends() -> None:
    samples = generate.generate_drum(0.5, SAMPLE_RATE, drum_type="kick", seed=1)
    early = samples[: int(0.025 * SAMPLE_RATE)]
    late = samples[int(0.1 * SAMPLE_RATE) : int(0.2 * SAMPLE_RATE)]
    assert _zero_crossing_rate(early) > 1.5 * _zero_crossing_rate(late)


def test_kick_decays() -> None:
    samples = generate.generate_drum(0.5, SAMPLE_RATE, drum_type="kick", seed=1)
    half = len(samples) // 2
    assert _rms(samples[half:]) < 0.5 * _rms(samples[:half])


def test_hihat_is_high_band() -> None:
    samples = generate.generate_drum(0.3, SAMPLE_RATE, drum_type="hihat")
    assert band_power_ratio(samples, SAMPLE_RATE) < 1.0


def test_sub_is_low_band() -> None:
    samples = generate.generate_drum(1.0, SAMPLE_RATE, drum_type="sub", seed=1)
    assert band_power_ratio(samples, SAMPLE_RATE) > 10.0


def test_snare_has_tone_and_noise() -> None:
    samples = generate.generate_drum(0.5, SAMPLE_RATE, drum_type="snare", seed=1)
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    power = mag**2
    total = power.sum()
    low = power[freqs < 500].sum()
    high = power[freqs > 1500].sum()
    assert low > 0.05 * total
    assert high > 0.05 * total


def test_drum_pads_short_hit_with_silence() -> None:
    samples = generate.generate_drum(1.0, SAMPLE_RATE, drum_type="hihat", decay=0.0)
    tail = samples[-int(0.2 * SAMPLE_RATE) :]
    assert float(np.max(np.abs(tail))) < 1e-3


def test_drum_stereo_duplicates_mono() -> None:
    samples = generate.generate_drum(
        0.5, SAMPLE_RATE, drum_type="kick", seed=1, stereo=True
    )
    assert samples.shape == (int(0.5 * SAMPLE_RATE), 2)
    assert np.array_equal(samples[:, 0], samples[:, 1])


def test_drum_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="drum type"):
        generate.generate_drum(0.5, SAMPLE_RATE, drum_type="tom")


def test_modal_partials_at_material_ratios() -> None:
    samples = generate.generate_modal(
        1.0, SAMPLE_RATE, freq=300.0, material="bar", decay=0.5, seed=1
    )
    freqs, mag = spectrum(samples, SAMPLE_RATE)
    for ratio in (1.0, 2.756, 5.404):
        expected = 300.0 * ratio
        window = (freqs > expected * 0.92) & (freqs < expected * 1.08)
        peak_freq = freqs[window][np.argmax(mag[window])]
        assert peak_freq == pytest.approx(expected, abs=10.0)


def test_modal_wood_decays_faster_than_bowl() -> None:
    def tail_ratio(material: str) -> float:
        samples = generate.generate_modal(
            2.0, SAMPLE_RATE, material=material, decay=0.5, seed=1
        )
        half = len(samples) // 2
        return _rms(samples[half:]) / _rms(samples[:half])

    assert tail_ratio("wood") < tail_ratio("bowl")


def test_modal_interval_restrikes() -> None:
    single = generate.generate_modal(2.0, SAMPLE_RATE, decay=0.2, seed=1)
    restruck = generate.generate_modal(
        2.0, SAMPLE_RATE, decay=0.2, interval=0.3, seed=1
    )
    quarter = len(single) // 4
    assert _rms(restruck[-quarter:]) > 5.0 * _rms(single[-quarter:])


def test_modal_brightness_darkens() -> None:
    dark = generate.generate_modal(1.0, SAMPLE_RATE, brightness=0.2, seed=1)
    bright = generate.generate_modal(1.0, SAMPLE_RATE, brightness=1.0, seed=1)
    assert spectral_centroid(dark, SAMPLE_RATE) < spectral_centroid(bright, SAMPLE_RATE)


def test_modal_skips_modes_above_nyquist() -> None:
    samples = generate.generate_modal(0.5, SAMPLE_RATE, freq=3000.0, material="bar")
    assert samples.shape == (int(0.5 * SAMPLE_RATE),)
    assert np.all(np.isfinite(samples))
    assert float(np.max(np.abs(samples))) <= 1.0


def test_modal_rejects_unknown_material() -> None:
    with pytest.raises(ValueError, match="modal material"):
        generate.generate_modal(0.5, SAMPLE_RATE, material="steel")
