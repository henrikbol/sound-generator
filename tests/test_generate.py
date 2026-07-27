"""Tests for the generate.py synthesis modes."""

import wave
from collections.abc import Callable

import numpy as np
import pytest

import generate

SAMPLE_RATE = 8000

Generator = Callable[[float, int], np.ndarray]

GENERATORS = [
    pytest.param(
        lambda d, sr: generate.generate_random_audio(d, sr, seed=1), id="white"
    ),
    pytest.param(
        lambda d, sr: generate.generate_random_audio(d, sr, color="pink", seed=1),
        id="pink",
    ),
    pytest.param(
        lambda d, sr: generate.generate_random_audio(d, sr, color="brown", seed=1),
        id="brown",
    ),
    pytest.param(lambda d, sr: generate.generate_bytebeat(d, sr), id="bytebeat"),
    pytest.param(lambda d, sr: generate.generate_fm(d, sr), id="fm"),
    pytest.param(lambda d, sr: generate.generate_harmonics(d, sr), id="harmonics"),
    pytest.param(lambda d, sr: generate.generate_swarm(d, sr, seed=1), id="swarm"),
    pytest.param(
        lambda d, sr: generate.generate_graincloud(d, sr, seed=1), id="graincloud"
    ),
    pytest.param(lambda d, sr: generate.generate_crackle(d, sr, seed=1), id="crackle"),
    pytest.param(lambda d, sr: generate.generate_pluck(d, sr, seed=1), id="pluck"),
]

STEREO_GENERATORS = [
    pytest.param(
        lambda d, sr: generate.generate_random_audio(d, sr, seed=1, stereo=True),
        id="white",
    ),
    pytest.param(
        lambda d, sr: generate.generate_bytebeat(d, sr, stereo=True), id="bytebeat"
    ),
    pytest.param(lambda d, sr: generate.generate_fm(d, sr, stereo=True), id="fm"),
    pytest.param(
        lambda d, sr: generate.generate_harmonics(d, sr, seed=1, stereo=True),
        id="harmonics",
    ),
    pytest.param(
        lambda d, sr: generate.generate_swarm(d, sr, seed=1, stereo=True), id="swarm"
    ),
    pytest.param(
        lambda d, sr: generate.generate_graincloud(d, sr, seed=1, stereo=True),
        id="graincloud",
    ),
    pytest.param(
        lambda d, sr: generate.generate_crackle(d, sr, seed=1, stereo=True),
        id="crackle",
    ),
    pytest.param(
        lambda d, sr: generate.generate_pluck(d, sr, seed=1, stereo=True), id="pluck"
    ),
]

SEEDED_TEXTURES = [
    pytest.param(
        lambda s: generate.generate_swarm(0.5, SAMPLE_RATE, seed=s), id="swarm"
    ),
    pytest.param(
        lambda s: generate.generate_graincloud(0.5, SAMPLE_RATE, seed=s),
        id="graincloud",
    ),
    pytest.param(
        lambda s: generate.generate_crackle(0.5, SAMPLE_RATE, seed=s), id="crackle"
    ),
    pytest.param(
        lambda s: generate.generate_pluck(0.5, SAMPLE_RATE, seed=s), id="pluck"
    ),
    pytest.param(
        lambda s: generate.generate_harmonics(0.5, SAMPLE_RATE, drift=0.5, seed=s),
        id="harmonics-drift",
    ),
]


def spectrum(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies in Hz, magnitude) of the one-sided FFT."""
    mag = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return freqs, mag


@pytest.mark.parametrize("gen", GENERATORS)
@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_generator_output_shape_and_range(gen: Generator, duration: float) -> None:
    samples = gen(duration, SAMPLE_RATE)
    assert len(samples) == int(duration * SAMPLE_RATE)
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) <= 1.0


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


def band_power_ratio(samples: np.ndarray, sample_rate: int) -> float:
    """Return mean per-bin power in 20–200 Hz over mean power in 2–20 kHz."""
    freqs, mag = spectrum(samples, sample_rate)
    power = mag**2
    low = power[(freqs >= 20) & (freqs < 200)].mean()
    high = power[(freqs >= 2000) & (freqs < 20000)].mean()
    return float(low / high)


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


def test_write_wav_round_trip(tmp_path) -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE)
    path = tmp_path / "out.wav"
    generate.write_wav(str(path), samples, SAMPLE_RATE)
    with wave.open(str(path)) as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == len(samples)


@pytest.mark.parametrize("gen", STEREO_GENERATORS)
def test_stereo_output_shape(gen: Generator) -> None:
    samples = gen(0.5, SAMPLE_RATE)
    assert samples.shape == (int(0.5 * SAMPLE_RATE), 2)
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) <= 1.0


@pytest.mark.parametrize("factory", SEEDED_TEXTURES)
def test_seeded_texture_is_deterministic(factory) -> None:
    assert np.array_equal(factory(7), factory(7))
    assert not np.array_equal(factory(7), factory(8))


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


def test_write_wav_stereo_round_trip(tmp_path) -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE, stereo=True)
    path = tmp_path / "stereo.wav"
    generate.write_wav(str(path), samples, SAMPLE_RATE)
    with wave.open(str(path)) as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == len(samples)
        frames = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    interleaved = frames.reshape(-1, 2)
    assert not np.array_equal(interleaved[:, 0], interleaved[:, 1])
