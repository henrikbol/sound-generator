"""Sample-contract invariants that span every synthesis mode."""

from collections.abc import Callable

import numpy as np
import pytest

import sound_generator as generate

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


@pytest.mark.parametrize("gen", GENERATORS)
@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_generator_output_shape_and_range(gen: Generator, duration: float) -> None:
    samples = gen(duration, SAMPLE_RATE)
    assert len(samples) == int(duration * SAMPLE_RATE)
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) <= 1.0


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
