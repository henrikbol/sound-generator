"""Tests for the databend binary-file sonifier: audify, scale, and granular."""

import subprocess
import sys

import numpy as np
import pytest

from sound_generator import databend

PAYLOAD = bytes(range(256)) * 8  # 2048 deterministic bytes


def test_audify_contract() -> None:
    samples = databend.mode_audify(PAYLOAD, 44100)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert np.max(np.abs(samples)) <= 1.0


@pytest.mark.parametrize(
    ("n_bytes", "expected_samples"),
    [
        pytest.param(16, 4, id="exact-boundary"),
        pytest.param(5, 2, id="zero-padded"),
    ],
)
def test_audify_four_bytes_per_sample(n_bytes: int, expected_samples: int) -> None:
    samples = databend.mode_audify(PAYLOAD[:n_bytes], 44100)
    assert len(samples) == expected_samples


def test_audify_cleans_non_finite() -> None:
    payload = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32).tobytes()
    samples = databend.mode_audify(payload, 44100)
    assert np.all(np.isfinite(samples))
    assert samples[0] == 0.0


def test_scale_contract() -> None:
    samples = databend.mode_scale(PAYLOAD[:32], "pentatonic", 240.0, 8)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert np.max(np.abs(samples)) <= 1.0


def test_scale_note_count() -> None:
    samples = databend.mode_scale(PAYLOAD[:32], "pentatonic", 240.0, 8)
    note_samples = int(44100 * 60.0 / 240.0 / 8)  # 0.03125 s notes -> 1378
    assert len(samples) == 32 * note_samples


@pytest.mark.parametrize("scale_name", sorted(databend.SCALES))
def test_scale_all_scales(scale_name: str) -> None:
    samples = databend.mode_scale(PAYLOAD[:4], scale_name, 240.0, 8)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert len(samples) > 0
    assert np.max(np.abs(samples)) <= 1.0


def test_granular_contract() -> None:
    samples = databend.mode_granular(PAYLOAD, 20.0, 2.0, seed=7)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert np.max(np.abs(samples)) <= 0.8 + 1e-6


def test_granular_seed_determinism() -> None:
    first = databend.mode_granular(PAYLOAD, 20.0, 2.0, seed=7)
    second = databend.mode_granular(PAYLOAD, 20.0, 2.0, seed=7)
    reseeded = databend.mode_granular(PAYLOAD, 20.0, 2.0, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, reseeded)
    # CLI parity: the default seed is the historical fixed 42.
    default = databend.mode_granular(PAYLOAD, 20.0, 2.0)
    assert np.array_equal(default, databend.mode_granular(PAYLOAD, 20.0, 2.0, seed=42))


def test_module_import_skips_scipy() -> None:
    code = (
        "import sys, sound_generator.databend; "
        "raise SystemExit(1 if 'scipy' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0
