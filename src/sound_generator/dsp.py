"""Shared DSP helpers and constants for the synthesis modules."""

import numpy as np

NoiseColor = str  # "white" | "pink" | "brown"

NOISE_COLORS = ("white", "pink", "brown")
WAVEFORMS = ("saw", "sine", "square", "triangle")
GRAIN_SOURCES = ("sine", "noise")
FILTER_TYPES = ("off", "lowpass", "bandpass", "highpass")
DRUM_TYPES = ("kick", "snare", "hihat", "sub")
RISER_DIRECTIONS = ("up", "down")

# Per material: (freq_ratio, gain, decay_scale) per mode. Ratios are classic
# modal series — free bar (transverse), Risset church bell, singing bowl,
# woodblock. decay_scale shortens higher modes; MODAL_DECAY_MULT scales the
# whole material's ring time.
MODAL_MATERIALS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "bar": (
        (1.0, 1.0, 1.0),
        (2.756, 0.7, 0.55),
        (5.404, 0.45, 0.35),
        (8.933, 0.3, 0.25),
        (13.345, 0.18, 0.18),
        (18.645, 0.1, 0.12),
    ),
    "bell": (
        (0.56, 0.8, 1.0),
        (0.92, 1.0, 0.8),
        (1.19, 0.75, 0.65),
        (1.71, 0.6, 0.5),
        (2.0, 0.55, 0.4),
        (2.74, 0.4, 0.32),
        (3.0, 0.35, 0.28),
        (3.76, 0.25, 0.22),
        (4.07, 0.2, 0.18),
    ),
    "bowl": (
        (1.0, 1.0, 1.0),
        (2.77, 0.55, 0.7),
        (5.18, 0.3, 0.5),
        (8.16, 0.18, 0.35),
        (11.66, 0.1, 0.25),
    ),
    "wood": (
        (1.0, 1.0, 1.0),
        (2.55, 0.5, 0.5),
        (4.62, 0.3, 0.3),
        (6.98, 0.18, 0.2),
    ),
}

MODAL_DECAY_MULT = {"bar": 0.8, "bell": 1.2, "bowl": 1.8, "wood": 0.15}


def _pan_gains(pan: float) -> tuple[float, float]:
    """Compute equal-power left/right gains for a pan position in [-1, 1]."""
    angle = (pan + 1.0) * np.pi / 4.0
    return float(np.cos(angle)), float(np.sin(angle))


def _new_buffer(num_samples: int, stereo: bool) -> np.ndarray:
    """Allocate a zeroed mono (n,) or stereo (n, 2) float buffer."""
    return np.zeros((num_samples, 2) if stereo else num_samples)


def _add_event(
    buffer: np.ndarray, start: int, chunk: np.ndarray, pan: float = 0.0
) -> None:
    """Mix a mono chunk into a mono or stereo buffer with equal-power panning.

    Chunks that run past the end of the buffer are truncated.

    Args:
        buffer: Target buffer, shape (n,) or (n, 2). Modified in place.
        start: Sample offset at which the chunk begins.
        chunk: Mono samples to add.
        pan: Stereo position in [-1, 1]; ignored for mono buffers.
    """
    if start < 0:
        chunk = chunk[-start:]
        start = 0
    end = min(start + len(chunk), buffer.shape[0])
    if end <= start:
        return
    part = chunk[: end - start]
    if buffer.ndim == 1:
        buffer[start:end] += part
    else:
        left, right = _pan_gains(pan)
        buffer[start:end, 0] += left * part
        buffer[start:end, 1] += right * part


def _normalize(signal: np.ndarray, amplitude: float) -> np.ndarray:
    """Scale a signal so its peak equals ``amplitude`` and cast to float32."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * (amplitude / peak)
    return signal.astype(np.float32)


def _waveform(phase: np.ndarray, wave_name: str) -> np.ndarray:
    """Evaluate a basic waveform at the given phase (radians).

    Naive (non-band-limited) shapes — aliasing above Nyquist is accepted
    as part of the texture character.

    Args:
        phase: Instantaneous phase in radians.
        wave_name: One of WAVEFORMS.

    Returns:
        Waveform samples in [-1, 1].
    """
    if wave_name not in WAVEFORMS:
        raise ValueError(f"Unknown waveform: {wave_name!r}")
    if wave_name == "sine":
        return np.sin(phase)
    cycles = (phase / (2.0 * np.pi)) % 1.0
    if wave_name == "saw":
        return 2.0 * cycles - 1.0
    if wave_name == "square":
        return np.where(cycles < 0.5, 1.0, -1.0)
    return 4.0 * np.abs(cycles - 0.5) - 1.0  # triangle
