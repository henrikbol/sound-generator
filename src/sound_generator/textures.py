"""Texture generators: drone swarm, grain cloud, crackle, and plucks."""

import numpy as np

from sound_generator.dsp import (
    GRAIN_SOURCES,
    _add_event,
    _new_buffer,
    _normalize,
    _pan_gains,
    _waveform,
)


def generate_swarm(
    duration_seconds: float,
    sample_rate: int = 44100,
    *,
    freq: float = 110.0,
    voices: int = 7,
    wave_shape: str = "saw",
    detune: float = 25.0,
    drift: float = 0.2,
    amplitude: float = 0.8,
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate a drone swarm: detuned oscillator voices with slow pitch drift.

    Voices are spread evenly across ``detune`` cents (plus a small random
    jitter) and, in stereo, panned across the field. ``drift`` adds a slow
    random pitch wander of up to ±30 cents per voice, which turns the
    static beating into an evolving texture.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        freq: Base frequency in Hz.
        voices: Number of oscillator voices (1–16 sensible).
        wave_shape: One of WAVEFORMS.
        detune: Total detune spread in cents.
        drift: Pitch-drift depth 0–1 (0 = static tuning).
        amplitude: Peak amplitude (0–1).
        seed: Optional RNG seed for reproducible output.
        stereo: Pan voices across the stereo field.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    num_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    if voices > 1:
        offsets = np.linspace(-detune / 2.0, detune / 2.0, voices)
        pans = np.linspace(-0.8, 0.8, voices)
    else:
        offsets = np.zeros(1)
        pans = np.zeros(1)
    offsets = offsets + rng.uniform(-1.0, 1.0, voices)  # break exact symmetry

    buffer = _new_buffer(num_samples, stereo)
    sample_t = np.arange(num_samples) / sample_rate
    num_ctrl = max(2, int(np.ceil(duration_seconds / 0.5)) + 1)
    ctrl_t = np.linspace(0.0, duration_seconds, num_ctrl)

    for offset, pan in zip(offsets, pans, strict=True):
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        ctrl_cents = rng.uniform(-1.0, 1.0, num_ctrl) * drift * 30.0
        if drift > 0:
            cents = offset + np.interp(sample_t, ctrl_t, ctrl_cents)
            inst_freq = freq * 2.0 ** (cents / 1200.0)
            phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate + phase0
        else:
            voice_freq = freq * 2.0 ** (offset / 1200.0)
            phase = 2.0 * np.pi * voice_freq * sample_t + phase0
        voice = _waveform(phase, wave_shape)
        if stereo:
            left, right = _pan_gains(pan)
            buffer[:, 0] += left * voice
            buffer[:, 1] += right * voice
        else:
            buffer += voice
    return _normalize(buffer, amplitude)


def generate_graincloud(
    duration_seconds: float,
    sample_rate: int = 44100,
    *,
    density: float = 40.0,
    grain_ms: float = 60.0,
    pitch: float = 440.0,
    spread: float = 12.0,
    source: str = "sine",
    amplitude: float = 0.8,
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate a granular cloud of Hanning-windowed micro-grains.

    Grains land at random onsets with random pitch offsets within
    ``±spread`` semitones of ``pitch``; in stereo each grain also gets a
    random pan. ``source`` selects sine grains (shimmering, pitched) or
    noise grains (airy, unpitched — ``pitch``/``spread`` are ignored).

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        density: Average grains per second.
        grain_ms: Grain length in milliseconds.
        pitch: Centre frequency in Hz for sine grains.
        spread: Random pitch range in semitones (±).
        source: One of GRAIN_SOURCES.
        amplitude: Peak amplitude (0–1).
        seed: Optional RNG seed for reproducible output.
        stereo: Randomly pan each grain.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    if source not in GRAIN_SOURCES:
        raise ValueError(f"Unknown grain source: {source!r}")
    num_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    grain_n = max(8, int(sample_rate * grain_ms / 1000.0))
    window = np.hanning(grain_n)
    grain_t = np.arange(grain_n) / sample_rate
    num_grains = max(1, round(density * duration_seconds))

    buffer = _new_buffer(num_samples, stereo)
    for _ in range(num_grains):
        # Fixed draw order (pan included in mono) keeps a pinned seed's
        # texture identical when toggling stereo.
        start = int(rng.integers(0, num_samples))
        semis = rng.uniform(-spread, spread)
        grain_phase = rng.uniform(0.0, 2.0 * np.pi)
        gain = rng.uniform(0.2, 1.0)
        pan = rng.uniform(-1.0, 1.0)
        if source == "noise":
            body = rng.uniform(-1.0, 1.0, grain_n)
        else:
            grain_freq = pitch * 2.0 ** (semis / 12.0)
            body = np.sin(2.0 * np.pi * grain_freq * grain_t + grain_phase)
        _add_event(buffer, start, gain * window * body, pan if stereo else 0.0)
    return _normalize(buffer, amplitude)


def generate_crackle(
    duration_seconds: float,
    sample_rate: int = 44100,
    *,
    density: float = 30.0,
    tone: float = 2500.0,
    decay_ms: float = 6.0,
    amplitude: float = 0.8,
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate crackle/dust: sparse impulses with short resonant tails.

    Each event is an exponentially decaying sine burst at a random
    frequency within ±1 octave of ``tone``. Squared-uniform amplitudes
    give many quiet ticks and a few loud pops — vinyl crackle at low
    density, Geiger-counter dust at high density.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        density: Average events per second.
        tone: Centre resonance frequency in Hz.
        decay_ms: Nominal tail time-constant in milliseconds.
        amplitude: Peak amplitude (0–1).
        seed: Optional RNG seed for reproducible output.
        stereo: Randomly pan each event.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    num_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    num_events = max(1, round(density * duration_seconds))

    buffer = _new_buffer(num_samples, stereo)
    for _ in range(num_events):
        start = int(rng.integers(0, num_samples))
        tau = (decay_ms / 1000.0) * rng.uniform(0.5, 2.0)
        event_freq = min(tone * 2.0 ** rng.uniform(-1.0, 1.0), 0.45 * sample_rate)
        event_phase = rng.uniform(0.0, 2.0 * np.pi)
        gain = rng.uniform(0.05, 1.0) ** 2 * rng.choice((-1.0, 1.0))
        pan = rng.uniform(-1.0, 1.0)
        tail_n = max(4, int(np.ceil(8.0 * tau * sample_rate)))
        tail_t = np.arange(tail_n) / sample_rate
        body = np.exp(-tail_t / tau) * np.sin(
            2.0 * np.pi * event_freq * tail_t + event_phase
        )
        _add_event(buffer, start, gain * body, pan if stereo else 0.0)
    return _normalize(buffer, amplitude)


def generate_pluck(
    duration_seconds: float,
    sample_rate: int = 44100,
    *,
    freq: float = 220.0,
    decay: float = 0.6,
    interval: float = 0.5,
    amplitude: float = 0.8,
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate Karplus-Strong plucked-string textures.

    Two persistent delay lines are re-excited alternately every
    ``interval`` seconds (``interval <= 0`` gives a single pluck), so
    overlapping tails ring naturally. In stereo the lines sit at ±0.5
    pan; in mono they are summed, so a pinned seed keeps the same
    performance either way. The loop filter adds half a sample of delay,
    so the effective pitch is ``sample_rate / (period + 0.5)`` with
    ``period = round(sample_rate / freq - 0.5)``.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        freq: Target string frequency in Hz.
        decay: Ring length 0–1, mapped geometrically to the feedback gain.
        interval: Seconds between plucks (0 = single pluck).
        amplitude: Peak amplitude (0–1).
        seed: Optional RNG seed for reproducible excitation noise.
        stereo: Pan the two strings left/right.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    num_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    # Compensate the loop filter's half-sample delay when picking the period.
    period = max(2, round(sample_rate / freq - 0.5))
    clamped = min(max(decay, 0.0), 1.0)
    # Geometric interpolation of (1 - feedback): 0.98 at decay=0, 0.9999 at 1.
    feedback = 1.0 - 0.02 * (0.0001 / 0.02) ** clamped

    if interval <= 0:
        onsets = [0]
    else:
        onsets = list(range(0, num_samples, max(1, int(interval * sample_rate))))

    num_blocks = max(1, int(np.ceil(num_samples / period)))
    excitations: dict[int, list[int]] = {}
    for i, onset in enumerate(onsets):
        block = min(onset // period, num_blocks - 1)
        excitations.setdefault(block, []).append(i % 2)

    lines = [np.zeros(period), np.zeros(period)]
    out = np.zeros((num_blocks * period, 2))
    for block in range(num_blocks):
        for line_index in excitations.get(block, ()):
            burst = rng.uniform(-1.0, 1.0, period)
            # Remove DC: the averaging loop filter has unity gain at 0 Hz,
            # so any excitation offset would never ring out.
            lines[line_index] = lines[line_index] + burst - burst.mean()
        start = block * period
        out[start : start + period, 0] = lines[0]
        out[start : start + period, 1] = lines[1]
        lines[0] = feedback * 0.5 * (lines[0] + np.roll(lines[0], 1))
        lines[1] = feedback * 0.5 * (lines[1] + np.roll(lines[1], 1))
    out = out[:num_samples]

    if stereo:
        buffer = _new_buffer(num_samples, stereo=True)
        for line_index, pan in ((0, -0.5), (1, 0.5)):
            left, right = _pan_gains(pan)
            buffer[:, 0] += left * out[:, line_index]
            buffer[:, 1] += right * out[:, line_index]
    else:
        buffer = out[:, 0] + out[:, 1]
    return _normalize(buffer, amplitude)
