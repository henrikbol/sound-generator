"""Percussive generators: drum one-shots and modal struck resonators."""

import numpy as np

from sound_generator.dsp import (
    DRUM_TYPES,
    MODAL_DECAY_MULT,
    MODAL_MATERIALS,
    _new_buffer,
    _normalize,
    _pan_gains,
    _waveform,
)


def _peak_norm(signal: np.ndarray) -> np.ndarray:
    """Scale a signal to unit peak; silence passes through unchanged."""
    peak = np.max(np.abs(signal))
    return signal / peak if peak > 0 else signal


def generate_drum(
    duration_seconds: float,
    sample_rate: int = 44100,
    *,
    drum_type: str = "kick",
    tune: float = 0.0,
    decay: float = 0.5,
    tone: float = 0.5,
    amplitude: float = 0.8,
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate a synthesized drum one-shot: kick, snare, hihat, or sub.

    One hit lands at t=0. A hit shorter than the clip is padded with
    silence; a longer tail is truncated, so the output length always
    matches ``duration_seconds``. Stereo duplicates the mono hit on both
    channels (drums sit centred). Recipes: kick is a pitch-swept sine plus
    a noise click; snare layers two drum modes with bright noise; hihat is
    a six-square inharmonic cluster, high-passed by differentiation
    (deterministic — ``seed`` has no effect); sub is an 808-style long
    sine with tanh harmonics.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        drum_type: One of DRUM_TYPES.
        tune: Pitch offset in semitones.
        decay: Tail length 0–1 (tight to boomy; closed to open hat).
        tone: Brightness 0–1 (click, snap, sizzle, sub drive).
        amplitude: Peak amplitude (0–1).
        seed: Optional RNG seed for the noise components.
        stereo: Duplicate the mono hit on both channels.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    if drum_type not in DRUM_TYPES:
        raise ValueError(f"Unknown drum type: {drum_type!r}")
    num_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    p = 2.0 ** (tune / 12.0)

    tau_main = {
        "kick": 0.08 + 0.35 * decay,
        "snare": 0.06 + 0.25 * decay,
        "hihat": 0.025 + 0.25 * decay,
        "sub": 0.25 + 1.25 * decay,
    }[drum_type]
    # Cap synthesis at the audible tail so a short hat inside a long clip
    # stays cheap; the rest of the buffer is silence.
    hit_n = max(8, min(num_samples, int(8.0 * tau_main * sample_rate)))
    t = np.arange(hit_n) / sample_rate

    if drum_type == "kick":
        sweep = 50.0 * p + 120.0 * p * np.exp(-t / 0.02)
        body = np.sin(2.0 * np.pi * np.cumsum(sweep) / sample_rate)
        hit = body * np.exp(-t / tau_main)
        click_n = min(max(4, int(0.004 * sample_rate)), hit_n)
        burst = _peak_norm(np.diff(rng.uniform(-1.0, 1.0, click_n + 1)))
        click_gain = 0.5 * (0.2 + 0.8 * tone)
        hit[:click_n] += click_gain * burst * np.exp(-t[:click_n] / 0.0015)
        hit = np.tanh(1.2 * hit) / np.tanh(1.2)
    elif drum_type == "snare":
        body = np.sin(2.0 * np.pi * 185.0 * p * t) + 0.6 * np.sin(
            2.0 * np.pi * 330.0 * p * t
        )
        body *= np.exp(-t / (0.04 + 0.08 * decay))
        white = rng.uniform(-1.0, 1.0, hit_n)
        bright = _peak_norm(np.diff(white, prepend=white[0]))
        noise = ((1.0 - tone) * white + tone * bright) * np.exp(-t / tau_main)
        hit = 0.6 * body + (0.5 + 0.5 * tone) * noise
    elif drum_type == "hihat":
        cluster = np.zeros(hit_n)
        for ratio in (2.0, 3.0, 4.16, 5.43, 6.79, 8.21):
            osc_freq = 130.0 * p * ratio
            if osc_freq >= 0.45 * sample_rate:
                continue
            cluster += _waveform(2.0 * np.pi * osc_freq * t, "square")
        once = _peak_norm(np.diff(cluster, prepend=cluster[0]))
        twice = _peak_norm(np.diff(once, prepend=once[0]))
        hit = ((1.0 - tone) * once + tone * twice) * np.exp(-t / tau_main)
    else:  # sub
        sweep = 40.0 * p * (1.0 + 0.5 * np.exp(-t / 0.01))
        body = np.sin(2.0 * np.pi * np.cumsum(sweep) / sample_rate)
        body *= np.exp(-t / tau_main)
        drive_gain = 1.0 + 2.0 * tone
        hit = np.tanh(drive_gain * body) / np.tanh(drive_gain)
        click_n = min(max(4, int(0.002 * sample_rate)), hit_n)
        burst = _peak_norm(np.diff(rng.uniform(-1.0, 1.0, click_n + 1)))
        hit[:click_n] += 0.15 * tone * burst * np.exp(-t[:click_n] / 0.0015)

    mono = np.zeros(num_samples)
    mono[:hit_n] = hit
    mono = _normalize(mono, amplitude)
    return np.column_stack((mono, mono)) if stereo else mono


def generate_modal(
    duration_seconds: float,
    sample_rate: int = 44100,
    *,
    freq: float = 220.0,
    material: str = "bell",
    decay: float = 0.6,
    brightness: float = 0.75,
    interval: float = 0.0,
    amplitude: float = 0.8,
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate struck modal resonances: bars, bells, bowls, and woodblocks.

    A shared noise-burst excitation (one strike, or a strike clock every
    ``interval`` seconds, cf. pluck) drives a bank of two-pole resonators —
    one per mode of the material's table in MODAL_MATERIALS. ``decay``
    maps geometrically to ring time (woodblock tick to minute-long bowl),
    and ``brightness`` weights higher modes by ``brightness**i``. In
    stereo each mode gets a static random pan, spreading the partials
    across the field.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        freq: Fundamental frequency in Hz (ratio-1.0 reference).
        material: Key of MODAL_MATERIALS.
        decay: Ring time 0–1, mapped geometrically.
        brightness: High-mode weighting 0–1.
        interval: Seconds between strikes; <= 0 gives a single strike.
        amplitude: Peak amplitude (0–1).
        seed: Optional RNG seed for strike gains/bursts and mode pans.
        stereo: Pan each mode across the stereo field.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    if material not in MODAL_MATERIALS:
        raise ValueError(f"Unknown modal material: {material!r}")
    # Lazy import keeps scipy off the package's default import path.
    from scipy.signal import lfilter  # type: ignore[import-untyped]

    num_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    table = MODAL_MATERIALS[material]
    # Pans are drawn for the full table before Nyquist skipping so the RNG
    # draw order is independent of freq, sample rate, and channel count.
    pans = rng.uniform(-0.6, 0.6, len(table))

    if interval <= 0:
        onsets = [0]
    else:
        onsets = list(range(0, num_samples, max(1, int(interval * sample_rate))))
    burst_n = max(8, int(0.002 * sample_rate))
    fade = np.linspace(1.0, 0.0, burst_n)
    excitation = np.zeros(num_samples)
    for onset in onsets:
        gain = rng.uniform(0.6, 1.0)
        burst = rng.uniform(-1.0, 1.0, burst_n)
        end = min(onset + burst_n, num_samples)
        excitation[onset:end] += gain * (burst * fade)[: end - onset]

    tau_base = MODAL_DECAY_MULT[material] * 0.12 * 25.0**decay
    buffer = _new_buffer(num_samples, stereo)
    for i, (ratio, gain, decay_scale) in enumerate(table):
        mode_freq = freq * ratio
        if mode_freq >= 0.45 * sample_rate:
            continue
        pole = np.exp(-1.0 / (tau_base * decay_scale * sample_rate))
        omega = 2.0 * np.pi * mode_freq / sample_rate
        # The sin(omega) numerator cancels the resonator's inherent gain so
        # mode balance follows the material table.
        b = [0.0, gain * brightness**i * np.sin(omega)]
        a = [1.0, -2.0 * pole * np.cos(omega), pole * pole]
        voice = lfilter(b, a, excitation)
        if stereo:
            left, right = _pan_gains(float(pans[i]))
            buffer[:, 0] += left * voice
            buffer[:, 1] += right * voice
        else:
            buffer += voice
    return _normalize(buffer, amplitude)
