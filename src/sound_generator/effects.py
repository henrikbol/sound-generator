"""Post-processing: distortion chain, resonant filter, and ADSR envelope."""

import math

import numpy as np

from sound_generator.dsp import FILTER_TYPES


def _biquad(
    filter_type: str, freq: float, q: float, sample_rate: int
) -> tuple[list[float], list[float]]:
    """Compute normalised RBJ biquad coefficients for one filter type."""
    omega = 2.0 * math.pi * freq / sample_rate
    alpha = math.sin(omega) / (2.0 * q)
    cos_w = math.cos(omega)
    if filter_type == "lowpass":
        b = [(1.0 - cos_w) / 2.0, 1.0 - cos_w, (1.0 - cos_w) / 2.0]
    elif filter_type == "highpass":
        b = [(1.0 + cos_w) / 2.0, -(1.0 + cos_w), (1.0 + cos_w) / 2.0]
    else:  # bandpass (constant 0 dB peak gain)
        b = [alpha, 0.0, -alpha]
    a0 = 1.0 + alpha
    return [c / a0 for c in b], [1.0, -2.0 * cos_w / a0, (1.0 - alpha) / a0]


def _apply_filter(
    out: np.ndarray,
    sample_rate: int,
    filter_type: str,
    cutoff: float,
    resonance: float,
    cutoff_end: float,
) -> np.ndarray:
    """Run the resonant biquad over the buffer, sweeping cutoff if requested."""
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"Unknown filter type: {filter_type!r}")
    # Lazy import keeps scipy off the package's default import path.
    from scipy.signal import lfilter  # type: ignore[import-untyped]

    q = float(np.clip(resonance, 0.5, 20.0))
    f_max = 0.45 * sample_rate
    f0 = float(np.clip(cutoff, 10.0, f_max))
    f1 = float(np.clip(cutoff_end, 10.0, f_max)) if cutoff_end > 0 else f0
    mono = out.ndim == 1
    work = out[:, np.newaxis] if mono else out
    zi = np.zeros((2, work.shape[1]))

    if f1 == f0:
        b, a = _biquad(filter_type, f0, q, sample_rate)
        work, _ = lfilter(b, a, work, axis=0, zi=zi)
    else:
        block = 128
        n = work.shape[0]
        filtered = np.empty_like(work)
        for start in range(0, n, block):
            end = min(start + block, n)
            # Exponential (log-domain) cutoff interpolation at block centre.
            u = (start + end) / 2.0 / n
            b, a = _biquad(filter_type, f0 * (f1 / f0) ** u, q, sample_rate)
            filtered[start:end], zi = lfilter(b, a, work[start:end], axis=0, zi=zi)
            # Flush denormals: long silent stretches (padded one-shots) would
            # otherwise decay the state into denormal range and stall lfilter.
            zi[np.abs(zi) < 1e-15] = 0.0
        work = filtered

    result = work[:, 0] if mono else work
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak
    return result


def apply_effects(
    samples: np.ndarray,
    sample_rate: int,
    drive: float = 0.0,
    fold: float = 0.0,
    crush_bits: int = 16,
    crush_rate: float = 0.0,
    filter_type: str = "off",
    cutoff: float = 1000.0,
    resonance: float = 0.707,
    cutoff_end: float = 0.0,
) -> np.ndarray:
    """Apply the effects chain: drive → wavefold → rate crush → bit crush → filter.

    All defaults are exact no-ops. Drive uses a normalised tanh whose gain
    goes to identity as the amount approaches 0, so the knob has no jump.
    The wavefolder pre-gains by ``1 + fold`` and reflects through a
    period-4 triangle that is exact identity on [-1, 1]. Rate crush and
    bit crush commute, so their order is presentational. The resonant
    filter runs last so it can tame and animate the wideband harmonics the
    distortions add; when ``filter_type`` is "off" no sample is touched.

    Args:
        samples: Input audio, shape (n,) or (n, 2), values in [-1, 1].
        sample_rate: Sample rate in Hz.
        drive: Saturation amount 0–1 (tanh gain = 10 * drive).
        fold: Wavefolder amount 0–4.
        crush_bits: Bit depth 1–16; 16 = off.
        crush_rate: Sample-hold rate in Hz; 0 or >= sample_rate = off.
        filter_type: One of FILTER_TYPES; "off" bypasses the filter.
        cutoff: Filter cutoff in Hz (sweep start), clamped to [10, 0.45 * sr].
        resonance: Filter Q, clamped to [0.5, 20].
        cutoff_end: Sweep target in Hz; 0 = static filter.

    Returns:
        Float32 array with the same shape as the input.
    """
    out = samples.astype(np.float64)
    gain = drive * 10.0
    if gain > 1e-6:
        out = np.tanh(gain * out) / np.tanh(gain)
    if fold > 0:
        gained = out * (1.0 + fold)
        out = 4.0 * np.abs(np.mod((gained - 1.0) / 4.0, 1.0) - 0.5) - 1.0
    if 0 < crush_rate < sample_rate:
        held = np.floor(np.arange(out.shape[0]) * crush_rate / sample_rate)
        idx = (held * (sample_rate / crush_rate)).astype(np.int64)
        out = out[np.minimum(idx, out.shape[0] - 1)]
    if crush_bits < 16:
        levels = float(2 ** (max(crush_bits, 1) - 1))
        out = np.round(out * levels) / levels
    if filter_type != "off":
        out = _apply_filter(
            out, sample_rate, filter_type, cutoff, resonance, cutoff_end
        )
    return out.astype(np.float32)


def apply_adsr(
    samples: np.ndarray,
    sample_rate: int,
    attack: float = 0.0,
    decay: float = 0.0,
    sustain: float = 1.0,
    release: float = 0.0,
) -> np.ndarray:
    """Apply an attack/decay/sustain/release amplitude envelope.

    Linear ramps: 0→1 over the attack, 1→sustain over the decay, hold at
    the sustain level, then sustain→0 over the release. The defaults form
    an identity envelope. If attack + decay + release exceed the clip
    length, the three segments are scaled proportionally to fit.

    Args:
        samples: Input audio samples.
        sample_rate: Sample rate in Hz.
        attack: Attack time in seconds.
        decay: Decay time in seconds.
        sustain: Sustain level (0–1).
        release: Release time in seconds.

    Returns:
        Float32 array of enveloped samples.
    """
    num_samples = len(samples)
    attack_n = int(attack * sample_rate)
    decay_n = int(decay * sample_rate)
    release_n = int(release * sample_rate)

    total = attack_n + decay_n + release_n
    if total > num_samples and total > 0:
        scale = num_samples / total
        attack_n = int(attack_n * scale)
        decay_n = int(decay_n * scale)
        release_n = num_samples - attack_n - decay_n
    sustain_n = num_samples - attack_n - decay_n - release_n

    envelope = np.concatenate(
        [
            np.linspace(0.0, 1.0, attack_n, endpoint=False),
            np.linspace(1.0, sustain, decay_n, endpoint=False),
            np.full(sustain_n, sustain),
            np.linspace(sustain, 0.0, release_n),
        ]
    )
    if samples.ndim == 2:
        envelope = envelope[:, np.newaxis]
    return (samples * envelope).astype(np.float32)
