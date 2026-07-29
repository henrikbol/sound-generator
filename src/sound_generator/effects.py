"""Post-processing: distortion chain and ADSR envelope."""

import numpy as np


def apply_effects(
    samples: np.ndarray,
    sample_rate: int,
    drive: float = 0.0,
    fold: float = 0.0,
    crush_bits: int = 16,
    crush_rate: float = 0.0,
) -> np.ndarray:
    """Apply the distortion chain: drive → wavefold → rate crush → bit crush.

    All defaults are exact no-ops. Drive uses a normalised tanh whose gain
    goes to identity as the amount approaches 0, so the knob has no jump.
    The wavefolder pre-gains by ``1 + fold`` and reflects through a
    period-4 triangle that is exact identity on [-1, 1]. Rate crush and
    bit crush commute, so their order is presentational.

    Args:
        samples: Input audio, shape (n,) or (n, 2), values in [-1, 1].
        sample_rate: Sample rate in Hz.
        drive: Saturation amount 0–1 (tanh gain = 10 * drive).
        fold: Wavefolder amount 0–4.
        crush_bits: Bit depth 1–16; 16 = off.
        crush_rate: Sample-hold rate in Hz; 0 or >= sample_rate = off.

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
