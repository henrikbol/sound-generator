"""Tone generators: noise, bytebeat, FM synthesis, and harmonic stacks."""

import numpy as np

from sound_generator.dsp import NOISE_COLORS, NoiseColor


def generate_random_audio(
    duration_seconds: float,
    sample_rate: int = 44100,
    color: NoiseColor = "white",
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate noise with a white, pink (1/f), or brown (1/f²) spectrum.

    Pink and brown noise are produced by spectrally shaping white noise:
    the FFT of a white-noise buffer is scaled by 1/f^0.5 (pink) or 1/f
    (brown) and transformed back, then normalised to full scale. Stereo
    output uses independent noise per channel, normalised jointly.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        color: Noise color — one of "white", "pink", "brown".
        seed: Optional RNG seed for reproducible output.
        stereo: Render two decorrelated channels.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    if color not in NOISE_COLORS:
        raise ValueError(f"Unknown noise color: {color!r}")

    num_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    shape = (num_samples, 2) if stereo else (num_samples,)
    white = rng.uniform(-1.0, 1.0, shape)
    if color == "white":
        return white.astype(np.float32)

    exponent = 0.5 if color == "pink" else 1.0
    spectrum = np.fft.rfft(white, axis=0)
    freqs = np.fft.rfftfreq(num_samples)
    scale = np.zeros_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / freqs[nonzero] ** exponent
    if stereo:
        scale = scale[:, np.newaxis]
    shaped = np.fft.irfft(spectrum * scale, n=num_samples, axis=0)

    peak = np.max(np.abs(shaped))
    if peak > 0:
        shaped /= peak
    return shaped.astype(np.float32)


# ---------------------------------------------------------------------------
# Bytebeat
#
# Classic bytebeat operates on an 8-bit counter `t` at 8000 Hz.  We scale
# the sample index so one "bytebeat tick" equals sample_rate / 8000 samples,
# keeping the pitch consistent regardless of the output sample rate.
#
# Parameters (all integers, tweak freely):
#   a  – primary frequency/rhythm divisor  (default 8)
#   b  – secondary modulator               (default 5)
#   c  – bitshift depth / harmonic mix     (default 3)
#   d  – XOR/AND mask for texture          (default 128)
#
# Formula:
#   t & (t >> a) | (t // b) ^ (t >> c) & d
#
# This produces a structured, lo-fi, algorithmically musical output.
# ---------------------------------------------------------------------------


def generate_bytebeat(
    duration_seconds: float,
    sample_rate: int = 44100,
    a: int = 8,
    b: int = 5,
    c: int = 3,
    d: int = 128,
    stereo: bool = False,
) -> np.ndarray:
    """Generate bytebeat audio from integer arithmetic on a tick counter.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        a: Primary right-shift amount.
        b: Integer divisor.
        c: Secondary right-shift amount.
        d: AND/XOR mask (0–255).
        stereo: Duplicate the mono signal on both channels.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    tick_rate = 8000  # classic bytebeat clock
    num_samples = int(duration_seconds * sample_rate)
    t = np.arange(num_samples, dtype=np.int64) * tick_rate // sample_rate
    byte_val = (t & (t >> a) | (t // max(b, 1)) ^ (t >> c) & d) & 0xFF
    # map 0-255 → -1.0..~1.0
    mono = ((byte_val - 128) / 128.0).astype(np.float32)
    return np.column_stack((mono, mono)) if stereo else mono


def generate_fm(
    duration_seconds: float,
    sample_rate: int = 44100,
    carrier: float = 220.0,
    ratio: float = 2.0,
    index: float = 5.0,
    amplitude: float = 0.8,
    stereo: bool = False,
) -> np.ndarray:
    """Generate FM synthesis: a carrier wave phase-modulated by a second oscillator.

    The modulator frequency is ``carrier * ratio``. Integer ratios give
    harmonic, bell-like timbres; non-integer ratios give inharmonic,
    metallic ones. The modulation index controls brightness — 0 is a pure
    sine, higher values add ever more sidebands. Stereo detunes the right
    channel by +6 cents (carrier and modulator together, preserving the
    ratio) for gentle width.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        carrier: Carrier frequency in Hz.
        ratio: Modulator/carrier frequency ratio.
        index: Modulation index (depth).
        amplitude: Peak amplitude (0–1).
        stereo: Render a detuned right channel.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate

    def render_channel(channel_carrier: float) -> np.ndarray:
        modulator = np.sin(2 * np.pi * channel_carrier * ratio * t)
        return amplitude * np.sin(2 * np.pi * channel_carrier * t + index * modulator)

    if not stereo:
        return render_channel(carrier).astype(np.float32)
    detuned = carrier * 2.0 ** (6.0 / 1200.0)
    return np.column_stack((render_channel(carrier), render_channel(detuned))).astype(
        np.float32
    )


def generate_harmonics(
    duration_seconds: float,
    sample_rate: int = 44100,
    freq: float = 110.0,
    count: int = 8,
    decay: float = 1.0,
    amplitude: float = 0.8,
    stretch: float = 0.0,
    odd_only: bool = False,
    drift: float = 0.0,
    seed: int | None = None,
    stereo: bool = False,
) -> np.ndarray:
    """Generate layered harmonics: stacked sines at multiples of a base frequency.

    Partial ``n`` sits at ``freq * n * sqrt(1 + stretch * n**2)`` (piano-style
    inharmonicity when ``stretch > 0``) with amplitude ``1 / n**decay``; the
    stack is normalised by the amplitude sum so the output never clips.
    Partials at or above Nyquist are skipped. Each partial gets a random
    phase; ``drift`` adds a slow per-partial amplitude LFO (modulating only
    downward, so the peak bound holds) for evolving drone textures. Stereo
    uses independent partial phases per channel.

    Args:
        duration_seconds: Length of the audio in seconds.
        sample_rate: Sample rate in Hz.
        freq: Base (fundamental) frequency in Hz.
        count: Number of partials to stack.
        decay: Amplitude rolloff exponent — higher = darker.
        amplitude: Peak amplitude (0–1).
        stretch: Inharmonicity coefficient (0 = perfectly harmonic).
        odd_only: Use the first ``count`` odd partials (hollow, square-like).
        drift: Per-partial amplitude-LFO depth 0–1 (0 = static).
        seed: Optional RNG seed for reproducible phases/drift.
        stereo: Render decorrelated channels.

    Returns:
        Float32 array of samples in [-1, 1], shape (n,) or (n, 2).
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate
    nyquist = sample_rate / 2
    rng = np.random.default_rng(seed)
    channels = 2 if stereo else 1
    numbers = range(1, 2 * count, 2) if odd_only else range(1, count + 1)

    signal = np.zeros((num_samples, channels))
    amp_sum = 0.0
    for n in numbers:
        partial_freq = freq * n * np.sqrt(1.0 + stretch * n * n)
        if partial_freq >= nyquist:
            break
        amp = 1.0 / n**decay
        # Draws happen in a fixed order regardless of channel count so a
        # pinned seed produces the same texture in mono and stereo.
        phases = rng.uniform(0.0, 2.0 * np.pi, 2)
        lfo_rate = rng.uniform(0.05, 0.3)
        lfo_phase = rng.uniform(0.0, 2.0 * np.pi)
        if drift > 0:
            mod = 1.0 - drift * (
                0.5 + 0.5 * np.sin(2.0 * np.pi * lfo_rate * t + lfo_phase)
            )
        else:
            mod = 1.0
        for ch in range(channels):
            signal[:, ch] += (
                amp * mod * np.sin(2.0 * np.pi * partial_freq * t + phases[ch])
            )
        amp_sum += amp

    if amp_sum > 0:
        signal *= amplitude / amp_sum
    out = signal if stereo else signal[:, 0]
    return out.astype(np.float32)
