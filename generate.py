"""Generate sound files: noise, bytebeat, FM, harmonics, and texture generators."""

import argparse
import wave

import numpy as np

NoiseColor = str  # "white" | "pink" | "brown"

NOISE_COLORS = ("white", "pink", "brown")
WAVEFORMS = ("saw", "sine", "square", "triangle")
GRAIN_SOURCES = ("sine", "noise")


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


def write_wav(filename: str, samples: np.ndarray, sample_rate: int = 44100) -> None:
    """Write a float array in [-1, 1] to a 16-bit WAV file.

    Channel count is inferred from the array shape: (n,) writes mono,
    (n, 2) writes interleaved stereo.

    Args:
        filename: Output file path.
        samples: Float audio samples; values outside [-1, 1] are clipped.
        sample_rate: Sample rate in Hz.
    """
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    channels = 2 if int16.ndim == 2 else 1
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16.tobytes())


def main() -> None:
    """Parse arguments, generate the requested audio, and write the WAV file."""
    parser = argparse.ArgumentParser(description="Generate a sound file.")
    parser.add_argument("duration", type=float, help="Length of the audio in seconds")
    parser.add_argument(
        "--output", default="output.wav", help="Output filename (default: output.wav)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--random", action="store_true", help="Use random noise (default)"
    )
    mode.add_argument("--bytebeat", action="store_true", help="Use bytebeat algorithm")
    mode.add_argument("--fm", action="store_true", help="Use FM synthesis")
    mode.add_argument("--harmonics", action="store_true", help="Use layered harmonics")
    mode.add_argument("--swarm", action="store_true", help="Use detuned drone swarm")
    mode.add_argument("--graincloud", action="store_true", help="Use granular cloud")
    mode.add_argument(
        "--crackle", action="store_true", help="Use crackle/dust impulses"
    )
    mode.add_argument("--pluck", action="store_true", help="Use Karplus-Strong plucks")

    parser.add_argument("--stereo", action="store_true", help="Render stereo output")

    # Noise parameters
    parser.add_argument(
        "--noise-color",
        choices=NOISE_COLORS,
        default="white",
        help="Noise color for random mode (default: white)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducible noise"
    )

    # Bytebeat parameters
    parser.add_argument(
        "--bb-a",
        type=int,
        default=8,
        metavar="A",
        help="Bytebeat param a: primary right-shift (default: 8)",
    )
    parser.add_argument(
        "--bb-b",
        type=int,
        default=5,
        metavar="B",
        help="Bytebeat param b: integer divisor (default: 5)",
    )
    parser.add_argument(
        "--bb-c",
        type=int,
        default=3,
        metavar="C",
        help="Bytebeat param c: secondary right-shift (default: 3)",
    )
    parser.add_argument(
        "--bb-d",
        type=int,
        default=128,
        metavar="D",
        help="Bytebeat param d: AND/XOR mask 0-255 (default: 128)",
    )

    # FM parameters
    parser.add_argument(
        "--fm-freq",
        type=float,
        default=220.0,
        metavar="HZ",
        help="FM carrier frequency in Hz (default: 220)",
    )
    parser.add_argument(
        "--fm-ratio",
        type=float,
        default=2.0,
        metavar="R",
        help="FM modulator/carrier ratio (default: 2.0)",
    )
    parser.add_argument(
        "--fm-index",
        type=float,
        default=5.0,
        metavar="I",
        help="FM modulation index/depth (default: 5.0)",
    )

    # Harmonics parameters
    parser.add_argument(
        "--harm-freq",
        type=float,
        default=110.0,
        metavar="HZ",
        help="Harmonics base frequency in Hz (default: 110)",
    )
    parser.add_argument(
        "--harm-count",
        type=int,
        default=8,
        metavar="N",
        help="Number of harmonics to stack (default: 8)",
    )
    parser.add_argument(
        "--harm-decay",
        type=float,
        default=1.0,
        metavar="P",
        help="Harmonic amplitude rolloff exponent (default: 1.0)",
    )
    parser.add_argument(
        "--harm-stretch",
        type=float,
        default=0.0,
        metavar="S",
        help="Harmonic inharmonicity/stretch coefficient (default: 0)",
    )
    parser.add_argument(
        "--harm-odd",
        action="store_true",
        help="Use odd partials only (hollow, square-like)",
    )
    parser.add_argument(
        "--harm-drift",
        type=float,
        default=0.0,
        metavar="D",
        help="Per-partial amplitude-LFO depth 0-1 (default: 0)",
    )

    # Swarm parameters
    parser.add_argument(
        "--swarm-freq",
        type=float,
        default=110.0,
        metavar="HZ",
        help="Swarm base frequency in Hz (default: 110)",
    )
    parser.add_argument(
        "--swarm-voices",
        type=int,
        default=7,
        metavar="N",
        help="Number of swarm voices (default: 7)",
    )
    parser.add_argument(
        "--swarm-wave",
        choices=WAVEFORMS,
        default="saw",
        help="Swarm oscillator waveform (default: saw)",
    )
    parser.add_argument(
        "--swarm-detune",
        type=float,
        default=25.0,
        metavar="CENTS",
        help="Swarm detune spread in cents (default: 25)",
    )
    parser.add_argument(
        "--swarm-drift",
        type=float,
        default=0.2,
        metavar="D",
        help="Swarm pitch-drift depth 0-1 (default: 0.2)",
    )

    # Grain cloud parameters
    parser.add_argument(
        "--gc-density",
        type=float,
        default=40.0,
        metavar="N",
        help="Grain cloud grains per second (default: 40)",
    )
    parser.add_argument(
        "--gc-grain-ms",
        type=float,
        default=60.0,
        metavar="MS",
        help="Grain length in milliseconds (default: 60)",
    )
    parser.add_argument(
        "--gc-pitch",
        type=float,
        default=440.0,
        metavar="HZ",
        help="Grain cloud centre pitch in Hz (default: 440)",
    )
    parser.add_argument(
        "--gc-spread",
        type=float,
        default=12.0,
        metavar="SEMI",
        help="Grain pitch spread in semitones (default: 12)",
    )
    parser.add_argument(
        "--gc-source",
        choices=GRAIN_SOURCES,
        default="sine",
        help="Grain source material (default: sine)",
    )

    # Crackle parameters
    parser.add_argument(
        "--ck-density",
        type=float,
        default=30.0,
        metavar="N",
        help="Crackle events per second (default: 30)",
    )
    parser.add_argument(
        "--ck-tone",
        type=float,
        default=2500.0,
        metavar="HZ",
        help="Crackle centre resonance in Hz (default: 2500)",
    )
    parser.add_argument(
        "--ck-decay-ms",
        type=float,
        default=6.0,
        metavar="MS",
        help="Crackle tail time-constant in milliseconds (default: 6)",
    )

    # Pluck parameters
    parser.add_argument(
        "--pk-freq",
        type=float,
        default=220.0,
        metavar="HZ",
        help="Pluck string frequency in Hz (default: 220)",
    )
    parser.add_argument(
        "--pk-decay",
        type=float,
        default=0.6,
        metavar="D",
        help="Pluck ring length 0-1 (default: 0.6)",
    )
    parser.add_argument(
        "--pk-interval",
        type=float,
        default=0.5,
        metavar="S",
        help="Seconds between plucks; 0 = single pluck (default: 0.5)",
    )

    # Effects chain (applies to any mode; defaults are a no-op)
    parser.add_argument(
        "--drive",
        type=float,
        default=0.0,
        metavar="D",
        help="Saturation amount 0-1 (default: 0)",
    )
    parser.add_argument(
        "--fold",
        type=float,
        default=0.0,
        metavar="F",
        help="Wavefolder amount 0-4 (default: 0)",
    )
    parser.add_argument(
        "--crush-bits",
        type=int,
        default=16,
        metavar="B",
        help="Bitcrusher bit depth 1-16; 16 = off (default: 16)",
    )
    parser.add_argument(
        "--crush-rate",
        type=float,
        default=0.0,
        metavar="HZ",
        help="Sample-hold rate in Hz; 0 = off (default: 0)",
    )

    # ADSR envelope (applies to any mode; defaults are a no-op)
    parser.add_argument(
        "--attack",
        type=float,
        default=0.0,
        metavar="S",
        help="Envelope attack time in seconds (default: 0)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.0,
        metavar="S",
        help="Envelope decay time in seconds (default: 0)",
    )
    parser.add_argument(
        "--sustain",
        type=float,
        default=1.0,
        metavar="L",
        help="Envelope sustain level 0-1 (default: 1.0)",
    )
    parser.add_argument(
        "--release",
        type=float,
        default=0.0,
        metavar="S",
        help="Envelope release time in seconds (default: 0)",
    )

    args = parser.parse_args()

    if args.bytebeat:
        print(
            f"Generating {args.duration}s bytebeat (a={args.bb_a} b={args.bb_b} "
            f"c={args.bb_c} d={args.bb_d}) at {args.sample_rate} Hz..."
        )
        samples = generate_bytebeat(
            args.duration,
            args.sample_rate,
            a=args.bb_a,
            b=args.bb_b,
            c=args.bb_c,
            d=args.bb_d,
            stereo=args.stereo,
        )
    elif args.fm:
        print(
            f"Generating {args.duration}s FM (carrier={args.fm_freq} Hz "
            f"ratio={args.fm_ratio} index={args.fm_index}) at {args.sample_rate} Hz..."
        )
        samples = generate_fm(
            args.duration,
            args.sample_rate,
            carrier=args.fm_freq,
            ratio=args.fm_ratio,
            index=args.fm_index,
            stereo=args.stereo,
        )
    elif args.harmonics:
        print(
            f"Generating {args.duration}s harmonics (base={args.harm_freq} Hz "
            f"count={args.harm_count} decay={args.harm_decay}) at {args.sample_rate} Hz..."
        )
        samples = generate_harmonics(
            args.duration,
            args.sample_rate,
            freq=args.harm_freq,
            count=args.harm_count,
            decay=args.harm_decay,
            stretch=args.harm_stretch,
            odd_only=args.harm_odd,
            drift=args.harm_drift,
            seed=args.seed,
            stereo=args.stereo,
        )
    elif args.swarm:
        print(
            f"Generating {args.duration}s swarm (freq={args.swarm_freq} Hz "
            f"voices={args.swarm_voices} wave={args.swarm_wave} "
            f"detune={args.swarm_detune}c drift={args.swarm_drift}) "
            f"at {args.sample_rate} Hz..."
        )
        samples = generate_swarm(
            args.duration,
            args.sample_rate,
            freq=args.swarm_freq,
            voices=args.swarm_voices,
            wave_shape=args.swarm_wave,
            detune=args.swarm_detune,
            drift=args.swarm_drift,
            seed=args.seed,
            stereo=args.stereo,
        )
    elif args.graincloud:
        print(
            f"Generating {args.duration}s grain cloud (density={args.gc_density}/s "
            f"grain={args.gc_grain_ms}ms pitch={args.gc_pitch} Hz "
            f"spread=±{args.gc_spread} source={args.gc_source}) "
            f"at {args.sample_rate} Hz..."
        )
        samples = generate_graincloud(
            args.duration,
            args.sample_rate,
            density=args.gc_density,
            grain_ms=args.gc_grain_ms,
            pitch=args.gc_pitch,
            spread=args.gc_spread,
            source=args.gc_source,
            seed=args.seed,
            stereo=args.stereo,
        )
    elif args.crackle:
        print(
            f"Generating {args.duration}s crackle (density={args.ck_density}/s "
            f"tone={args.ck_tone} Hz decay={args.ck_decay_ms}ms) "
            f"at {args.sample_rate} Hz..."
        )
        samples = generate_crackle(
            args.duration,
            args.sample_rate,
            density=args.ck_density,
            tone=args.ck_tone,
            decay_ms=args.ck_decay_ms,
            seed=args.seed,
            stereo=args.stereo,
        )
    elif args.pluck:
        print(
            f"Generating {args.duration}s plucks (freq={args.pk_freq} Hz "
            f"decay={args.pk_decay} interval={args.pk_interval}s) "
            f"at {args.sample_rate} Hz..."
        )
        samples = generate_pluck(
            args.duration,
            args.sample_rate,
            freq=args.pk_freq,
            decay=args.pk_decay,
            interval=args.pk_interval,
            seed=args.seed,
            stereo=args.stereo,
        )
    else:
        print(
            f"Generating {args.duration}s of {args.noise_color} noise at {args.sample_rate} Hz..."
        )
        samples = generate_random_audio(
            args.duration,
            args.sample_rate,
            color=args.noise_color,
            seed=args.seed,
            stereo=args.stereo,
        )

    samples = apply_effects(
        samples,
        args.sample_rate,
        drive=args.drive,
        fold=args.fold,
        crush_bits=args.crush_bits,
        crush_rate=args.crush_rate,
    )
    samples = apply_adsr(
        samples,
        args.sample_rate,
        attack=args.attack,
        decay=args.decay,
        sustain=args.sustain,
        release=args.release,
    )

    write_wav(args.output, samples, args.sample_rate)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
