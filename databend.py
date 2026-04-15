"""databend.py — Convert any binary file into audio.

Three modes:
  audify   Raw bytes interpreted directly as a PCM waveform (purest databending).
  scale    Each byte maps to a pitch in a musical scale → synthesised sine tones.
  granular Bytes control amplitude of noise grains → textural soundscapes.

Usage:
  uv run databend.py audify   myfile.exe  output.wav
  uv run databend.py scale    myfile.png  output.wav --scale phrygian --bpm 120
  uv run databend.py granular myfile.docx output.wav --grain-ms 30

Requirements:
  uv add numpy scipy
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# Musical scale definitions (semitone offsets from root, root = MIDI 36 = C2)
# ---------------------------------------------------------------------------
SCALES: dict[str, list[int]] = {
    "chromatic":  list(range(12)),
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "blues":      [0, 3, 5, 6, 7, 10],
    "wholetone":  [0, 2, 4, 6, 8, 10],
}

ROOT_MIDI = 36  # C2 — gives a rich low-mid range


def bytes_to_float32(data: bytes) -> np.ndarray:
    """Interpret raw bytes as little-endian float32 PCM samples.

    Args:
        data: Raw file bytes.

    Returns:
        Normalised float32 array in [-1.0, 1.0].
    """
    # Pad to float32 boundary
    remainder = len(data) % 4
    if remainder:
        data = data + b"\x00" * (4 - remainder)

    samples = np.frombuffer(data, dtype="<f4").copy()

    # Replace NaN/Inf that can come from non-audio binary data
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    return samples


def midi_to_freq(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12)


def sine_tone(freq: float, duration_s: float, amplitude: float = 0.3) -> np.ndarray:
    """Generate a sine wave with a short fade-in/out envelope.

    Args:
        freq: Frequency in Hz.
        duration_s: Duration in seconds.
        amplitude: Peak amplitude (0–1).

    Returns:
        Float32 array of samples.
    """
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)

    # Short fade to remove clicks (5 ms or 10% of note, whichever is smaller)
    fade_n = min(int(SAMPLE_RATE * 0.005), n // 10)
    if fade_n > 0:
        fade = np.linspace(0, 1, fade_n)
        wave[:fade_n] *= fade
        wave[-fade_n:] *= fade[::-1]

    return wave.astype(np.float32)


# ---------------------------------------------------------------------------
# Mode 1: Audify
# ---------------------------------------------------------------------------

def mode_audify(data: bytes, sample_rate: int) -> np.ndarray:
    """Interpret raw bytes directly as a PCM waveform.

    This is the purest databending technique — the file's binary content
    IS the audio waveform. Produces harsh, glitchy, electronic textures.

    Args:
        data: Raw file bytes.
        sample_rate: Target sample rate (affects pitch/speed).

    Returns:
        Normalised float32 audio array.
    """
    print(f"  [audify] {len(data):,} bytes → {len(data) / 4:,.0f} samples "
          f"→ {len(data) / 4 / sample_rate:.2f}s at {sample_rate} Hz")
    return bytes_to_float32(data)


# ---------------------------------------------------------------------------
# Mode 2: Scale mapping
# ---------------------------------------------------------------------------

def mode_scale(
    data: bytes,
    scale_name: str,
    bpm: float,
    note_divisions: int,
) -> np.ndarray:
    """Map each byte to a pitch in a musical scale.

    Each byte value (0–255) selects a note from the chosen scale across
    multiple octaves. Duration is quantised to a musical grid (BPM-based).

    Args:
        data: Raw file bytes.
        scale_name: Key into SCALES dict.
        bpm: Tempo in beats per minute.
        note_divisions: Note value as divisions of a beat (4 = 16th notes).

    Returns:
        Float32 audio array.
    """
    scale = SCALES[scale_name]
    n_scale = len(scale)
    beat_s = 60.0 / bpm
    note_s = beat_s / note_divisions

    # Build full pitch table: scale across 5 octaves = up to 60 distinct pitches
    octaves = 5
    pitch_table: list[float] = []
    for octave in range(octaves):
        for semitone in scale:
            midi = ROOT_MIDI + octave * 12 + semitone
            pitch_table.append(midi_to_freq(midi))

    n_pitches = len(pitch_table)
    print(f"  [scale] {len(data):,} bytes → {len(data)} notes "
          f"at {bpm} BPM ({note_s * 1000:.1f} ms each) using '{scale_name}' scale")

    chunks: list[np.ndarray] = []
    for byte_val in data:
        pitch_idx = int((byte_val / 255) * (n_pitches - 1))
        freq = pitch_table[pitch_idx]
        # Amplitude modulated by byte value too — louder = higher byte
        amp = 0.15 + 0.25 * (byte_val / 255)
        chunks.append(sine_tone(freq, note_s, amplitude=amp))

    return np.concatenate(chunks).astype(np.float32)


# ---------------------------------------------------------------------------
# Mode 3: Granular
# ---------------------------------------------------------------------------

def mode_granular(
    data: bytes,
    grain_ms: float,
    density: float,
) -> np.ndarray:
    """Use bytes as amplitude envelopes over noise grains.

    Bytes are grouped into grains. Each grain's byte values modulate
    the amplitude of bandpass-filtered noise, producing dense textural
    soundscapes reminiscent of granular synthesis.

    Args:
        data: Raw file bytes.
        grain_ms: Grain duration in milliseconds.
        density: Overlap factor — 1.0 = adjacent grains, 2.0 = 50% overlap.

    Returns:
        Float32 audio array.
    """
    grain_n = int(SAMPLE_RATE * grain_ms / 1000)
    hop_n = max(1, int(grain_n / density))

    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0

    # Pad to fill grains
    n_grains = int(np.ceil(len(arr) / grain_n))
    pad = n_grains * grain_n - len(arr)
    arr = np.pad(arr, (0, pad))

    total_samples = (n_grains - 1) * hop_n + grain_n
    output = np.zeros(total_samples, dtype=np.float32)
    envelope = np.hanning(grain_n).astype(np.float32)

    print(f"  [granular] {len(data):,} bytes → {n_grains} grains "
          f"× {grain_ms:.0f} ms, {density:.1f}× overlap → "
          f"{total_samples / SAMPLE_RATE:.2f}s")

    rng = np.random.default_rng(seed=42)

    for i in range(n_grains):
        grain_bytes = arr[i * grain_n: (i + 1) * grain_n]

        # White noise shaped by the byte-amplitude envelope
        noise = rng.standard_normal(grain_n).astype(np.float32)

        # Byte values modulate grain amplitude sample-by-sample
        grain = noise * grain_bytes * envelope

        start = i * hop_n
        output[start: start + grain_n] += grain

    # Normalise
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak
    output *= 0.8  # leave headroom

    return output


# ---------------------------------------------------------------------------
# Write WAV
# ---------------------------------------------------------------------------

def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a float32 array to a 16-bit WAV file.

    Args:
        path: Output file path.
        samples: Float32 audio data normalised to [-1, 1].
        sample_rate: Sample rate in Hz.
    """
    # Clip and convert to int16 for maximum DAW compatibility
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, int16)
    duration_s = len(int16) / sample_rate
    print(f"  ✓ Wrote {path} — {duration_s:.2f}s, {path.stat().st_size / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="databend.py — Convert any binary file into audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run databend.py audify   /usr/bin/python3 output.wav
  uv run databend.py audify   myimage.png output.wav --sample-rate 22050
  uv run databend.py scale    myfile.docx output.wav --scale phrygian --bpm 140
  uv run databend.py scale    myfile.exe  output.wav --scale blues --divisions 8
  uv run databend.py granular myfile.png  output.wav --grain-ms 50 --density 3
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- audify --
    p_a = sub.add_parser("audify", help="Raw bytes as PCM waveform")
    p_a.add_argument("input", type=Path, help="Any binary file")
    p_a.add_argument("output", type=Path, help="Output .wav file")
    p_a.add_argument(
        "--sample-rate", type=int, default=SAMPLE_RATE,
        help=f"Sample rate in Hz (default {SAMPLE_RATE}). "
             "Lower = slower + darker; higher = faster + brighter",
    )
    p_a.add_argument(
        "--max-bytes", type=int, default=None,
        help="Limit input to first N bytes (useful for huge files)",
    )

    # -- scale --
    p_s = sub.add_parser("scale", help="Bytes mapped to musical pitches")
    p_s.add_argument("input", type=Path)
    p_s.add_argument("output", type=Path)
    p_s.add_argument(
        "--scale", choices=list(SCALES), default="pentatonic",
        help="Musical scale (default: pentatonic)",
    )
    p_s.add_argument("--bpm", type=float, default=120.0, help="Tempo (default 120)")
    p_s.add_argument(
        "--divisions", type=int, default=4,
        help="Note grid: 1=quarter, 2=8th, 4=16th, 8=32nd notes (default 4)",
    )
    p_s.add_argument("--max-bytes", type=int, default=512,
                     help="Limit input bytes (default 512 — longer = very long file)")

    # -- granular --
    p_g = sub.add_parser("granular", help="Bytes as noise-grain amplitudes")
    p_g.add_argument("input", type=Path)
    p_g.add_argument("output", type=Path)
    p_g.add_argument(
        "--grain-ms", type=float, default=20.0,
        help="Grain duration in milliseconds (default 20). "
             "Shorter = granular buzz; longer = smoother texture",
    )
    p_g.add_argument(
        "--density", type=float, default=2.0,
        help="Grain overlap factor (default 2.0). Higher = denser wash",
    )
    p_g.add_argument("--max-bytes", type=int, default=None)

    return parser


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Read input
    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    data = args.input.read_bytes()
    max_bytes = getattr(args, "max_bytes", None)
    if max_bytes:
        data = data[:max_bytes]

    print(f"\n{'─' * 50}")
    print(f"  Input : {args.input}  ({len(data):,} bytes)")
    print(f"  Mode  : {args.mode}")
    print(f"{'─' * 50}")

    sample_rate = getattr(args, "sample_rate", SAMPLE_RATE)

    if args.mode == "audify":
        audio = mode_audify(data, sample_rate)

    elif args.mode == "scale":
        audio = mode_scale(data, args.scale, args.bpm, args.divisions)
        sample_rate = SAMPLE_RATE

    elif args.mode == "granular":
        audio = mode_granular(data, args.grain_ms, args.density)
        sample_rate = SAMPLE_RATE

    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

    write_wav(args.output, audio, sample_rate)
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()