#!/usr/bin/env python3
"""Generate a sound file with random or bytebeat audio data."""

import argparse
import struct
import random
import wave


def generate_random_audio(duration_seconds: float, sample_rate: int = 44100) -> list[int]:
    num_samples = int(duration_seconds * sample_rate)
    return [random.randint(-32768, 32767) for _ in range(num_samples)]


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
) -> list[int]:
    tick_rate = 8000  # classic bytebeat clock
    num_samples = int(duration_seconds * sample_rate)
    samples = []
    for i in range(num_samples):
        t = int(i * tick_rate / sample_rate)
        byte_val = (t & (t >> a) | (t // max(b, 1)) ^ (t >> c) & d) & 0xFF
        # map 0-255 → -32768..32767
        sample = (byte_val - 128) * 256
        samples.append(sample)
    return samples


def write_wav(filename: str, samples: list[int], sample_rate: int = 44100) -> None:
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        data = struct.pack(f"<{len(samples)}h", *samples)
        wav_file.writeframes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sound file.")
    parser.add_argument("duration", type=float, help="Length of the audio in seconds")
    parser.add_argument("--output", default="output.wav", help="Output filename (default: output.wav)")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz (default: 44100)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--random", action="store_true", help="Use random white noise (default)")
    mode.add_argument("--bytebeat", action="store_true", help="Use bytebeat algorithm")

    # Bytebeat parameters
    parser.add_argument("--bb-a", type=int, default=8,   metavar="A", help="Bytebeat param a: primary right-shift (default: 8)")
    parser.add_argument("--bb-b", type=int, default=5,   metavar="B", help="Bytebeat param b: integer divisor (default: 5)")
    parser.add_argument("--bb-c", type=int, default=3,   metavar="C", help="Bytebeat param c: secondary right-shift (default: 3)")
    parser.add_argument("--bb-d", type=int, default=128, metavar="D", help="Bytebeat param d: AND/XOR mask 0-255 (default: 128)")

    args = parser.parse_args()

    if args.bytebeat:
        print(
            f"Generating {args.duration}s bytebeat (a={args.bb_a} b={args.bb_b} "
            f"c={args.bb_c} d={args.bb_d}) at {args.sample_rate} Hz..."
        )
        samples = generate_bytebeat(
            args.duration, args.sample_rate,
            a=args.bb_a, b=args.bb_b, c=args.bb_c, d=args.bb_d,
        )
    else:
        print(f"Generating {args.duration}s of random audio at {args.sample_rate} Hz...")
        samples = generate_random_audio(args.duration, args.sample_rate)

    write_wav(args.output, samples, args.sample_rate)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
