"""Command-line interface: parse arguments, synthesize, and write a WAV."""

import argparse

from sound_generator.dsp import GRAIN_SOURCES, NOISE_COLORS, WAVEFORMS
from sound_generator.effects import apply_adsr, apply_effects
from sound_generator.textures import (
    generate_crackle,
    generate_graincloud,
    generate_pluck,
    generate_swarm,
)
from sound_generator.tones import (
    generate_bytebeat,
    generate_fm,
    generate_harmonics,
    generate_random_audio,
)
from sound_generator.wav import write_wav


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
