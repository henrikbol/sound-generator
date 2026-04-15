# Sound Generator

Two approaches to generating sound files from raw data in pure Python — no DAW, no plugins, just code and math.

---

## Approaches

| Approach | Script | Input | Character |
|---|---|---|---|
| [**Algorithmic generation**](#approach-1--algorithmic-generation-generatepy) | `generate.py` | Nothing — sound is synthesised from formulas | Bytebeat patterns, white noise |
| [**Databending**](#approach-2--databending-databendpy) | `databend.py` | Any binary file (image, executable, document…) | Glitch music, soundscapes, melodic textures |

---

## Approach 1 — Algorithmic Generation (`generate.py`)

Sound is generated entirely from mathematical formulas operating on a sample counter. No input file needed.

### Usage

```bash
uv run generate.py <duration> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `duration` | Length of audio in seconds |
| `--output` | Output filename (default: `output.wav`) |
| `--sample-rate` | Sample rate in Hz (default: `44100`) |
| `--random` | Use random white noise (default mode) |
| `--bytebeat` | Use bytebeat algorithm |

### Modes

#### Random (white noise)

Pure random 16-bit samples. All frequencies at equal energy. Useful as a baseline or noise source.

```bash
uv run generate.py 5
uv run generate.py 5 --output noise.wav
```

#### Bytebeat

Bytebeat is a technique where simple integer arithmetic on a sample counter `t` produces structured, lo-fi, algorithmically musical output. Classic bytebeat runs at an 8000 Hz tick rate — the script scales `t` automatically so pitch stays consistent regardless of `--sample-rate`.

```bash
uv run generate.py 5 --bytebeat
uv run generate.py 5 --bytebeat --output my_sound.wav
```

##### Formula

```
(t & (t >> a) | (t // b) ^ (t >> c) & d) & 0xFF
```

where `t` is the 8000 Hz tick counter and `a`, `b`, `c`, `d` are tunable parameters.

##### Bytebeat Parameters

| Flag | Default | Role | Low values | High values |
|---|---|---|---|---|
| `--bb-a` | `8` | Primary right-shift | Dense, buzzy | Slower rhythm, more structure |
| `--bb-b` | `5` | Integer divisor | Fast texture | Slower sub-pattern |
| `--bb-c` | `3` | Secondary right-shift | Harsh harmonics | Smoother mix |
| `--bb-d` | `128` | AND/XOR mask (0–255) | Sparse, glitchy | Full texture |

##### Presets

```bash
# Rhythmic
uv run generate.py 5 --bytebeat --bb-a 6 --bb-b 7 --bb-c 4 --bb-d 255

# Metallic
uv run generate.py 5 --bytebeat --bb-a 10 --bb-b 2 --bb-c 6 --bb-d 64

# Lo-fi melody
uv run generate.py 5 --bytebeat --bb-a 8 --bb-b 11 --bb-c 2 --bb-d 192
```

### Future Algorithms

Planned additions for `generate.py`:

- **FM synthesis** — carrier wave modulated by a second oscillator; ratio and depth control the timbre from bell-like to metallic
- **Layered harmonics** — stack sine waves at integer multiples of a base frequency with decreasing amplitude
- **Pink/brown noise** — shaped random noise (`1/f`, `1/f²`) that sounds more organic than white noise
- **ADSR envelope** — attack/decay/sustain/release shaping applied on top of any generator

---

## Approach 2 — Databending (`databend.py`)

> Convert any binary file into audio — glitch music, soundscapes, and melodic textures from raw data.

### What is databending?

**Databending** is the art of feeding a non-audio file — an image, an executable, a Word document, anything — into an audio engine as if it were sound. The file's binary content becomes the waveform. The results range from harsh electronic noise to structured melodies to dense granular drones, depending on the technique used.

This script implements three distinct databending modes in pure Python, producing standard WAV files compatible with any DAW (Ableton, Logic, Reaper, etc.).

### Background

This technique sits at the intersection of several named practices:

| Term | Meaning |
|---|---|
| **Databending** | Manipulating a file using software designed for a different format. Derived from *circuit bending* — short-circuiting toys and instruments for unpredictable sounds. |
| **Audification** | The purest form: shifting a raw data stream directly into the audible realm by treating it as PCM audio samples. |
| **Data Sonification** | The broader practice of mapping any data to sound parameters — pitch, amplitude, grain density, etc. The sonic equivalent of data visualisation. |
| **Parameter-mapped sonification** | Data values mapped to musical parameters (pitch, amplitude, scale) to produce more structured, musical results. |
| **Granular sonification** | Data values used to control the amplitude of noise grains — producing textural soundscapes reminiscent of granular synthesis. |

#### Artistic tradition

Databending is firmly rooted in **glitch music**. Pioneering practitioners include:

- **stAllio!** — built entire albums from sonified image files, executables, and DLLs. Coined the term *IDM (Interpreted Data Music)* and released the landmark all-databending EP *Dissonance is Bliss* in 1999.
- **Alva Noto** — prominent use of the sonification technique in glitch and minimal electronic music.
- **r2blend** — the 2011 YouTube video *"MS Paint Interpreted as audio data = Awesome music!"* went viral and introduced the technique to a new generation.

### Requirements

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
# Uses uv with inline script dependencies — no manual install needed
uv run databend.py --help

# Or add dependencies explicitly:
uv add numpy scipy
uv run databend.py --help
```

### Modes

#### `audify` — Raw bytes as a PCM waveform

The purest databending technique. Raw bytes are interpreted directly as little-endian float32 PCM samples — the file's binary content *is* the audio waveform. Produces harsh, glitchy, electronic textures. Ideal for drum one-shots, noise sources, and abstract sound design.

**Key insight:** The `--sample-rate` parameter is a powerful creative handle:
- Lower sample rate → sound is slower and lower in pitch (darker, heavier)
- Higher sample rate → sound is faster and higher in pitch (brighter, shorter)

```bash
# Default — full fidelity
uv run databend.py audify myimage.png output.wav

# Half sample rate — one octave lower, twice as long
uv run databend.py audify myfile.exe output.wav --sample-rate 22050

# Limit to first 8 KB — useful for huge files
uv run databend.py audify mybinary output.wav --max-bytes 8192
```

**Best source files:** Uncompressed images (BMP, TIFF) have dense, structured byte patterns and tend to sound rich. Executables (`.exe`, `.so`, `.dylib`) produce complex electronic textures. Avoid compressed files (ZIP, MP3) — the entropy is too uniform.

#### `scale` — Bytes mapped to musical pitches

Each byte value (0–255) is mapped to a pitch in a musical scale across 5 octaves, synthesised as a sine tone with a BPM-quantised duration. Byte value also modulates amplitude — louder notes correspond to higher byte values. Produces actual melodies and harmonic patterns directly from the data.

**Available scales:**

| Scale | Character |
|---|---|
| `pentatonic` | Open, universal, rarely clashes — good default |
| `blues` | Gritty, expressive, works well with glitchy textures |
| `phrygian` | Dark, modal, Middle Eastern flavour |
| `minor` | Melancholic, familiar |
| `major` | Bright, resolving |
| `lydian` | Dreamy, floating |
| `wholetone` | Ambiguous, impressionistic |
| `chromatic` | All 12 pitches — atonal, dense |

```bash
# Phrygian mode at 140 BPM, 16th notes (default)
uv run databend.py scale myfile.docx output.wav --scale phrygian --bpm 140

# Blues at 90 BPM, 32nd notes — very fast melodic runs
uv run databend.py scale myimage.png output.wav --scale blues --bpm 90 --divisions 8

# Slow, quarter-note pentatonic — spacious and meditative
uv run databend.py scale mybinary output.wav --scale pentatonic --bpm 60 --divisions 1

# Limit to 256 bytes — produces a concise 256-note phrase
uv run databend.py scale myfile.png output.wav --max-bytes 256
```

> **Note:** Scale mode default `--max-bytes 512` keeps output to a manageable length. A 10 KB file at 120 BPM, 16th notes would produce ~85 seconds of audio.

#### `granular` — Bytes as noise-grain amplitudes

Bytes are grouped into grains. Each grain's byte values modulate the amplitude of a Hanning-windowed white noise burst, producing dense textural soundscapes. This is conceptually identical to granular synthesis — but with file data as the control signal instead of a human performer.

**Parameters:**
- `--grain-ms` — grain duration. Short (5–20 ms) = granular buzz and crunch. Long (50–200 ms) = smoother evolving washes.
- `--density` — overlap factor. 1.0 = adjacent grains (no overlap). 4.0+ = dense, reverberant wash.

```bash
# Default — 20 ms grains, 2× overlap
uv run databend.py granular myfile.docx output.wav

# Long grains, high density — ambient drone/pad
uv run databend.py granular myfile.png output.wav --grain-ms 80 --density 4

# Short grains, low density — percussive, chattery texture
uv run databend.py granular myfile.exe output.wav --grain-ms 8 --density 1.2
```

### All options

```
usage: databend.py {audify,scale,granular} input output [options]

AUDIFY:
  --sample-rate INT    Sample rate in Hz (default 44100)
  --max-bytes INT      Limit input to first N bytes

SCALE:
  --scale NAME         Musical scale: pentatonic (default), blues, phrygian,
                       minor, major, lydian, wholetone, chromatic
  --bpm FLOAT          Tempo in BPM (default 120)
  --divisions INT      Note grid: 1=quarter, 2=8th, 4=16th (default), 8=32nd
  --max-bytes INT      Limit input bytes (default 512)

GRANULAR:
  --grain-ms FLOAT     Grain duration in milliseconds (default 20)
  --density FLOAT      Overlap factor (default 2.0)
  --max-bytes INT      Limit input to first N bytes
```

### Ableton Live workflow

The output WAVs are standard 16-bit / 44.1 kHz and drop straight into Ableton.

**Suggested routing:**

1. **`audify` → Simpler**
   Drop the WAV into Simpler. Use it as a one-shot sample. Pitch it down with the transpose knob, apply a low-pass filter, add reverb. Works especially well as a percussive hit or a noise sweep.

2. **`scale` → Clip**
   The scale output is already a melodic stem. Drop it into an audio clip, warp it, and layer your own instruments on top. The melody is deterministic — same file + same settings = same melody every time, so it's reproducible.

3. **`granular` → Return track with reverb**
   Long-grain, high-density output works as an ambient pad. Route it to a return channel with long reverb and subtle modulation. Sidechain compress it against your kick for rhythmic pumping.

**Tip:** Run the same file through all three modes and layer the results — the `audify` version provides rhythm and texture, `scale` provides melody, and `granular` provides atmosphere.

### What files sound best?

| File type | Character | Best mode |
|---|---|---|
| `.bmp`, `.tiff` (uncompressed images) | Rich, structured — headers create repeating patterns | `audify`, `granular` |
| `.exe`, `.so`, `.dylib` (executables) | Complex, metallic, electronic | `audify` |
| `.png`, `.jpg` (compressed images) | More uniform entropy, smoother | `granular` |
| `.docx`, `.pdf` (documents) | Mix of structured headers and text — melodic | `scale` |
| `.mp3`, `.zip` (compressed data) | High entropy, close to white noise | `granular` |
| Small files (< 1 KB) | Tight, repeating motifs | `scale` |
| Large files (> 1 MB) | Dense, evolving, long-form | `granular`, `audify` |

### Further reading

- **stAllio!'s Databending Primer** — the definitive guide to sonification technique: `blog.animalswithinanimals.com`
- **CDM article on Binary Synth** — browser-based binary-to-MIDI tool: `cdm.link/transform-any-binary-file-into-sound`
- **Wikipedia: Databending** — history and context
- **Wikipedia: Data Sonification** — the broader scientific and artistic field
- **Audacity Import Raw Data** — `File → Import → Raw Data` lets you audify any file without writing code; a good way to preview what a file sounds like before processing it

