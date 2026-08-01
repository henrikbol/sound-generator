"use strict";

/* ========================================================================
 * Sound Generator UI — vanilla JS, no dependencies.
 * Talks to the FastAPI backend: /api/render, /api/clips, /api/reveal.
 * ====================================================================== */

/* ------------------------------ State --------------------------------- */

// Modes whose params take a seed — fm/bytebeat are deterministic and never get one.
// databend always sends one; the server threads it into granular only.
const STOCHASTIC_MODES = ["random", "harmonics", "swarm", "graincloud", "crackle", "pluck", "riser", "drum", "modal", "databend"];

function rollSeed() {
  return 1 + Math.floor(Math.random() * 0x7fffffff);
}

const state = {
  mode: "random",
  duration: 2.0,
  sampleRate: 44100,
  stereo: false,
  adsr: { attack: 0.0, decay: 0.0, sustain: 1.0, release: 0.0 },
  effects: { drive: 0.0, fold: 0.0, crush_bits: 16, crush_rate: 0, filter_type: "off", cutoff: 1000, resonance: 0.707, cutoff_end: 0 },
  params: {
    random: { color: "white" },
    bytebeat: { a: 8, b: 5, c: 3, d: 128 },
    fm: { carrier: 220.0, ratio: 2.0, index: 5.0 },
    harmonics: { freq: 110.0, count: 8, decay: 1.0, stretch: 0.0, odd_only: false, drift: 0.0 },
    swarm: { freq: 110.0, voices: 7, wave: "saw", detune: 25, drift: 0.2 },
    graincloud: { density: 40, grain_ms: 60, pitch: 440.0, spread: 12, source: "sine" },
    crackle: { density: 30, tone: 2500, decay_ms: 6.0 },
    pluck: { freq: 220.0, decay: 0.6, interval: 0.5 },
    riser: { freq: 110.0, octaves: 2.0, voices: 3, detune: 18, noise_mix: 0.5, curve: 2.0, direction: "up" },
    drum: { drum_type: "kick", tune: 0, decay: 0.5, tone: 0.5 },
    modal: { freq: 220.0, material: "bell", decay: 0.6, brightness: 0.75, interval: 0 },
    databend: { bend: "granular", source: "demo", sample_rate: 44100, scale: "pentatonic",
                bpm: 120, divisions: 4, grain_ms: 20, density: 2.0 },
  },
  pinnedSeed: null, // global seed input; null = unpinned
  rolledSeed: rollSeed(), // rerolled only by Generate / the dice button
  auto: true,
  loop: false,
  gain: 0.8,
};

let audioCtx = null;
let gainNode = null;
let currentBuffer = null; // decoded AudioBuffer of the last render
let currentSource = null; // playing AudioBufferSourceNode, or null
let playStartTime = 0; // audioCtx.currentTime when playback began
let playing = false;
let rafId = 0;

let renderAbort = null; // AbortController for the in-flight render
let renderSeq = 0; // guards against out-of-order responses
let debounceTimer = 0;

let nameEdited = false; // user typed a custom clip name — stop auto-suggesting
let clipAudio = null; // Audio element for previewing library clips
let clipAudioFile = null; // filename currently loaded into clipAudio

// Databend source display state — NOT in state.params (that object is
// spread verbatim into the request body).
let bendSource = { label: "Water Lilies (demo)", filename: "", isDemo: true };
let bendPreviewUrl = null; // object URL of the current upload preview, if any

/* -------------------------- Parameter specs --------------------------- */

const NOISE_HINTS = {
  white: "Flat spectrum — bright, hissy, all frequencies at equal energy.",
  pink: "1/f spectrum — balanced, organic, equal energy per octave.",
  brown: "1/f² spectrum — deep, rumbling, like distant surf.",
};

// Each def: mount point, label, get/set into state, range, and README hint.
const PARAM_DEFS = [
  // Bytebeat
  { mount: "bytebeat", key: "a", label: "a · shift", min: 0, max: 31, step: 1,
    hint: "Primary right-shift. Low: dense, buzzy — high: slower rhythm, more structure.",
    get: () => state.params.bytebeat.a, set: (v) => (state.params.bytebeat.a = v) },
  { mount: "bytebeat", key: "b", label: "b · divisor", min: 1, max: 64, step: 1,
    hint: "Integer divisor. Low: fast texture — high: slower sub-pattern.",
    get: () => state.params.bytebeat.b, set: (v) => (state.params.bytebeat.b = v) },
  { mount: "bytebeat", key: "c", label: "c · shift", min: 0, max: 31, step: 1,
    hint: "Secondary right-shift. Low: harsh harmonics — high: smoother mix.",
    get: () => state.params.bytebeat.c, set: (v) => (state.params.bytebeat.c = v) },
  { mount: "bytebeat", key: "d", label: "d · mask", min: 0, max: 255, step: 1,
    hint: "AND/XOR mask (0–255). Low: sparse, glitchy — high: full texture.",
    get: () => state.params.bytebeat.d, set: (v) => (state.params.bytebeat.d = v) },
  // FM
  { mount: "fm", key: "carrier", label: "carrier Hz", min: 20, max: 2000, step: 1,
    hint: "Carrier frequency. Low: deep, bassy — high: piercing.",
    get: () => state.params.fm.carrier, set: (v) => (state.params.fm.carrier = v) },
  { mount: "fm", key: "ratio", label: "ratio", min: 0.25, max: 12, step: 0.01,
    hint: "Modulator/carrier ratio. Integer = harmonic, bell-like — non-integer = clangorous, metallic.",
    get: () => state.params.fm.ratio, set: (v) => (state.params.fm.ratio = v) },
  { mount: "fm", key: "index", label: "index", min: 0, max: 20, step: 0.1,
    hint: "Modulation depth. 0 = pure sine — higher adds ever more sidebands (bright, noisy).",
    get: () => state.params.fm.index, set: (v) => (state.params.fm.index = v) },
  // Harmonics
  { mount: "harmonics", key: "freq", label: "base Hz", min: 20, max: 1000, step: 1,
    hint: "Base (fundamental) frequency. Low: deep, bassy — high: thin.",
    get: () => state.params.harmonics.freq, set: (v) => (state.params.harmonics.freq = v) },
  { mount: "harmonics", key: "count", label: "harmonics", min: 1, max: 32, step: 1,
    hint: "Number of stacked partials. Few: pure, flute-like — many: rich, buzzy.",
    get: () => state.params.harmonics.count, set: (v) => (state.params.harmonics.count = v) },
  { mount: "harmonics", key: "decay", label: "rolloff", min: 0, max: 4, step: 0.05,
    hint: "Amplitude rolloff exponent. 1.0 ≈ sawtooth — low: bright, saw-like — high: dark, round.",
    get: () => state.params.harmonics.decay, set: (v) => (state.params.harmonics.decay = v) },
  { mount: "harmonics", key: "stretch", label: "stretch", min: 0, max: 0.02, step: 0.0005,
    hint: "Piano-string inharmonicity — bell-like above 0.005.",
    get: () => state.params.harmonics.stretch, set: (v) => (state.params.harmonics.stretch = v) },
  { mount: "harmonics", key: "drift", label: "drift", min: 0, max: 1, step: 0.05,
    hint: "Slow partial shimmer — turns tone into drone.",
    get: () => state.params.harmonics.drift, set: (v) => (state.params.harmonics.drift = v) },
  { mount: "harmonics", key: "odd_only", label: "odd only", type: "checkbox",
    hint: "Odd harmonics only — hollow, clarinet-like.",
    get: () => state.params.harmonics.odd_only, set: (v) => (state.params.harmonics.odd_only = v) },
  // Swarm
  { mount: "swarm", key: "freq", label: "freq Hz", min: 20, max: 2000, step: 1,
    hint: "Centre frequency of the voice stack. Low: massive drone — high: shrill cluster.",
    get: () => state.params.swarm.freq, set: (v) => (state.params.swarm.freq = v) },
  { mount: "swarm", key: "voices", label: "voices", min: 1, max: 16, step: 1,
    hint: "Detuned oscillators stacked in unison. 1: plain tone — 7+: thick ensemble.",
    get: () => state.params.swarm.voices, set: (v) => (state.params.swarm.voices = v) },
  { mount: "swarm", key: "wave", label: "wave", type: "select",
    options: ["saw", "sine", "square", "triangle"],
    hint: "Per-voice waveform. Saw: classic supersaw — sine: soft choir pad.",
    get: () => state.params.swarm.wave, set: (v) => (state.params.swarm.wave = v) },
  { mount: "swarm", key: "detune", label: "detune ¢", min: 0, max: 100, step: 1,
    hint: "Beating thickness — supersaw at 20-40.",
    get: () => state.params.swarm.detune, set: (v) => (state.params.swarm.detune = v) },
  { mount: "swarm", key: "drift", label: "drift", min: 0, max: 1, step: 0.01,
    hint: "Slow pitch wander — evolving drones.",
    get: () => state.params.swarm.drift, set: (v) => (state.params.swarm.drift = v) },
  // Grain cloud
  { mount: "graincloud", key: "density", label: "density /s", min: 1, max: 500, step: 1,
    hint: "Grains per second. Sparse ticks → dense wash.",
    get: () => state.params.graincloud.density, set: (v) => (state.params.graincloud.density = v) },
  { mount: "graincloud", key: "grain_ms", label: "grain ms", min: 5, max: 500, step: 1,
    hint: "Grain length. Short: clicky spray — long: smeared, pad-like.",
    get: () => state.params.graincloud.grain_ms, set: (v) => (state.params.graincloud.grain_ms = v) },
  { mount: "graincloud", key: "pitch", label: "pitch Hz", min: 20, max: 8000, step: 1,
    hint: "Centre pitch of the cloud. Low: rumbling swarm — high: glassy sparkle.",
    get: () => state.params.graincloud.pitch, set: (v) => (state.params.graincloud.pitch = v) },
  { mount: "graincloud", key: "spread", label: "spread st", min: 0, max: 48, step: 1,
    hint: "Random pitch scatter in semitones. 0: tuned — 48: four-octave haze.",
    get: () => state.params.graincloud.spread, set: (v) => (state.params.graincloud.spread = v) },
  { mount: "graincloud", key: "source", label: "source", type: "select",
    options: ["sine", "noise"],
    hint: "Grain material. Sine: tonal shimmer — noise: breathy dust.",
    get: () => state.params.graincloud.source, set: (v) => (state.params.graincloud.source = v) },
  // Crackle
  { mount: "crackle", key: "density", label: "density /s", min: 1, max: 1000, step: 1,
    hint: "Pops per second. Vinyl at low density, Geiger dust high.",
    get: () => state.params.crackle.density, set: (v) => (state.params.crackle.density = v) },
  { mount: "crackle", key: "tone", label: "tone Hz", min: 100, max: 12000, step: 10,
    hint: "Resonant centre of each pop. Low: woody knocks — high: sharp ticks.",
    get: () => state.params.crackle.tone, set: (v) => (state.params.crackle.tone = v) },
  { mount: "crackle", key: "decay_ms", label: "decay ms", min: 0.5, max: 100, step: 0.5,
    hint: "Ring-out per pop. Short: dry dust — long: pinging droplets.",
    get: () => state.params.crackle.decay_ms, set: (v) => (state.params.crackle.decay_ms = v) },
  // Pluck
  { mount: "pluck", key: "freq", label: "freq Hz", min: 20, max: 2000, step: 1,
    hint: "String pitch. Low: dub bass — high: harp-like ping.",
    get: () => state.params.pluck.freq, set: (v) => (state.params.pluck.freq = v) },
  { mount: "pluck", key: "decay", label: "decay", min: 0, max: 1, step: 0.01,
    hint: "String damping. Low: muted thud — high: long ringing sustain.",
    get: () => state.params.pluck.decay, set: (v) => (state.params.pluck.decay = v) },
  { mount: "pluck", key: "interval", label: "interval s", min: 0, max: 10, step: 0.05,
    hint: "Strum clock — 0 = one pluck.",
    get: () => state.params.pluck.interval, set: (v) => (state.params.pluck.interval = v) },
  // Riser
  { mount: "riser", key: "freq", label: "start Hz", min: 20, max: 2000, step: 1,
    hint: "Start pitch of the swept tone layer — the sweep climbs from here.",
    get: () => state.params.riser.freq, set: (v) => (state.params.riser.freq = v) },
  { mount: "riser", key: "octaves", label: "octaves", min: 0, max: 6, step: 0.1,
    hint: "Sweep span — how far pitch travels over the clip.",
    get: () => state.params.riser.octaves, set: (v) => (state.params.riser.octaves = v) },
  { mount: "riser", key: "voices", label: "voices", min: 1, max: 8, step: 1,
    hint: "Detuned saw voices. 1: clean sweep — more: thick ensemble rise.",
    get: () => state.params.riser.voices, set: (v) => (state.params.riser.voices = v) },
  { mount: "riser", key: "detune", label: "detune ¢", min: 0, max: 100, step: 1,
    hint: "Voice spread in cents — beating thickness.",
    get: () => state.params.riser.detune, set: (v) => (state.params.riser.detune = v) },
  { mount: "riser", key: "noise_mix", label: "noise mix", min: 0, max: 1, step: 0.01,
    hint: "0: pure swept tone — 1: pure noise crescendo.",
    get: () => state.params.riser.noise_mix, set: (v) => (state.params.riser.noise_mix = v) },
  { mount: "riser", key: "curve", label: "curve", min: 0.25, max: 6, step: 0.05,
    hint: "Volume-ramp shape. 1: linear — high: slow start, explosive finish.",
    get: () => state.params.riser.curve, set: (v) => (state.params.riser.curve = v) },
  { mount: "riser", key: "direction", label: "direction", type: "select",
    options: ["up", "down"],
    hint: "up: riser — down: downlifter (falling, fading mirror).",
    get: () => state.params.riser.direction, set: (v) => (state.params.riser.direction = v) },
  // Drum
  { mount: "drum", key: "drum_type", label: "type", type: "select",
    options: ["kick", "snare", "hihat", "sub"],
    hint: "One-shot recipe: kick, snare, hihat, or 808-style sub.",
    get: () => state.params.drum.drum_type, set: (v) => (state.params.drum.drum_type = v) },
  { mount: "drum", key: "tune", label: "tune st", min: -12, max: 12, step: 0.5,
    hint: "Pitch offset in semitones.",
    get: () => state.params.drum.tune, set: (v) => (state.params.drum.tune = v) },
  { mount: "drum", key: "decay", label: "decay", min: 0, max: 1, step: 0.01,
    hint: "Tail length. Tight hit → boomy ring; closed → open hat.",
    get: () => state.params.drum.decay, set: (v) => (state.params.drum.decay = v) },
  { mount: "drum", key: "tone", label: "tone", min: 0, max: 1, step: 0.01,
    hint: "Dark → bright: click, snare snap, hat sizzle, sub drive.",
    get: () => state.params.drum.tone, set: (v) => (state.params.drum.tone = v) },
  // Modal
  { mount: "modal", key: "freq", label: "freq Hz", min: 20, max: 4000, step: 1,
    hint: "Fundamental of the resonator bank.",
    get: () => state.params.modal.freq, set: (v) => (state.params.modal.freq = v) },
  { mount: "modal", key: "material", label: "material", type: "select",
    options: ["bar", "bell", "bowl", "wood"],
    hint: "Mode table: metal bar, church bell, singing bowl, woodblock.",
    get: () => state.params.modal.material, set: (v) => (state.params.modal.material = v) },
  { mount: "modal", key: "decay", label: "decay", min: 0, max: 1, step: 0.01,
    hint: "Ring time — woodblock tick to minute-long bowl.",
    get: () => state.params.modal.decay, set: (v) => (state.params.modal.decay = v) },
  { mount: "modal", key: "brightness", label: "brightness", min: 0, max: 1, step: 0.01,
    hint: "High-mode weighting. Low: fundamental only — high: full spectrum.",
    get: () => state.params.modal.brightness, set: (v) => (state.params.modal.brightness = v) },
  { mount: "modal", key: "interval", label: "interval s", min: 0, max: 10, step: 0.05,
    hint: "Strike clock — 0 = single hit.",
    get: () => state.params.modal.interval, set: (v) => (state.params.modal.interval = v) },
  // Databend — audify
  { mount: "databend-audify", key: "sample_rate", label: "rate Hz", min: 8000, max: 96000, step: 1000,
    hint: "Playback rate for the raw bytes — lower: slower + darker, higher: faster + brighter. Sets the clip's sample rate.",
    get: () => state.params.databend.sample_rate, set: (v) => (state.params.databend.sample_rate = v) },
  // Databend — scale
  { mount: "databend-scale", key: "scale", label: "scale", type: "select",
    options: ["chromatic", "major", "minor", "pentatonic", "phrygian", "lydian", "blues", "wholetone"],
    hint: "Musical scale the byte values are quantised into (5 octaves from C2).",
    get: () => state.params.databend.scale, set: (v) => (state.params.databend.scale = v) },
  { mount: "databend-scale", key: "bpm", label: "bpm", min: 30, max: 300, step: 1,
    hint: "Tempo of the byte melody.",
    get: () => state.params.databend.bpm, set: (v) => (state.params.databend.bpm = v) },
  { mount: "databend-scale", key: "divisions", label: "grid", type: "select",
    options: [ { value: "1", label: "quarter" }, { value: "2", label: "8th" },
               { value: "4", label: "16th" }, { value: "8", label: "32nd" } ],
    hint: "Note length as divisions of a beat.",
    get: () => String(state.params.databend.divisions),
    set: (v) => (state.params.databend.divisions = Number(v)) },
  // Databend — granular
  { mount: "databend-granular", key: "grain_ms", label: "grain ms", min: 5, max: 200, step: 1,
    hint: "Grain duration. Short: granular buzz — long: smoother texture.",
    get: () => state.params.databend.grain_ms, set: (v) => (state.params.databend.grain_ms = v) },
  { mount: "databend-granular", key: "density", label: "density", min: 0.5, max: 8, step: 0.1,
    hint: "Grain overlap factor. Higher: denser wash (and more bytes per second). Grain noise follows the seed.",
    get: () => state.params.databend.density, set: (v) => (state.params.databend.density = v) },
  // Global
  { mount: "global", key: "duration", label: "duration s", min: 0.05, max: 30, step: 0.05,
    hint: "Clip length in seconds.",
    get: () => state.duration, set: (v) => (state.duration = v) },
  // ADSR
  { mount: "adsr", key: "attack", label: "attack s", min: 0, max: 3, step: 0.01,
    hint: "Fade-in time, 0 → 1.",
    get: () => state.adsr.attack, set: (v) => (state.adsr.attack = v) },
  { mount: "adsr", key: "decay", label: "decay s", min: 0, max: 3, step: 0.01,
    hint: "Time to fall from 1 to the sustain level.",
    get: () => state.adsr.decay, set: (v) => (state.adsr.decay = v) },
  { mount: "adsr", key: "sustain", label: "sustain", min: 0, max: 1, step: 0.01,
    hint: "Hold level (0–1).",
    get: () => state.adsr.sustain, set: (v) => (state.adsr.sustain = v) },
  { mount: "adsr", key: "release", label: "release s", min: 0, max: 3, step: 0.01,
    hint: "Fade-out time, sustain → 0.",
    get: () => state.adsr.release, set: (v) => (state.adsr.release = v) },
  // Effects
  { mount: "effects", key: "drive", label: "drive", min: 0, max: 1, step: 0.01,
    hint: "Tanh saturation — warm to fuzz.",
    get: () => state.effects.drive, set: (v) => (state.effects.drive = v) },
  { mount: "effects", key: "fold", label: "fold", min: 0, max: 4, step: 0.05,
    hint: "Wavefolder — West Coast buzz.",
    get: () => state.effects.fold, set: (v) => (state.effects.fold = v) },
  { mount: "effects", key: "crush_bits", label: "crush bits", min: 1, max: 16, step: 1,
    hint: "Bit depth. 16 = off; grit below 8.",
    get: () => state.effects.crush_bits, set: (v) => (state.effects.crush_bits = v) },
  { mount: "effects", key: "crush_rate", label: "crush Hz", min: 0, max: 16000, step: 100,
    hint: "Sample-rate crush; 0 = off.",
    get: () => state.effects.crush_rate, set: (v) => (state.effects.crush_rate = v) },
  { mount: "effects", key: "filter_type", label: "filter", type: "select",
    options: ["off", "lowpass", "bandpass", "highpass"],
    hint: "Resonant filter, last in the chain. off = bypass.",
    get: () => state.effects.filter_type, set: (v) => (state.effects.filter_type = v) },
  { mount: "effects", key: "cutoff", label: "cutoff Hz", min: 20, max: 16000, step: 10,
    hint: "Filter cutoff — the sweep start when a sweep target is set.",
    get: () => state.effects.cutoff, set: (v) => (state.effects.cutoff = v) },
  { mount: "effects", key: "resonance", label: "res Q", min: 0.5, max: 20, step: 0.1,
    hint: "Peak at the cutoff. 0.7: flat — high: ringing, acid.",
    get: () => state.effects.resonance, set: (v) => (state.effects.resonance = v) },
  { mount: "effects", key: "cutoff_end", label: "sweep to Hz", min: 0, max: 16000, step: 10,
    hint: "0 = static; otherwise the cutoff glides here over the clip.",
    get: () => state.effects.cutoff_end, set: (v) => (state.effects.cutoff_end = v) },
];

/* ----------------------------- Helpers -------------------------------- */

const $ = (sel) => document.querySelector(sel);

function fmtNum(v) {
  const s = Number(v).toFixed(2);
  return s.replace(/\.?0+$/, "");
}

function fmtBytes(n) {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

// Prefer the FastAPI error detail (e.g. "Unknown or expired source…") over
// a bare status code when surfacing API failures in a toast.
async function apiErrorMessage(res, fallback) {
  try {
    return (await res.json()).detail || `${fallback} (${res.status})`;
  } catch {
    return `${fallback} (${res.status})`;
  }
}

let toastTimer = 0;
function toast(msg, isError = true) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("toast-error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 4000);
}

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    gainNode = audioCtx.createGain();
    gainNode.gain.value = state.gain;
    gainNode.connect(audioCtx.destination);
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

/* ------------------------- Request body ------------------------------- */

function effectiveSeed() {
  return state.pinnedSeed ?? state.rolledSeed;
}

function renderBody() {
  const params = { ...state.params[state.mode] };
  if (STOCHASTIC_MODES.includes(state.mode)) params.seed = effectiveSeed();
  if (state.mode === "databend") {
    // No top-level sample_rate/stereo — databend is mono and audify's rate
    // lives in params.sample_rate (scale/granular are fixed at 44100).
    return {
      mode: "databend",
      duration: state.duration,
      params,
      adsr: { ...state.adsr },
      effects: { ...state.effects },
    };
  }
  return {
    mode: state.mode,
    duration: state.duration,
    sample_rate: state.sampleRate,
    stereo: state.stereo,
    params,
    adsr: { ...state.adsr },
    effects: { ...state.effects },
  };
}

function suggestName() {
  const p = state.params;
  let name = "clip";
  switch (state.mode) {
    case "random":
      name = `noise-${p.random.color}-s${effectiveSeed()}`;
      break;
    case "bytebeat":
      name = `bb-a${p.bytebeat.a}-b${p.bytebeat.b}-c${p.bytebeat.c}-d${p.bytebeat.d}`;
      break;
    case "fm":
      name = `fm-${fmtNum(p.fm.carrier)}-r${fmtNum(p.fm.ratio)}-i${fmtNum(p.fm.index)}`;
      break;
    case "harmonics":
      name = `harm-${fmtNum(p.harmonics.freq)}-n${p.harmonics.count}-d${fmtNum(p.harmonics.decay)}`;
      break;
    case "swarm":
      name = `swarm-${fmtNum(p.swarm.freq)}-v${p.swarm.voices}-${p.swarm.wave}`;
      break;
    case "graincloud":
      name = `cloud-${fmtNum(p.graincloud.pitch)}-d${p.graincloud.density}`;
      break;
    case "crackle":
      name = `crackle-d${p.crackle.density}-t${fmtNum(p.crackle.tone)}`;
      break;
    case "pluck":
      name = `pluck-${fmtNum(p.pluck.freq)}-i${fmtNum(p.pluck.interval)}`;
      break;
    case "riser":
      name = `riser-${p.riser.direction}-${fmtNum(p.riser.freq)}-o${fmtNum(p.riser.octaves)}`;
      break;
    case "drum":
      name = `${p.drum.drum_type}-d${fmtNum(p.drum.decay)}-t${fmtNum(p.drum.tone)}`;
      break;
    case "modal":
      name = `modal-${p.modal.material}-${fmtNum(p.modal.freq)}`;
      break;
    case "databend": {
      const d = p.databend;
      const src = bendSource.isDemo
        ? "monet"
        : bendSource.filename.toLowerCase().replace(/\.[^.]*$/, "")
            .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 16) || "file";
      name = `bend-${d.bend}-${src}`;
      if (d.bend === "granular") name += `-s${effectiveSeed()}`;
      else if (d.bend === "scale") name += `-${d.scale}`;
      break;
    }
  }
  return state.stereo && state.mode !== "databend" ? `${name}-st` : name;
}

function refreshSuggestedName() {
  if (!nameEdited) $("#clip-name").value = suggestName();
}

function refreshSeedUI() {
  // Show the active rolled seed as the placeholder so renders stay traceable.
  $("#global-seed").placeholder = String(state.rolledSeed);
}

/* ------------------------ Control generation -------------------------- */

// Re-sync closures for every built control, so programmatic state changes
// (e.g. Randomize) can refresh the DOM without rebuilding it.
const controlSyncs = [];

function syncControls() {
  for (const sync of controlSyncs) sync();
}

function paramRow(def) {
  // The hint lives only in the title attribute (mouse-over tooltip), so
  // hovering or focusing a control never shifts the layout.
  const row = document.createElement("div");
  row.className = "param";
  row.title = def.hint;

  const label = document.createElement("label");
  label.className = "param-label";
  label.textContent = def.label;

  return { row, label };
}

function buildSlider(def) {
  const { row, label } = paramRow(def);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = def.min;
  slider.max = def.max;
  slider.step = def.step;
  slider.value = def.get();
  slider.setAttribute("aria-label", def.label);

  const num = document.createElement("input");
  num.type = "number";
  num.className = "param-num";
  num.min = def.min;
  num.max = def.max;
  num.step = def.step;
  num.value = def.get();
  num.setAttribute("aria-label", `${def.label} value`);

  const apply = (raw) => {
    let v = Number(raw);
    if (!Number.isFinite(v)) v = def.get();
    v = Math.min(def.max, Math.max(def.min, v));
    if (def.step === 1) v = Math.round(v);
    def.set(v);
    slider.value = v;
    num.value = v;
    onParamChange();
  };

  slider.addEventListener("input", () => apply(slider.value));
  num.addEventListener("change", () => apply(num.value));
  controlSyncs.push(() => {
    slider.value = def.get();
    num.value = def.get();
  });

  row.append(label, slider, num);
  return row;
}

function buildSelect(def) {
  const { row, label } = paramRow(def);

  const select = document.createElement("select");
  select.className = "param-select";
  for (const opt of def.options) {
    const o = document.createElement("option");
    o.value = typeof opt === "string" ? opt : opt.value;
    o.textContent = typeof opt === "string" ? opt : opt.label;
    select.appendChild(o);
  }
  select.value = def.get();
  select.setAttribute("aria-label", def.label);
  select.addEventListener("change", () => {
    def.set(select.value);
    onParamChange();
  });
  controlSyncs.push(() => (select.value = def.get()));

  row.append(label, select);
  return row;
}

function buildCheckbox(def) {
  const { row, label } = paramRow(def);

  const box = document.createElement("input");
  box.type = "checkbox";
  box.className = "param-check";
  box.checked = def.get();
  box.setAttribute("aria-label", def.label);
  box.addEventListener("change", () => {
    def.set(box.checked);
    onParamChange();
  });
  controlSyncs.push(() => (box.checked = def.get()));

  row.append(label, box);
  return row;
}

function buildControl(def) {
  if (def.type === "select") return buildSelect(def);
  if (def.type === "checkbox") return buildCheckbox(def);
  return buildSlider(def);
}

function buildControls() {
  const mounts = {};
  document.querySelectorAll("[data-mount]").forEach((el) => {
    mounts[el.dataset.mount] = el;
  });
  for (const def of PARAM_DEFS) mounts[def.mount].appendChild(buildControl(def));
}

/* ---------------------------- Randomize -------------------------------- */

// Frequency-like params draw log-uniform so low octaves are as likely as high.
const LOG_RANDOM_KEYS = new Set(["carrier", "freq", "pitch", "tone"]);

// Full-range uniforms are almost always harsh for these — use tamed draws.
const RANDOM_OVERRIDES = {
  "pluck.interval": () => randomBetween(0.05, 2, 0.05),
  // audify's rate is pitch-like — draw it log-uniform (it is not in
  // LOG_RANDOM_KEYS, which matches by bare key).
  "databend-audify.sample_rate": () => logRandomBetween(8000, 96000, 1000),
  // drum.tone is a 0-1 amount — without this it would hit the "tone"
  // log-random key meant for crackle's tone-in-Hz.
  "drum.tone": () => randomBetween(0, 1, 0.01),
  "modal.interval": () => randomBetween(0, 2, 0.05),
  "effects.drive": () => (Math.random() < 0.35 ? 0 : randomBetween(0, 0.7, 0.01)),
  "effects.fold": () => (Math.random() < 0.5 ? 0 : randomBetween(0, 2, 0.05)),
  "effects.crush_bits": () => (Math.random() < 0.6 ? 16 : randomBetween(4, 12, 1)),
  "effects.crush_rate": () => (Math.random() < 0.6 ? 0 : randomBetween(2000, 12000, 100)),
  "effects.filter_type": () =>
    Math.random() < 0.5 ? "off" : ["lowpass", "bandpass", "highpass"][Math.floor(Math.random() * 3)],
  "effects.cutoff": () => logRandomBetween(200, 8000, 10),
  "effects.resonance": () => randomBetween(0.7, 6, 0.1),
  "effects.cutoff_end": () => (Math.random() < 0.5 ? 0 : logRandomBetween(200, 8000, 10)),
};

function randomBetween(min, max, step) {
  const v = min + Math.random() * (max - min);
  if (step >= 1) return Math.round(v / step) * step;
  const decimals = String(step).split(".")[1]?.length ?? 0;
  return Number(v.toFixed(decimals));
}

function logRandomBetween(min, max, step) {
  const v = Math.exp(Math.log(min) + Math.random() * (Math.log(max) - Math.log(min)));
  return randomBetween(v, v, step);
}

function randomizePatch() {
  // The databend pane's bend select is hand-written, not a PARAM_DEFS entry.
  // Roll it before the loop so the new bend's sub-mount is the one randomized.
  if (state.mode === "databend") {
    const bends = ["audify", "scale", "granular"];
    state.params.databend.bend = bends[Math.floor(Math.random() * bends.length)];
    $("#bend-mode").value = state.params.databend.bend;
    updateBendMounts();
  }
  for (const def of PARAM_DEFS) {
    // Databend params live in per-bend sub-mounts; only the active bend's
    // mount randomizes. source has no def, so it can't be touched.
    const active = def.mount === state.mode || def.mount === "effects" ||
      (state.mode === "databend" && def.mount === `databend-${state.params.databend.bend}`);
    if (!active) continue;
    const override = RANDOM_OVERRIDES[`${def.mount}.${def.key}`];
    if (override) {
      def.set(override());
    } else if (def.type === "select") {
      const values = def.options.map((o) => (typeof o === "string" ? o : o.value));
      def.set(values[Math.floor(Math.random() * values.length)]);
    } else if (def.type === "checkbox") {
      def.set(Math.random() < 0.5);
    } else if (LOG_RANDOM_KEYS.has(def.key)) {
      def.set(logRandomBetween(Math.max(def.min, 1), def.max, def.step));
    } else {
      def.set(randomBetween(def.min, def.max, def.step));
    }
  }
  // The noise pane's color select is hand-written, not a PARAM_DEFS entry.
  if (state.mode === "random") {
    const colors = Object.keys(NOISE_HINTS);
    state.params.random.color = colors[Math.floor(Math.random() * colors.length)];
    $("#noise-color").value = state.params.random.color;
    $("#noise-color-hint").textContent = NOISE_HINTS[state.params.random.color];
  }
  if (state.pinnedSeed === null) state.rolledSeed = rollSeed();
  syncControls();
}

/* ----------------------------- Tabs ----------------------------------- */

// Databend output is always mono at its own rate — lock the global
// sample-rate select and stereo toggle while that tab is active.
function updateGlobalControlLocks() {
  const bend = state.mode === "databend";
  $("#sample-rate").disabled = bend;
  $("#btn-stereo").disabled = bend;
  $("#sample-rate").title = bend
    ? "Databend sets its own rate (audify rate param; scale/granular are 44100 Hz)"
    : "";
  $("#btn-stereo").title = bend ? "Databend output is mono" : "Render a 2-channel stereo file";
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((t) => {
    const active = t.dataset.mode === mode;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".pane").forEach((p) => {
    p.classList.toggle("active", p.dataset.mode === mode);
  });
  updateGlobalControlLocks();
  onParamChange();
}

/* ------------------------ Databend source ------------------------------ */

function updateBendMounts() {
  document.querySelectorAll('[data-mount^="databend-"]').forEach((el) => {
    el.hidden = el.dataset.mount !== `databend-${state.params.databend.bend}`;
  });
}

function setBendSource(source, label, isDemo, previewUrl, filename = "") {
  state.params.databend.source = source;
  bendSource = { label, filename, isDemo };
  if (bendPreviewUrl && bendPreviewUrl !== previewUrl) URL.revokeObjectURL(bendPreviewUrl);
  bendPreviewUrl = isDemo ? null : previewUrl;
  $("#bend-source-name").textContent = label;
  $("#btn-bend-demo").hidden = isDemo;
  $("#bend-caption").hidden = !isDemo;
  const img = $("#bend-image");
  if (isDemo) {
    img.src = "/static/databend-demo.jpg";
    img.hidden = false;
  } else if (previewUrl) {
    img.src = previewUrl;
    img.hidden = false;
  } else {
    img.hidden = true;
  }
  onParamChange();
}

async function uploadBendFile(file) {
  if (file.size > 16 * 1024 * 1024) {
    toast("File too large (max 16 MB)");
    return;
  }
  try {
    const res = await fetch(`/api/databend/upload?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    if (!res.ok) throw new Error(await apiErrorMessage(res, "upload failed"));
    const info = await res.json();
    const preview = file.type.startsWith("image/") ? URL.createObjectURL(file) : null;
    setBendSource(info.source, `${info.filename} · ${fmtBytes(info.size_bytes)}`, false, preview, info.filename);
  } catch (err) {
    toast(err instanceof TypeError ? "Backend not reachable" : err.message);
  }
}

/* -------------------------- Param changes ----------------------------- */

function onParamChange() {
  refreshSuggestedName();
  if (!state.auto) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => doRender(), 300);
}

/* ----------------------------- Render --------------------------------- */

async function doRender() {
  if (renderAbort) renderAbort.abort();
  const abort = new AbortController();
  renderAbort = abort;
  const seq = ++renderSeq;

  $("#render-status").hidden = false;
  try {
    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(renderBody()),
      signal: abort.signal,
    });
    if (!res.ok) throw new Error(await apiErrorMessage(res, "render failed"));
    const bytes = await res.arrayBuffer();
    const buffer = await ensureAudioCtx().decodeAudioData(bytes);
    if (seq !== renderSeq) return; // a newer render superseded this one

    currentBuffer = buffer;
    $("#scope-empty").hidden = true;
    $("#clip-duration").textContent = `${buffer.duration.toFixed(2)} s`;
    $("#btn-play").disabled = false;
    cacheWaveform();
    drawFrame();
    if (playing) startSource(); // hot-swap the buffer while playing
  } catch (err) {
    if (err.name === "AbortError") return;
    toast(err instanceof TypeError ? "Backend not reachable" : err.message);
  } finally {
    if (seq === renderSeq) $("#render-status").hidden = true;
  }
}

/* ---------------------------- Playback -------------------------------- */

function startSource() {
  stopSource();
  const ctx = ensureAudioCtx();
  const src = ctx.createBufferSource();
  src.buffer = currentBuffer;
  src.loop = state.loop;
  src.connect(gainNode);
  src.onended = () => {
    if (currentSource === src) {
      currentSource = null;
      setPlaying(false);
    }
  };
  src.start();
  currentSource = src;
  playStartTime = ctx.currentTime;
  setPlaying(true);
}

function stopSource() {
  if (currentSource) {
    const src = currentSource;
    currentSource = null; // detach before stop so onended doesn't flip state
    src.onended = null;
    try { src.stop(); } catch { /* already stopped */ }
  }
}

function setPlaying(on) {
  playing = on;
  $("#btn-play").textContent = on ? "Stop" : "Play";
  $("#btn-play").classList.toggle("active", on);
  cancelAnimationFrame(rafId);
  if (on) rafId = requestAnimationFrame(animate);
  else drawFrame();
}

async function togglePlay() {
  if (playing) {
    stopSource();
    setPlaying(false);
    return;
  }
  if (!currentBuffer) await doRender();
  if (currentBuffer) startSource();
}

/* ---------------------------- Waveform -------------------------------- */

const canvas = $("#wave");
const waveCache = document.createElement("canvas");

function accentColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#f5a623";
}

function sizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const { clientWidth: w, clientHeight: h } = canvas;
  if (w === 0 || h === 0) return;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  waveCache.width = canvas.width;
  waveCache.height = canvas.height;
  cacheWaveform();
  drawFrame();
}

function cacheWaveform() {
  const g = waveCache.getContext("2d");
  const W = waveCache.width;
  const H = waveCache.height;
  g.clearRect(0, 0, W, H);

  // centre line
  g.fillStyle = "rgba(255,255,255,0.08)";
  g.fillRect(0, H / 2, W, 1);

  if (!currentBuffer) return;
  let data = currentBuffer.getChannelData(0);
  if (currentBuffer.numberOfChannels > 1) {
    const right = currentBuffer.getChannelData(1);
    const mix = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) mix[i] = (data[i] + right[i]) / 2;
    data = mix;
  }
  const step = data.length / W;
  g.fillStyle = accentColor();
  for (let x = 0; x < W; x++) {
    const start = Math.floor(x * step);
    const end = Math.min(data.length, Math.max(start + 1, Math.floor((x + 1) * step)));
    let min = 1.0;
    let max = -1.0;
    for (let i = start; i < end; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const y1 = ((1 - max) / 2) * H;
    const y2 = ((1 - min) / 2) * H;
    g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }
}

function drawFrame() {
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.drawImage(waveCache, 0, 0);

  if (playing && currentBuffer && audioCtx) {
    const elapsed = audioCtx.currentTime - playStartTime;
    const pos = state.loop
      ? (elapsed % currentBuffer.duration) / currentBuffer.duration
      : Math.min(1, elapsed / currentBuffer.duration);
    const x = Math.round(pos * canvas.width);
    g.fillStyle = "rgba(255,255,255,0.9)";
    g.fillRect(x, 0, Math.max(1, window.devicePixelRatio || 1), canvas.height);
  }
}

function animate() {
  drawFrame();
  if (playing) rafId = requestAnimationFrame(animate);
}

/* --------------------------- Clip library ----------------------------- */

async function loadClips() {
  try {
    const res = await fetch("/api/clips");
    if (!res.ok) throw new Error(`listing clips failed (${res.status})`);
    const data = await res.json();
    renderClipList(data.clips || []);
  } catch (err) {
    renderClipList([]);
    toast(err instanceof TypeError ? "Backend not reachable" : err.message);
  }
}

function renderClipList(clips) {
  const list = $("#clip-list");
  list.replaceChildren();
  $("#library-empty").hidden = clips.length > 0;
  for (const clip of clips) list.appendChild(clipRow(clip));
}

function clipRow(clip) {
  const li = document.createElement("li");
  li.className = "clip";
  li.draggable = true;

  li.addEventListener("dragstart", (e) => {
    const href = new URL(clip.url, location.origin).href;
    e.dataTransfer.setData("DownloadURL", `audio/wav:${clip.filename}:${href}`);
    e.dataTransfer.setData("text/uri-list", href);
    e.dataTransfer.effectAllowed = "copy";
    li.classList.add("dragging");
  });
  li.addEventListener("dragend", () => li.classList.remove("dragging"));

  const name = document.createElement("span");
  name.className = "clip-name";
  name.textContent = clip.filename;
  name.title = clip.filename;

  const meta = document.createElement("span");
  meta.className = "clip-meta";
  meta.textContent = `${clip.duration_seconds.toFixed(2)} s · ${fmtBytes(clip.size_bytes)}`;

  const play = clipButton("▶", "Play clip", () => toggleClipPlayback(clip, play));

  const dl = document.createElement("a");
  dl.className = "btn btn-icon";
  dl.textContent = "↓";
  dl.title = "Download";
  dl.href = clip.url;
  dl.download = clip.filename;

  const reveal = clipButton("⌘", "Reveal in Finder", () => revealClip(clip.filename));

  const del = clipButton("✕", "Delete clip", () => {
    if (del.dataset.armed === "1") {
      deleteClip(clip.filename);
    } else {
      del.dataset.armed = "1";
      del.textContent = "sure?";
      del.classList.add("armed");
      setTimeout(() => {
        del.dataset.armed = "";
        del.textContent = "✕";
        del.classList.remove("armed");
      }, 2500);
    }
  });

  li.append(name, meta, play, dl, reveal, del);
  return li;
}

function clipButton(text, title, onClick) {
  const b = document.createElement("button");
  b.className = "btn btn-icon";
  b.textContent = text;
  b.title = title;
  b.addEventListener("click", onClick);
  return b;
}

function toggleClipPlayback(clip, btn) {
  if (clipAudio && clipAudioFile === clip.filename && !clipAudio.paused) {
    clipAudio.pause();
    return;
  }
  if (clipAudio) clipAudio.pause();
  clipAudio = new Audio(clip.url);
  clipAudioFile = clip.filename;
  const mark = (on) => {
    document.querySelectorAll(".clip .playing").forEach((el) => el.classList.remove("playing"));
    btn.classList.toggle("playing", on);
    btn.textContent = on ? "■" : "▶";
  };
  clipAudio.addEventListener("play", () => mark(true));
  clipAudio.addEventListener("pause", () => mark(false));
  clipAudio.addEventListener("ended", () => mark(false));
  clipAudio.play().catch(() => toast("Could not play clip"));
}

async function saveClip() {
  const name = $("#clip-name").value.trim() || suggestName();
  try {
    const res = await fetch("/api/clips", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...renderBody(), name }),
    });
    if (!res.ok) throw new Error(await apiErrorMessage(res, "save failed"));
    const saved = await res.json();
    toast(`Saved ${saved.filename}`, false);
    nameEdited = false;
    refreshSuggestedName();
    await loadClips();
  } catch (err) {
    toast(err instanceof TypeError ? "Backend not reachable" : err.message);
  }
}

async function deleteClip(filename) {
  try {
    const res = await fetch(`/api/clips/${encodeURIComponent(filename)}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) throw new Error(`delete failed (${res.status})`);
    if (clipAudioFile === filename && clipAudio) clipAudio.pause();
    await loadClips();
  } catch (err) {
    toast(err instanceof TypeError ? "Backend not reachable" : err.message);
  }
}

async function revealClip(filename) {
  try {
    const res = await fetch("/api/reveal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(filename ? { filename } : {}),
    });
    if (!res.ok && res.status !== 204) throw new Error(`reveal failed (${res.status})`);
  } catch (err) {
    toast(err instanceof TypeError ? "Backend not reachable" : err.message);
  }
}

/* ------------------------------ Wiring -------------------------------- */

function wire() {
  buildControls();

  // Tabs
  $("#mode-tabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab) setMode(tab.dataset.mode);
  });

  // Noise controls
  $("#noise-color").addEventListener("change", (e) => {
    state.params.random.color = e.target.value;
    $("#noise-color-hint").textContent = NOISE_HINTS[e.target.value];
    onParamChange();
  });
  // Databend controls (hand-written, like the noise color select)
  $("#bend-mode").addEventListener("change", (e) => {
    state.params.databend.bend = e.target.value;
    updateBendMounts();
    onParamChange();
  });
  $("#btn-bend-file").addEventListener("click", () => $("#bend-file").click());
  $("#bend-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) uploadBendFile(file);
    e.target.value = ""; // so re-picking the same file fires change again
  });
  $("#btn-bend-demo").addEventListener("click", () => {
    setBendSource("demo", "Water Lilies (demo)", true, null);
  });
  const dropZone = $("#bend-source");
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadBendFile(file);
  });
  // Sample rate
  $("#sample-rate").addEventListener("change", (e) => {
    state.sampleRate = Number(e.target.value);
    onParamChange();
  });

  // Global seed + stereo
  $("#global-seed").addEventListener("change", (e) => {
    const raw = e.target.value.trim();
    state.pinnedSeed = raw === "" ? null : Math.trunc(Number(raw)) || 0;
    onParamChange();
  });
  $("#btn-reroll").addEventListener("click", () => {
    state.pinnedSeed = null;
    $("#global-seed").value = "";
    state.rolledSeed = rollSeed();
    refreshSeedUI();
    refreshSuggestedName();
    doRender();
  });
  $("#btn-stereo").addEventListener("click", () => {
    state.stereo = !state.stereo;
    $("#btn-stereo").setAttribute("aria-pressed", String(state.stereo));
    $("#btn-stereo").classList.toggle("active", state.stereo);
    onParamChange();
  });

  // Transport
  $("#btn-generate").addEventListener("click", () => {
    state.rolledSeed = rollSeed(); // fresh variation on every explicit Generate
    refreshSeedUI();
    refreshSuggestedName();
    doRender();
  });
  $("#btn-random").addEventListener("click", () => {
    randomizePatch();
    refreshSeedUI();
    refreshSuggestedName();
    doRender();
  });
  $("#btn-play").addEventListener("click", togglePlay);
  $("#btn-loop").addEventListener("click", () => {
    state.loop = !state.loop;
    $("#btn-loop").setAttribute("aria-pressed", String(state.loop));
    $("#btn-loop").classList.toggle("active", state.loop);
    if (currentSource) currentSource.loop = state.loop;
  });
  $("#btn-auto").addEventListener("click", () => {
    state.auto = !state.auto;
    $("#btn-auto").setAttribute("aria-pressed", String(state.auto));
    $("#btn-auto").classList.toggle("active", state.auto);
  });
  $("#gain").addEventListener("input", (e) => {
    state.gain = Number(e.target.value);
    if (gainNode) gainNode.gain.value = state.gain;
  });

  // Library
  $("#clip-name").addEventListener("input", () => {
    nameEdited = $("#clip-name").value.trim() !== "";
  });
  $("#btn-save").addEventListener("click", saveClip);
  $("#btn-folder").addEventListener("click", () => revealClip(null));

  // Keyboard: Space toggles play/stop outside form fields
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    if (e.target instanceof Element && e.target.closest("input, select, textarea, button, a")) return;
    e.preventDefault();
    togglePlay();
  });

  window.addEventListener("resize", sizeCanvas);

  // Initial UI state
  $("#btn-auto").classList.add("active");
  updateBendMounts();
  setModeInitial();
  sizeCanvas();
  refreshSeedUI();
  refreshSuggestedName();
  loadClips();
}

function setModeInitial() {
  // Like setMode() but without triggering an auto-render before any gesture.
  document.querySelectorAll(".tab").forEach((t) => {
    const active = t.dataset.mode === state.mode;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".pane").forEach((p) => {
    p.classList.toggle("active", p.dataset.mode === state.mode);
  });
  updateGlobalControlLocks();
}

document.addEventListener("DOMContentLoaded", wire);
