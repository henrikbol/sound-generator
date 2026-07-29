# sound-generator

Pure-Python audio toy: synthesize WAV clips from formulas (the `generate` CLI), sonify binary files (`databend`), and drive it all from a local web UI (`ui/`). Output targets DAW use (Ableton) — clips, textures, granulation fodder.

## Layout

- `src/sound_generator/` — the synthesis package (installed editable via uv; `[project.scripts]` exposes `generate` and `databend`):
  - `dsp.py` — shared helpers (`_pan_gains`, `_new_buffer`, `_add_event`, `_normalize`, `_waveform`) + constants (`NOISE_COLORS`, `WAVEFORMS`, `GRAIN_SOURCES`). Root of the import DAG — imports nothing from the package.
  - `tones.py` — random (white/pink/brown noise), bytebeat, fm, harmonics.
  - `textures.py` — swarm, graincloud, crackle, pluck.
  - `effects.py` — `apply_effects` (drive/fold/bitcrush), `apply_adsr`.
  - `wav.py` — `write_wav`.
  - `cli.py` — argparse + mode dispatch (`generate` entry point).
  - `databend.py` — standalone binary-file sonifier (audify/scale/granular modes; `databend` entry point). Independent of the rest of the package.
  - `__init__.py` — public API re-exports (generators, effects, `write_wav`, constants).
- `ui/server.py` — FastAPI backend; `import sound_generator as generate`. `ui/static/` — vanilla-JS frontend (no build step, no CDN).
- `tests/` — pytest suites mirroring the package: `test_contract.py` (cross-mode sample-contract invariants + parametrized generator lists), `test_tones.py`, `test_textures.py`, `test_effects.py`, `test_wav.py`, `test_server.py`, with shared spectral helpers in `tests/helpers.py`.
- `clips/` — WAVs saved from the web UI (gitignored, created on demand).

## Commands

```bash
uv run generate 5 --swarm --stereo             # CLI synthesis
uv run uvicorn ui.server:app --port 8765       # web UI at http://localhost:8765
uv run ruff format . && uv run ruff check .    # format + lint
uv run ty check                                # type check
uv run pytest -q                               # full test suite
```

Run all four gates after every non-trivial change (see global workflow).

## Architecture invariants

- **Sample contract**: every generator returns float32, values in [-1, 1], shape `(n,)` mono or `(n, 2)` stereo (`stereo=True` kwarg). Channel count in WAV writers is inferred from `ndim` — never from a parameter.
- **Signal chain**: generator → `apply_effects` → `apply_adsr` → WAV. Effects and envelope default to exact no-ops.
- **Seed discipline**: stochastic generators take `seed`; RNG draws happen in a fixed order regardless of channel count, so a pinned seed gives the same performance in mono and stereo. Never use global `np.random` — always `np.random.default_rng(seed)`.
- **Adding a synthesis mode touches three layers**, each with a single extension point:
  1. `src/sound_generator/`: `generate_<mode>()` in `tones.py` or `textures.py`, re-export in `__init__.py`, mutually-exclusive CLI flag + prefixed params (`--xx-*`) in `cli.py`. Tests go in the matching `tests/test_*.py` plus the parametrized lists in `tests/test_contract.py`.
  2. `ui/server.py`: `<X>Params` model, `<X>Render`/`<X>Clip` in both discriminated unions, one `isinstance` branch in `_synthesize`.
  3. `ui/static/app.js`: `state.params.<mode>` defaults, `PARAM_DEFS` entries, `suggestName()` case; tab + pane in `index.html`.
- Frontend/backend API field names must match exactly (snake_case, e.g. `crush_bits`); the swarm API field `wave` maps to the `wave_shape` kwarg in dispatch.
- The frontend rolls a client-side seed when none is pinned and reuses it for Save, so a saved clip is identical to the last preview. Reroll happens only on the Generate button (and Randomize, which also draws fresh values for the active mode's params + effects — never the ADSR).
- Frontend controls are built once from `PARAM_DEFS`; any programmatic state change (e.g. Randomize) must call `syncControls()` to refresh the DOM. Parameter hints live ONLY in `title`-attribute tooltips — no inline hint elements (they caused layout shift on hover).

## Gotchas

- Karplus-Strong (`generate_pluck`): excitation noise must be mean-subtracted (the loop filter has unity DC gain) and the loop adds half a sample of delay — effective pitch is `sr / (period + 0.5)`.
- Spectral tests on windowed/granular material need wide tolerances (a 60 ms Hanning grain has a ~±30 Hz main lobe).
- `tests/test_server.py` monkeypatches `server.CLIPS_DIR` — server code must resolve it at call time via the module global.
- A user-level ruff config flags shebang lines in non-executable files — don't add shebangs.
- Perf budget: any 30 s / 44.1 kHz stereo render should stay well under 1 s (auto-render UX in the web UI).
- Browser drag-out (`DownloadURL`) is a macOS file promise: Finder accepts it, Ableton does not (it needs real file paths, which web pages can't put on a drag). The supported Ableton workflow is adding `clips/` to Live's Places sidebar — don't chase direct browser-to-Ableton drags.
