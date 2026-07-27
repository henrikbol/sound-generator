# sound-generator

Pure-Python audio toy: synthesize WAV clips from formulas (`generate.py`), sonify binary files (`databend.py`), and drive it all from a local web UI (`ui/`). Output targets DAW use (Ableton) — clips, textures, granulation fodder.

## Layout

- `generate.py` — synthesis engine + CLI. 8 modes: random (white/pink/brown noise), bytebeat, fm, harmonics, swarm, graincloud, crackle, pluck. Plus `apply_effects` (drive/fold/bitcrush), `apply_adsr`, `write_wav`.
- `databend.py` — standalone binary-file sonifier (audify/scale/granular modes). Independent of generate.py.
- `ui/server.py` — FastAPI backend; imports `generate` directly. `ui/static/` — vanilla-JS frontend (no build step, no CDN).
- `tests/test_generate.py`, `tests/test_server.py` — pytest suites.
- `clips/` — WAVs saved from the web UI (gitignored, created on demand).

## Commands

```bash
uv run generate.py 5 --swarm --stereo          # CLI synthesis
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
  1. `generate.py`: `generate_<mode>()` + mutually-exclusive CLI flag + prefixed params (`--xx-*`).
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
