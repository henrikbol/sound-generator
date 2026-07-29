"""FastAPI backend for the local sound-generator web UI."""

import re
import subprocess
import wave
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sound_generator as generate

_REPO_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = _REPO_ROOT / "clips"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_FILENAME_RE = re.compile(r"^[a-z0-9-]+\.wav$")

app = FastAPI(title="Sound Generator")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AdsrEnvelope(BaseModel):
    """ADSR envelope settings whose defaults form an identity envelope."""

    attack: float = Field(default=0.0, ge=0.0)
    decay: float = Field(default=0.0, ge=0.0)
    sustain: float = Field(default=1.0, ge=0.0, le=1.0)
    release: float = Field(default=0.0, ge=0.0)


class EffectsParams(BaseModel):
    """Distortion-chain settings whose defaults are an exact no-op."""

    drive: float = Field(default=0.0, ge=0.0, le=1.0)
    fold: float = Field(default=0.0, ge=0.0, le=4.0)
    crush_bits: int = Field(default=16, ge=1, le=16)
    crush_rate: float = Field(default=0.0, ge=0.0, le=96000.0)


class RandomParams(BaseModel):
    """Parameters for noise generation."""

    color: Literal["white", "pink", "brown"] = "white"
    seed: int | None = None


class BytebeatParams(BaseModel):
    """Parameters for bytebeat generation."""

    a: int = Field(default=8, ge=0, le=31)
    b: int = Field(default=5, ge=1)
    c: int = Field(default=3, ge=0, le=31)
    d: int = Field(default=128, ge=0, le=255)


class FmParams(BaseModel):
    """Parameters for FM synthesis."""

    carrier: float = Field(default=220.0, gt=0.0)
    ratio: float = Field(default=2.0, gt=0.0)
    index: float = Field(default=5.0, ge=0.0)


class HarmonicsParams(BaseModel):
    """Parameters for layered-harmonic synthesis."""

    freq: float = Field(default=110.0, gt=0.0)
    count: int = Field(default=8, ge=1, le=128)
    decay: float = Field(default=1.0)
    stretch: float = Field(default=0.0, ge=0.0, le=0.05)
    odd_only: bool = False
    drift: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int | None = None


class SwarmParams(BaseModel):
    """Parameters for drone-swarm synthesis."""

    freq: float = Field(default=110.0, gt=0.0, le=2000.0)
    voices: int = Field(default=7, ge=1, le=16)
    wave: Literal["saw", "sine", "square", "triangle"] = "saw"
    detune: float = Field(default=25.0, ge=0.0, le=100.0)
    drift: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int | None = None


class GrainCloudParams(BaseModel):
    """Parameters for granular-cloud synthesis."""

    density: float = Field(default=40.0, gt=0.0, le=500.0)
    grain_ms: float = Field(default=60.0, ge=5.0, le=500.0)
    pitch: float = Field(default=440.0, gt=0.0, le=8000.0)
    spread: float = Field(default=12.0, ge=0.0, le=48.0)
    source: Literal["sine", "noise"] = "sine"
    seed: int | None = None


class CrackleParams(BaseModel):
    """Parameters for crackle/dust synthesis."""

    density: float = Field(default=30.0, gt=0.0, le=1000.0)
    tone: float = Field(default=2500.0, gt=0.0, le=12000.0)
    decay_ms: float = Field(default=6.0, gt=0.0, le=100.0)
    seed: int | None = None


class PluckParams(BaseModel):
    """Parameters for Karplus-Strong pluck synthesis."""

    freq: float = Field(default=220.0, ge=20.0, le=2000.0)
    decay: float = Field(default=0.6, ge=0.0, le=1.0)
    interval: float = Field(default=0.5, ge=0.0, le=10.0)
    seed: int | None = None


class _RenderBase(BaseModel):
    duration: float = Field(ge=0.05, le=60.0)
    sample_rate: int = Field(default=44100, ge=8000, le=96000)
    stereo: bool = False
    effects: EffectsParams = Field(default_factory=EffectsParams)
    adsr: AdsrEnvelope = Field(default_factory=AdsrEnvelope)


class RandomRender(_RenderBase):
    """Render request for noise mode."""

    mode: Literal["random"]
    params: RandomParams = Field(default_factory=RandomParams)


class BytebeatRender(_RenderBase):
    """Render request for bytebeat mode."""

    mode: Literal["bytebeat"]
    params: BytebeatParams = Field(default_factory=BytebeatParams)


class FmRender(_RenderBase):
    """Render request for FM mode."""

    mode: Literal["fm"]
    params: FmParams = Field(default_factory=FmParams)


class HarmonicsRender(_RenderBase):
    """Render request for harmonics mode."""

    mode: Literal["harmonics"]
    params: HarmonicsParams = Field(default_factory=HarmonicsParams)


class SwarmRender(_RenderBase):
    """Render request for swarm mode."""

    mode: Literal["swarm"]
    params: SwarmParams = Field(default_factory=SwarmParams)


class GrainCloudRender(_RenderBase):
    """Render request for grain-cloud mode."""

    mode: Literal["graincloud"]
    params: GrainCloudParams = Field(default_factory=GrainCloudParams)


class CrackleRender(_RenderBase):
    """Render request for crackle mode."""

    mode: Literal["crackle"]
    params: CrackleParams = Field(default_factory=CrackleParams)


class PluckRender(_RenderBase):
    """Render request for pluck mode."""

    mode: Literal["pluck"]
    params: PluckParams = Field(default_factory=PluckParams)


RenderModel = (
    RandomRender
    | BytebeatRender
    | FmRender
    | HarmonicsRender
    | SwarmRender
    | GrainCloudRender
    | CrackleRender
    | PluckRender
)
RenderRequest = Annotated[RenderModel, Field(discriminator="mode")]


class _NamedMixin(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class RandomClip(RandomRender, _NamedMixin):
    """Save request for noise mode."""


class BytebeatClip(BytebeatRender, _NamedMixin):
    """Save request for bytebeat mode."""


class FmClip(FmRender, _NamedMixin):
    """Save request for FM mode."""


class HarmonicsClip(HarmonicsRender, _NamedMixin):
    """Save request for harmonics mode."""


class SwarmClip(SwarmRender, _NamedMixin):
    """Save request for swarm mode."""


class GrainCloudClip(GrainCloudRender, _NamedMixin):
    """Save request for grain-cloud mode."""


class CrackleClip(CrackleRender, _NamedMixin):
    """Save request for crackle mode."""


class PluckClip(PluckRender, _NamedMixin):
    """Save request for pluck mode."""


ClipSaveRequest = Annotated[
    RandomClip
    | BytebeatClip
    | FmClip
    | HarmonicsClip
    | SwarmClip
    | GrainCloudClip
    | CrackleClip
    | PluckClip,
    Field(discriminator="mode"),
]


class ClipInfo(BaseModel):
    """Metadata for a saved clip."""

    filename: str
    url: str
    duration_seconds: float
    size_bytes: int


class ClipListResponse(BaseModel):
    """Response wrapper for the clip listing."""

    clips: list[ClipInfo]


class RevealRequest(BaseModel):
    """Request to reveal a clip (or the clips folder) in Finder."""

    filename: str | None = None


def _synthesize(request: RenderModel) -> np.ndarray:
    """Generate samples for a render request and apply its effects and ADSR envelope."""
    if isinstance(request, RandomRender):
        samples = generate.generate_random_audio(
            request.duration,
            request.sample_rate,
            color=request.params.color,
            seed=request.params.seed,
            stereo=request.stereo,
        )
    elif isinstance(request, BytebeatRender):
        samples = generate.generate_bytebeat(
            request.duration,
            request.sample_rate,
            a=request.params.a,
            b=request.params.b,
            c=request.params.c,
            d=request.params.d,
            stereo=request.stereo,
        )
    elif isinstance(request, FmRender):
        samples = generate.generate_fm(
            request.duration,
            request.sample_rate,
            carrier=request.params.carrier,
            ratio=request.params.ratio,
            index=request.params.index,
            stereo=request.stereo,
        )
    elif isinstance(request, HarmonicsRender):
        samples = generate.generate_harmonics(
            request.duration,
            request.sample_rate,
            freq=request.params.freq,
            count=request.params.count,
            decay=request.params.decay,
            stretch=request.params.stretch,
            odd_only=request.params.odd_only,
            drift=request.params.drift,
            seed=request.params.seed,
            stereo=request.stereo,
        )
    elif isinstance(request, SwarmRender):
        samples = generate.generate_swarm(
            request.duration,
            request.sample_rate,
            freq=request.params.freq,
            voices=request.params.voices,
            wave_shape=request.params.wave,
            detune=request.params.detune,
            drift=request.params.drift,
            seed=request.params.seed,
            stereo=request.stereo,
        )
    elif isinstance(request, GrainCloudRender):
        samples = generate.generate_graincloud(
            request.duration,
            request.sample_rate,
            density=request.params.density,
            grain_ms=request.params.grain_ms,
            pitch=request.params.pitch,
            spread=request.params.spread,
            source=request.params.source,
            seed=request.params.seed,
            stereo=request.stereo,
        )
    elif isinstance(request, CrackleRender):
        samples = generate.generate_crackle(
            request.duration,
            request.sample_rate,
            density=request.params.density,
            tone=request.params.tone,
            decay_ms=request.params.decay_ms,
            seed=request.params.seed,
            stereo=request.stereo,
        )
    else:
        samples = generate.generate_pluck(
            request.duration,
            request.sample_rate,
            freq=request.params.freq,
            decay=request.params.decay,
            interval=request.params.interval,
            seed=request.params.seed,
            stereo=request.stereo,
        )
    samples = generate.apply_effects(
        samples,
        request.sample_rate,
        drive=request.effects.drive,
        fold=request.effects.fold,
        crush_bits=request.effects.crush_bits,
        crush_rate=request.effects.crush_rate,
    )
    return generate.apply_adsr(
        samples,
        request.sample_rate,
        attack=request.adsr.attack,
        decay=request.adsr.decay,
        sustain=request.adsr.sustain,
        release=request.adsr.release,
    )


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float samples in [-1, 1] as 16-bit WAV bytes.

    Channel count is inferred from the array shape: (n,) encodes mono,
    (n, 2) encodes interleaved stereo (C-order tobytes interleaves frames).
    """
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(2 if samples.ndim == 2 else 1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16.tobytes())
    return buffer.getvalue()


def _slugify(name: str) -> str:
    """Reduce a clip name to a [a-z0-9-]{1,64} slug, defaulting to "clip"."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:64].strip("-")
    return slug or "clip"


def _unique_clip_path(clips_dir: Path, slug: str) -> Path:
    """Return a path for the slug in clips_dir, suffixing -2, -3, ... on collisions."""
    candidate = clips_dir / f"{slug}.wav"
    counter = 2
    while candidate.exists():
        candidate = clips_dir / f"{slug}-{counter}.wav"
        counter += 1
    return candidate


def _validated_clip_path(filename: str) -> Path:
    """Validate a clip filename and return its path inside CLIPS_DIR."""
    if not _FILENAME_RE.fullmatch(filename):
        raise HTTPException(
            status_code=400, detail=f"Invalid clip filename: {filename!r}"
        )
    return CLIPS_DIR / filename


def _clip_info(path: Path) -> ClipInfo:
    """Build clip metadata by reading the WAV header of a saved file."""
    with wave.open(str(path), "rb") as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()
    return ClipInfo(
        filename=path.name,
        url=f"/clips/{path.name}",
        duration_seconds=duration,
        size_bytes=path.stat().st_size,
    )


@app.post("/api/render")
def render_audio(request: RenderRequest) -> Response:
    """Render the requested audio and return it as WAV bytes."""
    samples = _synthesize(request)
    return Response(
        content=_wav_bytes(samples, request.sample_rate), media_type="audio/wav"
    )


@app.post("/api/clips")
def save_clip(request: ClipSaveRequest) -> ClipInfo:
    """Render the requested audio and save it as a named clip."""
    data = _wav_bytes(_synthesize(request), request.sample_rate)
    clips_dir = CLIPS_DIR
    clips_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_clip_path(clips_dir, _slugify(request.name))
    path.write_bytes(data)
    return _clip_info(path)


@app.get("/api/clips")
def list_clips() -> ClipListResponse:
    """List saved clips, newest first."""
    clips_dir = CLIPS_DIR
    if not clips_dir.is_dir():
        return ClipListResponse(clips=[])
    paths = sorted(
        clips_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    clips = []
    for path in paths:
        try:
            clips.append(_clip_info(path))
        except (wave.Error, EOFError):
            continue
    return ClipListResponse(clips=clips)


@app.delete("/api/clips/{filename}", status_code=204)
def delete_clip(filename: str) -> Response:
    """Delete a saved clip."""
    path = _validated_clip_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such clip: {filename}")
    path.unlink()
    return Response(status_code=204)


@app.get("/clips/{filename}")
def serve_clip(filename: str) -> FileResponse:
    """Serve a saved clip as a WAV download."""
    path = _validated_clip_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such clip: {filename}")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.post("/api/reveal", status_code=204)
def reveal_clip(request: RevealRequest | None = None) -> Response:
    """Reveal a clip in Finder, or open the clips folder when no name is given."""
    filename = request.filename if request is not None else None
    if filename:
        path = _validated_clip_path(filename)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No such clip: {filename}")
        command = ["open", "-R", str(path)]
    else:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        command = ["open", str(CLIPS_DIR)]
    subprocess.run(command, check=False, timeout=10)
    return Response(status_code=204)


@app.get("/")
def index() -> Response:
    """Serve the web UI entry point, or a 503 hint while it is missing."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return JSONResponse(
            status_code=503,
            content={"detail": "UI not built yet: ui/static/index.html is missing"},
        )
    return FileResponse(index_path, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
