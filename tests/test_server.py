"""Tests for the FastAPI backend in ui/server.py."""

import os
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ui import server

SAMPLE_RATE = 22050


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """Return a test client whose CLIPS_DIR points at a temp directory."""
    monkeypatch.setattr(server, "CLIPS_DIR", tmp_path)
    return TestClient(server.app)


def last_frames(wav_data: bytes, count: int) -> np.ndarray:
    """Return the final `count` int16 frames of a WAV byte string."""
    with wave.open(BytesIO(wav_data), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype="<i2")[-count:]


@pytest.mark.parametrize(
    "mode",
    [
        "random",
        "bytebeat",
        "fm",
        "harmonics",
        "swarm",
        "graincloud",
        "crackle",
        "pluck",
    ],
)
def test_render_returns_valid_wav(client: TestClient, mode: str) -> None:
    response = client.post(
        "/api/render",
        json={"mode": mode, "duration": 0.25, "sample_rate": SAMPLE_RATE},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == int(0.25 * SAMPLE_RATE)


@pytest.mark.parametrize("mode", ["fm", "swarm", "graincloud", "pluck"])
def test_render_stereo_wav(client: TestClient, mode: str) -> None:
    response = client.post(
        "/api/render",
        json={
            "mode": mode,
            "duration": 0.25,
            "sample_rate": SAMPLE_RATE,
            "stereo": True,
        },
    )
    assert response.status_code == 200
    expected_frames = int(0.25 * SAMPLE_RATE)
    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == expected_frames
        raw = wav_file.readframes(wav_file.getnframes())
    frames = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
    assert frames.shape == (expected_frames, 2)


def test_render_seeded_noise_is_deterministic(client: TestClient) -> None:
    body = {
        "mode": "random",
        "duration": 0.1,
        "sample_rate": SAMPLE_RATE,
        "params": {"color": "pink", "seed": 42},
    }
    first = client.post("/api/render", json=body)
    second = client.post("/api/render", json=body)
    assert first.content == second.content


def test_render_seeded_texture_is_deterministic(client: TestClient) -> None:
    body = {
        "mode": "graincloud",
        "duration": 0.2,
        "sample_rate": SAMPLE_RATE,
        "params": {"seed": 7},
    }
    first = client.post("/api/render", json=body)
    second = client.post("/api/render", json=body)
    reseeded = client.post("/api/render", json={**body, "params": {"seed": 8}})
    assert first.status_code == 200
    assert reseeded.status_code == 200
    assert first.content == second.content
    assert first.content != reseeded.content


def test_default_effects_block_is_identity(client: TestClient) -> None:
    body = {"mode": "fm", "duration": 0.25, "sample_rate": SAMPLE_RATE}
    plain = client.post("/api/render", json=body)
    explicit = client.post(
        "/api/render",
        json={
            **body,
            "effects": {"drive": 0.0, "fold": 0.0, "crush_bits": 16, "crush_rate": 0.0},
        },
    )
    assert plain.status_code == 200
    assert explicit.status_code == 200
    assert plain.content == explicit.content


@pytest.mark.parametrize(
    "effects",
    [
        pytest.param({"crush_bits": 2}, id="bitcrush"),
        pytest.param({"drive": 0.8}, id="drive"),
    ],
)
def test_effects_change_the_output(
    client: TestClient, effects: dict[str, object]
) -> None:
    body = {"mode": "fm", "duration": 0.25, "sample_rate": SAMPLE_RATE}
    clean = client.post("/api/render", json=body)
    processed = client.post("/api/render", json={**body, "effects": effects})
    assert clean.status_code == 200
    assert processed.status_code == 200
    assert clean.content != processed.content


def test_adsr_release_silences_tail(client: TestClient) -> None:
    body = {"mode": "fm", "duration": 0.5}
    plain = client.post("/api/render", json=body)
    released = client.post("/api/render", json={**body, "adsr": {"release": 0.5}})
    assert plain.status_code == 200
    assert released.status_code == 200
    assert np.abs(last_frames(plain.content, 200)).max() > 3000
    assert np.abs(last_frames(released.content, 200)).max() < 500


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"mode": "chiptune", "duration": 1.0}, id="unknown-mode"),
        pytest.param({"mode": "fm", "duration": 0.01}, id="duration-too-short"),
        pytest.param({"mode": "fm", "duration": 120.0}, id="duration-too-long"),
        pytest.param(
            {"mode": "fm", "duration": 1.0, "sample_rate": 1000}, id="bad-rate"
        ),
        pytest.param(
            {"mode": "random", "duration": 1.0, "params": {"color": "mauve"}},
            id="bad-color",
        ),
        pytest.param(
            {"mode": "bytebeat", "duration": 1.0, "params": {"a": 99}}, id="bad-shift"
        ),
        pytest.param(
            {"mode": "harmonics", "duration": 1.0, "params": {"count": 0}},
            id="bad-count",
        ),
        pytest.param(
            {"mode": "fm", "duration": 1.0, "params": {"carrier": -220.0}},
            id="bad-frequency",
        ),
        pytest.param(
            {"mode": "fm", "duration": 1.0, "adsr": {"sustain": 2.0}},
            id="bad-sustain",
        ),
        pytest.param(
            {"mode": "fm", "duration": 1.0, "adsr": {"attack": -0.1}},
            id="negative-attack",
        ),
        pytest.param(
            {"mode": "swarm", "duration": 1.0, "params": {"voices": 0}},
            id="swarm-voices-low",
        ),
        pytest.param(
            {"mode": "swarm", "duration": 1.0, "params": {"voices": 17}},
            id="swarm-voices-high",
        ),
        pytest.param(
            {"mode": "graincloud", "duration": 1.0, "params": {"density": 0}},
            id="graincloud-density-zero",
        ),
        pytest.param(
            {"mode": "crackle", "duration": 1.0, "params": {"decay_ms": 0}},
            id="crackle-decay-zero",
        ),
        pytest.param(
            {"mode": "pluck", "duration": 1.0, "params": {"freq": 5000}},
            id="pluck-freq-high",
        ),
        pytest.param(
            {"mode": "fm", "duration": 1.0, "effects": {"crush_bits": 0}},
            id="bad-crush-bits",
        ),
        pytest.param(
            {"mode": "fm", "duration": 1.0, "effects": {"drive": 2.0}},
            id="bad-drive",
        ),
        pytest.param(
            {"mode": "harmonics", "duration": 1.0, "params": {"stretch": -0.1}},
            id="bad-stretch",
        ),
        pytest.param(
            {"mode": "fm", "duration": 1.0, "stereo": "banana"},
            id="bad-stereo",
        ),
    ],
)
def test_render_rejects_invalid_input(
    client: TestClient, body: dict[str, object]
) -> None:
    assert client.post("/api/render", json=body).status_code == 422


def test_clip_round_trip(client: TestClient, tmp_path: Path) -> None:
    body = {
        "mode": "harmonics",
        "duration": 0.25,
        "sample_rate": SAMPLE_RATE,
        "name": "My Clip!",
    }
    saved_response = client.post("/api/clips", json=body)
    assert saved_response.status_code == 200
    saved = saved_response.json()
    assert saved["filename"] == "my-clip.wav"
    assert saved["url"] == "/clips/my-clip.wav"
    assert saved["duration_seconds"] == pytest.approx(0.25, abs=0.001)
    assert saved["size_bytes"] == (tmp_path / "my-clip.wav").stat().st_size

    listing = client.get("/api/clips").json()
    assert [clip["filename"] for clip in listing["clips"]] == ["my-clip.wav"]
    assert listing["clips"][0] == saved

    served = client.get("/clips/my-clip.wav")
    assert served.status_code == 200
    assert served.headers["content-type"] == "audio/wav"
    assert served.content == (tmp_path / "my-clip.wav").read_bytes()

    assert client.delete("/api/clips/my-clip.wav").status_code == 204
    assert not (tmp_path / "my-clip.wav").exists()
    assert client.delete("/api/clips/my-clip.wav").status_code == 404
    assert client.get("/clips/my-clip.wav").status_code == 404


def test_stereo_clip_round_trip(client: TestClient, tmp_path: Path) -> None:
    body = {
        "mode": "pluck",
        "duration": 0.25,
        "sample_rate": SAMPLE_RATE,
        "stereo": True,
        "name": "stereo pluck",
    }
    saved_response = client.post("/api/clips", json=body)
    assert saved_response.status_code == 200
    saved = saved_response.json()
    assert saved["filename"] == "stereo-pluck.wav"
    assert saved["duration_seconds"] == pytest.approx(0.25, abs=0.001)
    with wave.open(str(tmp_path / "stereo-pluck.wav"), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == int(0.25 * SAMPLE_RATE)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("My Clip", "my-clip.wav"),
        ("  Weird   ##  Name!! ", "weird-name.wav"),
        ("UPPER_case.stuff", "upper-case-stuff.wav"),
        ("a" * 100, f"{'a' * 64}.wav"),
        ("###", "clip.wav"),
    ],
)
def test_clip_name_slugification(client: TestClient, name: str, expected: str) -> None:
    response = client.post(
        "/api/clips", json={"mode": "fm", "duration": 0.1, "name": name}
    )
    assert response.status_code == 200
    assert response.json()["filename"] == expected


def test_duplicate_clip_names_get_suffixes(client: TestClient) -> None:
    body = {"mode": "fm", "duration": 0.1, "name": "Loop"}
    filenames = [
        client.post("/api/clips", json=body).json()["filename"] for _ in range(3)
    ]
    assert filenames == ["loop.wav", "loop-2.wav", "loop-3.wav"]


def test_clip_empty_name_rejected(client: TestClient) -> None:
    body = {"mode": "fm", "duration": 0.1, "name": ""}
    assert client.post("/api/clips", json=body).status_code == 422


def test_clips_listed_newest_first(client: TestClient, tmp_path: Path) -> None:
    for name in ["first", "second", "third"]:
        response = client.post(
            "/api/clips", json={"mode": "fm", "duration": 0.1, "name": name}
        )
        assert response.status_code == 200
    for age, filename in enumerate(["third.wav", "second.wav", "first.wav"]):
        mtime = 1_700_000_000 - age * 60
        os.utime(tmp_path / filename, (mtime, mtime))
    listing = client.get("/api/clips").json()
    filenames = [clip["filename"] for clip in listing["clips"]]
    assert filenames == ["third.wav", "second.wav", "first.wav"]


def test_list_clips_empty_when_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "CLIPS_DIR", tmp_path / "missing")
    client = TestClient(server.app)
    assert client.get("/api/clips").json() == {"clips": []}


@pytest.mark.parametrize(
    "filename",
    [
        "..%2F..%2Fescape.wav",
        "escape.txt",
        "UPPER.wav",
        "semi;colon.wav",
        ".wav",
        "sneaky%20name.wav",
    ],
)
def test_bad_clip_filenames_rejected(
    client: TestClient, tmp_path: Path, filename: str
) -> None:
    outside = tmp_path.parent / "escape.wav"
    outside.write_bytes(b"secret-bytes")
    get_response = client.get(f"/clips/{filename}")
    delete_response = client.delete(f"/api/clips/{filename}")
    assert get_response.status_code in (400, 404)
    assert delete_response.status_code in (400, 404)
    assert b"secret-bytes" not in get_response.content
    assert outside.exists()


def test_reveal_clip_runs_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        calls.append(list(command))

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    saved = client.post(
        "/api/clips", json={"mode": "fm", "duration": 0.1, "name": "reveal me"}
    ).json()

    response = client.post("/api/reveal", json={"filename": saved["filename"]})
    assert response.status_code == 204
    assert calls[-1] == ["open", "-R", str(tmp_path / "reveal-me.wav")]

    response = client.post("/api/reveal", json={})
    assert response.status_code == 204
    assert calls[-1] == ["open", str(tmp_path)]


def test_reveal_missing_clip_is_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        calls.append(list(command))

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    response = client.post("/api/reveal", json={"filename": "nope.wav"})
    assert response.status_code == 404
    assert calls == []


def test_index_reports_missing_ui_then_serves_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    static_dir = tmp_path / "static"
    monkeypatch.setattr(server, "STATIC_DIR", static_dir)
    assert client.get("/").status_code == 503

    static_dir.mkdir()
    (static_dir / "index.html").write_text("<h1>sound generator</h1>")
    response = client.get("/")
    assert response.status_code == 200
    assert "sound generator" in response.text
