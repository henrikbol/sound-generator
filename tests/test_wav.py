"""Tests for WAV file writing."""

import wave

import numpy as np

import sound_generator as generate

SAMPLE_RATE = 8000


def test_write_wav_round_trip(tmp_path) -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE)
    path = tmp_path / "out.wav"
    generate.write_wav(str(path), samples, SAMPLE_RATE)
    with wave.open(str(path)) as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == len(samples)


def test_write_wav_stereo_round_trip(tmp_path) -> None:
    samples = generate.generate_fm(0.5, SAMPLE_RATE, stereo=True)
    path = tmp_path / "stereo.wav"
    generate.write_wav(str(path), samples, SAMPLE_RATE)
    with wave.open(str(path)) as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == len(samples)
        frames = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    interleaved = frames.reshape(-1, 2)
    assert not np.array_equal(interleaved[:, 0], interleaved[:, 1])
