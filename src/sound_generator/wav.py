"""Write float sample buffers to 16-bit WAV files."""

import wave

import numpy as np


def write_wav(filename: str, samples: np.ndarray, sample_rate: int = 44100) -> None:
    """Write a float array in [-1, 1] to a 16-bit WAV file.

    Channel count is inferred from the array shape: (n,) writes mono,
    (n, 2) writes interleaved stereo.

    Args:
        filename: Output file path.
        samples: Float audio samples; values outside [-1, 1] are clipped.
        sample_rate: Sample rate in Hz.
    """
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    channels = 2 if int16.ndim == 2 else 1
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16.tobytes())
