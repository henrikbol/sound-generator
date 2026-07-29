"""Shared spectral-analysis helpers for the synthesis test suite."""

import numpy as np


def spectrum(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies in Hz, magnitude) of the one-sided FFT."""
    mag = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return freqs, mag


def band_power_ratio(samples: np.ndarray, sample_rate: int) -> float:
    """Return mean per-bin power in 20–200 Hz over mean power in 2–20 kHz."""
    freqs, mag = spectrum(samples, sample_rate)
    power = mag**2
    low = power[(freqs >= 20) & (freqs < 200)].mean()
    high = power[(freqs >= 2000) & (freqs < 20000)].mean()
    return float(low / high)
