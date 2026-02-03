from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_phonon_band_structure(
    frequencies: Any,  # Array-like
    distances: Any,  # Array-like
    output_path: Path,
) -> None:
    """
    Plots the phonon band structure.

    Args:
        frequencies: 2D array of frequencies (q-points x bands).
        distances: 1D array of distances along the path.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    if len(frequencies.shape) > 1:
        for i in range(frequencies.shape[1]):
            plt.plot(distances, frequencies[:, i], color="blue")
    else:
        plt.plot(distances, frequencies, color="blue")

    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.ylabel("Frequency (THz)")
    plt.xlabel("Wave Vector")
    plt.title("Phonon Band Structure")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
