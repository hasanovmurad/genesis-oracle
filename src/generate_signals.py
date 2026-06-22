from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)

    n = 2000
    t = np.linspace(0, 20, n)

    signal = (
        np.sin(2 * np.pi * 0.4 * t)
        + 0.35 * np.sin(2 * np.pi * 1.3 * t)
        + 0.08 * rng.normal(size=n)
    )

    malfunction_start = rng.integers(900, 1300)
    malfunction_width = 120
    malfunction_end = malfunction_start + malfunction_width

    high_freq = 2.8 * np.sin(2 * np.pi * 18 * t[malfunction_start:malfunction_end])
    corrupted = signal.copy()
    corrupted[malfunction_start:malfunction_end] += high_freq
    corrupted[malfunction_start:malfunction_end] = np.clip(
        corrupted[malfunction_start:malfunction_end],
        -1.4,
        1.4,
    )

    plt.figure(figsize=(12, 5))
    plt.plot(t, corrupted, linewidth=1.2)
    plt.title("Dynamic Wave Signal with Hidden Malfunction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(data_dir / "audit_target.png", dpi=200)
    plt.close()

    print("Saved plot to data/audit_target.png")


if __name__ == "__main__":
    main()