import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    n_points = 1_000_000
    n_plot = 5_000

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    start = time.perf_counter()

    x = np.random.uniform(0.0, 1.0, n_points)
    y = np.random.uniform(0.0, 1.0, n_points)

    inside = x**2 + y**2 <= 1.0
    pi_estimate = 4.0 * np.mean(inside)

    end = time.perf_counter()
    execution_time = end - start

    idx = np.random.choice(n_points, n_plot, replace=False)
    x_plot = x[idx]
    y_plot = y[idx]
    inside_plot = inside[idx]

    plt.figure(figsize=(7, 7))
    plt.scatter(x_plot[inside_plot], y_plot[inside_plot], s=4, label="Inside circle")
    plt.scatter(x_plot[~inside_plot], y_plot[~inside_plot], s=4, label="Outside circle")

    theta = np.linspace(0, np.pi / 2, 300)
    plt.plot(np.cos(theta), np.sin(theta), linewidth=2, label="Quarter circle")

    plt.title("Classical NumPy Monte Carlo Pi Estimation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "classical_pi_disp.png", dpi=200)
    plt.close()

    print(f"Number of points: {n_points}")
    print(f"Estimated pi: {pi_estimate:.6f}")
    print(f"Execution time: {execution_time:.6f} seconds")
    print("Saved plot to data/classical_pi_disp.png")


if __name__ == "__main__":
    main()