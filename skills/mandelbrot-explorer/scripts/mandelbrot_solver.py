from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run_simulation(center_real, center_imag, zoom, resolution=250, max_iterations=200, output_path=None):
    x = np.linspace(center_real - 1.5 / zoom, center_real + 1.5 / zoom, resolution)
    y = np.linspace(center_imag - 1.5 / zoom, center_imag + 1.5 / zoom, resolution)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    counts = np.zeros(C.shape, dtype=np.int32)
    active = np.ones(C.shape, dtype=bool)

    for k in range(max_iterations):
        Z[active] = Z[active] ** 2 + C[active]
        escaped = np.abs(Z) > 2
        newly_escaped = escaped & active
        counts[newly_escaped] = k
        active &= ~escaped

    counts[active] = max_iterations

    hist, _ = np.histogram(counts, bins=20)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))

    boundary = np.sum((counts > 0) & (counts < max_iterations))
    boundary_ratio = boundary / counts.size

    metrics = {
        "entropy": float(entropy),
        "boundary_complexity": float(boundary_ratio),
        "center_real": float(center_real),
        "center_imag": float(center_imag),
        "zoom": float(zoom),
        "max_iterations": int(max_iterations),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(counts, cmap="twilight_shifted")
        plt.colorbar(label="Iterations until escape")
        plt.title(f"Mandelbrot center=({center_real:.5f}, {center_imag:.5f}), zoom={zoom}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    return counts, metrics


if __name__ == "__main__":
    _, metrics = run_simulation(
        center_real=-0.745,
        center_imag=0.105,
        zoom=250.0,
        output_path="data/mandelbrot_step3.png",
    )

    print("Simulation Metrics:")
    print(metrics)
    print("Saved plot to data/mandelbrot_global.png")