import time
import numpy as np


N_OSCILLATORS = 100_000
N_STEPS = 1_000
DT = 0.01
DAMPING = 0.05


def simulate_legacy():
    rng = np.random.default_rng(42)

    w = rng.uniform(0.5, 2.0, size=N_OSCILLATORS)
    x = rng.normal(0.0, 1.0, size=N_OSCILLATORS)
    v = np.zeros(N_OSCILLATORS)

    for _ in range(N_STEPS):
        acceleration = -(w ** 2) * x - DAMPING * v
        v = v + acceleration * DT
        x = x + v * DT

    return x, v


if __name__ == "__main__":
    start = time.time()
    x_final, v_final = simulate_legacy()
    end = time.time()

    print(f"Legacy simulation time: {end - start:.6f} seconds")
    print(f"Final mean position: {np.mean(x_final):.6f}")
    print(f"Final mean velocity: {np.mean(v_final):.6f}")