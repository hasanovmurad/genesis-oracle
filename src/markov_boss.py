from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


DAYS = 365


def get_transition_matrix(day):
    base = jnp.array([
        [0.85, 0.10, 0.05],
        [0.15, 0.75, 0.10],
        [0.10, 0.20, 0.70]
    ])

    shock = jnp.array([
        [0.10, 0.10, 0.80],
        [0.10, 0.10, 0.80],
        [0.10, 0.20, 0.70]
    ])

    is_shock = (day >= 180) & (day <= 190)

    return jnp.where(is_shock, shock, base)


def step(state, day):
    matrix = get_transition_matrix(day)
    next_state = state @ matrix
    return next_state, next_state


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    initial_state = jnp.array([1.0, 0.0, 0.0])

    days = jnp.arange(DAYS)

    _, history = jax.lax.scan(
        step,
        initial_state,
        days
    )

    history = jnp.vstack([initial_state, history])

    history_np = jax.device_get(history)

    plt.figure(figsize=(12, 6))

    plt.plot(history_np[:, 0], label="Bull Market")
    plt.plot(history_np[:, 1], label="Stagnation")
    plt.plot(history_np[:, 2], label="Recession")

    plt.axvspan(
        180,
        190,
        alpha=0.2,
        label="Black Swan Shock"
    )

    plt.title("Markov Chain Economic States")
    plt.xlabel("Day")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        data_dir / "markov_states.png",
        dpi=200
    )

    plt.close()

    print("Saved plot to data/markov_states.png")


if __name__ == "__main__":
    main()