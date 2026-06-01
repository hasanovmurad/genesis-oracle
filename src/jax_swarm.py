import time
import jax
import jax.numpy as jnp


N_OSCILLATORS = 100_000
N_STEPS = 1_000
DT = 0.01
DAMPING = 0.05


def oscillator_step(x, v, w):
    acceleration = -(w ** 2) * x - DAMPING * v
    v_new = v + acceleration * DT
    x_new = x + v_new * DT
    return x_new, v_new


batched_oscillator_step = jax.vmap(oscillator_step, in_axes=(0, 0, 0))


@jax.jit
def simulate_jax(x, v, w):
    for _ in range(N_STEPS):
        x, v = batched_oscillator_step(x, v, w)
    return x, v


def projectile_loss(v_initial):
    target_distance = 150.0
    time_of_flight = 5.0
    simulated_distance = v_initial * time_of_flight
    loss = (simulated_distance - target_distance) ** 2
    return loss


def optimize_projectile_velocity():
    grad_fn = jax.grad(projectile_loss)

    v_initial = 5.0
    learning_rate = 0.01

    for i in range(20):
        gradient = grad_fn(v_initial)
        v_initial = v_initial - learning_rate * gradient
        print(
            f"Iteration {i + 1}: "
            f"v_initial={v_initial:.6f}, "
            f"loss={projectile_loss(v_initial):.6f}"
        )

    print(f"Optimized initial velocity: {v_initial:.6f} m/s")


def main():
    key = jax.random.PRNGKey(42)
    key_w, key_x = jax.random.split(key)

    w = jax.random.uniform(key_w, shape=(N_OSCILLATORS,), minval=0.5, maxval=2.0)
    x = jax.random.normal(key_x, shape=(N_OSCILLATORS,))
    v = jnp.zeros(N_OSCILLATORS)

    print("Warm-up run / compilation...")
    x_final, v_final = simulate_jax(x, v, w)
    x_final.block_until_ready()

    print("Timed second run...")
    start = time.time()
    x_final, v_final = simulate_jax(x, v, w)
    x_final.block_until_ready()
    end = time.time()

    print(f"JAX second run time: {end - start:.6f} seconds")
    print(f"Final mean position: {jnp.mean(x_final):.6f}")
    print(f"Final mean velocity: {jnp.mean(v_final):.6f}")

    print("\nProjectile optimization:")
    optimize_projectile_velocity()


if __name__ == "__main__":
    main()
