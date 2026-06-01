import jax
import jax.numpy as jnp


def generate_pinn_data():
    key = jax.random.PRNGKey(42)

    key_pde_x, key_pde_t, key_ic, key_bc = jax.random.split(key, 4)

    # PDE collocation points
    x_pde = jax.random.uniform(
        key_pde_x,
        shape=(5000, 1),
        minval=0.0,
        maxval=1.0,
    )

    t_pde = jax.random.uniform(
        key_pde_t,
        shape=(5000, 1),
        minval=0.0,
        maxval=1.0,
    )

    # Initial condition (t = 0)
    x_ic = jax.random.uniform(
        key_ic,
        shape=(500, 1),
        minval=0.0,
        maxval=1.0,
    )

    t_ic = jnp.zeros((500, 1))

    u_ic = -jnp.sin(jnp.pi * x_ic)

    # Boundary conditions
    t_bc = jax.random.uniform(
        key_bc,
        shape=(500, 1),
        minval=0.0,
        maxval=1.0,
    )

    x_left = jnp.zeros((250, 1))
    x_right = jnp.ones((250, 1))

    x_bc = jnp.vstack([x_left, x_right])

    u_bc = jnp.zeros((500, 1))

    return {
        "x_pde": x_pde,
        "t_pde": t_pde,
        "x_ic": x_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "x_bc": x_bc,
        "t_bc": t_bc,
        "u_bc": u_bc,
    }


if __name__ == "__main__":
    data = generate_pinn_data()

    for key, value in data.items():
        print(key, value.shape)