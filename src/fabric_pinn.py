import jax
import jax.numpy as jnp
import optax
import plotly.graph_objects as go
from flax import linen as nn
from pinn_data import generate_pinn_data

ALPHA = 0.01


class HeatSurrogate(nn.Module):

    @nn.compact
    def __call__(self, x, t):

        inputs = jnp.concatenate([x, t], axis=-1)

        x = nn.Dense(32)(inputs)
        x = nn.tanh(x)

        x = nn.Dense(32)(x)
        x = nn.tanh(x)

        x = nn.Dense(32)(x)
        x = nn.tanh(x)

        x = nn.Dense(32)(x)
        x = nn.tanh(x)

        output = nn.Dense(1)(x)

        return output


def predict_u(params, model, x, t):
    x = jnp.array([[x]])
    t = jnp.array([[t]])
    return model.apply(params, x, t)[0, 0]


def pde_residual(params, model, x, t):
    du_dt = jax.grad(lambda time: predict_u(params, model, x, time))(t)

    d2u_dx2 = jax.grad(
        jax.grad(lambda space: predict_u(params, model, space, t))
    )(x)

    return du_dt - ALPHA * d2u_dx2


def physics_loss(params, model, x_pde, t_pde):
    residuals = jax.vmap(
        lambda x, t: pde_residual(params, model, x[0], t[0])
    )(x_pde, t_pde)

    return jnp.mean(residuals ** 2)

def ic_loss(params, model, x_ic, t_ic, u_ic):
    predictions = model.apply(params, x_ic, t_ic)
    return jnp.mean((predictions - u_ic) ** 2)


def bc_loss(params, model, x_bc, t_bc, u_bc):
    predictions = model.apply(params, x_bc, t_bc)
    return jnp.mean((predictions - u_bc) ** 2)


def total_loss(params, model, data):
    loss_physics = physics_loss(
        params,
        model,
        data["x_pde"],
        data["t_pde"],
    )

    loss_ic = ic_loss(
        params,
        model,
        data["x_ic"],
        data["t_ic"],
        data["u_ic"],
    )

    loss_bc = bc_loss(
        params,
        model,
        data["x_bc"],
        data["t_bc"],
        data["u_bc"],
    )

    return loss_physics + loss_ic + loss_bc

@jax.jit
def train_step(params, opt_state, data):

    loss_value, grads = jax.value_and_grad(
        lambda p: total_loss(p, model, data)
    )(params)

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params,
    )

    params = optax.apply_updates(
        params,
        updates,
    )

    return params, opt_state, loss_value

if __name__ == "__main__":

    model = HeatSurrogate()

    key = jax.random.PRNGKey(42)

    x = jnp.ones((1, 1))
    t = jnp.ones((1, 1))

    params = model.init(key, x, t)

    data = generate_pinn_data()

    optimizer = optax.adam(1e-3)

    opt_state = optimizer.init(params)

    for epoch in range(5000):

        params, opt_state, loss = train_step(
            params,
            opt_state,
            data,
        )

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:03d} | Loss = {loss:.6f}"
            )

    x_values = jnp.linspace(0.0, 1.0, 100)
    t_values = jnp.linspace(0.0, 1.0, 100)

    X, T = jnp.meshgrid(x_values, t_values)

    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    U = model.apply(params, x_flat, t_flat)
    U = U.reshape(100, 100)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=T,
                z=U,
                colorscale="Inferno",
            )
        ]
    )

    fig.update_layout(
        title="PINN Solution of the 1D Heat Equation",
        scene=dict(
            xaxis_title="Space x",
            yaxis_title="Time t",
            zaxis_title="Temperature u",
        ),
    )

    fig.write_html("data/pinn_3d_fabric.html")

    print("Saved interactive plot to data/pinn_3d_fabric.html")