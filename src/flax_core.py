import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    hidden_features: int
    output_features: int

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_features)
        self.dense2 = nn.Dense(self.output_features)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x


def main():
    model = MLP(
        hidden_features=16,
        output_features=1
    )

    key = jax.random.PRNGKey(0)

    sample_input = jnp.ones((1, 4))

    params = model.init(key, sample_input)

    output = model.apply(params, sample_input)

    print("Parameters initialized successfully.")
    print("\nOutput:")
    print(output)

    print("\nFlax Explanation:")
    print(
        "Flax separates architecture from parameters. "
        "model.init explicitly creates parameters using a PRNGKey, "
        "while model.apply performs the forward pass using those "
        "parameters. Unlike Keras, the weights are not implicitly "
        "stored inside the model object."
    )


if __name__ == "__main__":
    main()
