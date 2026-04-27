import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

print("Backend:", keras.backend.backend())

x = keras.random.normal(shape=(3, 3))

print("Tensor:")
print(x)

print("Type:", type(x))