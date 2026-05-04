import os

# Set KERAS_BACKEND to "jax" before importing keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import numpy as np

def create_windows(signal, window_size=50):
    """
    Converts a 1D NumPy array into overlapping windows.
    """
    windows = []
    for i in range(len(signal) - window_size + 1):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

def split_train_test(windows, train_ratio=0.6):
    """
    Splits the first train_ratio portion as training data and the rest as test data.
    """
    split_index = int(len(windows) * train_ratio)
    train_data = windows[:split_index]
    test_data = windows[split_index:]
    return train_data, test_data

class SignalCompression(layers.Layer):
    """
    Compresses a 50-dimensional input into a latent dimension of 8.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.dense = layers.Dense(self.latent_dim, activation="relu")
        super().build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

class SignalExpansion(layers.Layer):
    """
    Reconstructs from a latent dimension back to the original dimension (50).
    """
    def __init__(self, output_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.dense = layers.Dense(self.output_dim)
        super().build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

class PhysicsAutoencoder(keras.Model):
    """
    Autoencoder model that applies SignalCompression then SignalExpansion.
    """
    def __init__(self, latent_dim=8, output_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(output_dim=output_dim)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    # Test block
    print("Generating random 1D signal...")
    signal = np.random.rand(200)

    print("Creating windows...")
    windows = create_windows(signal, window_size=50)
    print(f"Windows shape: {windows.shape}")

    print("Splitting into train and test sets...")
    train_windows, test_windows = split_train_test(windows, train_ratio=0.6)
    print(f"Train windows shape: {train_windows.shape}")
    print(f"Test windows shape: {test_windows.shape}")

    print("\nBuilding PhysicsAutoencoder...")
    # Initialize the model
    model = PhysicsAutoencoder(latent_dim=8, output_dim=50)
    
    # Run a dummy forward pass to build the model and initialize weights
    sample_input = train_windows[:5]
    sample_output = model(sample_input)
    
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")
    print("\nModel Summary:")
    model.summary()
