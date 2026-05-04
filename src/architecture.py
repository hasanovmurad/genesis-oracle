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
#------------------------------------------------------------------------------------------



class ConvSignalEncoder(layers.Layer):
    """
    Convolutional encoder for 1D time-series windows.

    Input shape:
        (batch_size, 50, 1)

    Output shape:
        (batch_size, 13, 8)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu"
        )
        self.conv2 = layers.Conv1D(
            filters=8,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)   # (batch, 50, 1) -> (batch, 25, 16)
        x = self.conv2(x)        # (batch, 25, 16) -> (batch, 13, 8)
        return x


class ConvSignalDecoder(layers.Layer):
    """
    Convolutional decoder for reconstructing 1D time-series windows.

    Input shape:
        (batch_size, 13, 8)

    Output shape:
        (batch_size, 50, 1)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deconv1 = layers.Conv1DTranspose(
            filters=16,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu"
        )
        self.deconv2 = layers.Conv1DTranspose(
            filters=1,
            kernel_size=3,
            strides=2,
            padding="same"
        )
        self.crop = layers.Cropping1D(cropping=(1, 1))

    def call(self, inputs):
        x = self.deconv1(inputs)   # (batch, 13, 8) -> approximately (batch, 26, 16)
        x = self.deconv2(x)        # (batch, 26, 16) -> approximately (batch, 52, 1)
        x = self.crop(x)           # (batch, 52, 1) -> (batch, 50, 1)
        return x


class ConvPhysicsAutoencoder(keras.Model):
    """
    Conv1D Autoencoder for RC time-series anomaly detection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = ConvSignalEncoder()
        self.decoder = ConvSignalDecoder()

    def call(self, inputs):
        # Original dense model used input shape (batch, 50).
        # Conv1D expects (batch, timesteps, channels), so we add one channel dimension.
        if len(inputs.shape) == 2:
            inputs = keras.ops.expand_dims(inputs, axis=-1)

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return decoded
#------------------------------------------------------------------------------------------

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
    print("\nBuilding ConvPhysicsAutoencoder...")
    conv_model = ConvPhysicsAutoencoder()

    conv_output = conv_model(sample_input)

    print(f"Conv Input shape:  {sample_input.shape}")
    print(f"Conv Output shape: {conv_output.shape}")
    print("\nConv Model Summary:")
    conv_model.summary()
