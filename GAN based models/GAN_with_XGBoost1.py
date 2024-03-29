import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read data
csvpath = 'combined.csv'
data = pd.read_csv(csvpath, engine='python')

# Preprocess data as needed

# Define the Generator
latent_dim = 32

generator = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_dim=latent_dim),
    layers.Dense(256, activation='relu'),
    layers.Dense(data.shape[1], activation='sigmoid')
])

# Define the Discriminator
discriminator = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_dim=data.shape[1]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the models
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training the GAN
batch_size = 64
epochs = 1000

for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    synthetic_data = generator.predict(noise)

    real_samples = data.iloc[np.random.randint(0, data.shape[0], batch_size)]
    real_labels = np.ones((batch_size, 1))
    synthetic_labels = np.zeros((batch_size, 1))

    discriminator_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    discriminator_loss_synthetic = discriminator.train_on_batch(synthetic_data, synthetic_labels)

    total_discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_synthetic)

    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {total_discriminator_loss}, GAN Loss: {generator_loss}")

# Generate synthetic data
num_synthetic_samples = 1000
noise = np.random.normal(0, 1, size=(num_synthetic_samples, latent_dim))
synthetic_data = generator.predict(noise)

# Split data into features and target
X = data.drop('Label', axis=1)
y = data['Label']


