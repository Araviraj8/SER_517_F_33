import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model

# Read data
csvpath = 'processed_binary_training.csv'
data = pd.read_csv(csvpath)

csvpath1 = 'test_binary.csv'
data1 = pd.read_csv(csvpath1)

# Split data into features and target
X_test = data1.drop('Label', axis=1)
y_test = data1['Label']
# Split data into features and target
X_train = data.drop('Label', axis=1)
y_train = data['Label']

# Train/test split (80/20)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Generator
latent_dim = 32

# Define the Discriminator
num_features = X_train.shape[1]
discriminator_input = layers.Input(shape=(num_features,))
x = layers.Dense(128, activation='relu')(discriminator_input)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
discriminator_output = layers.Dense(1, activation='sigmoid')(x)
discriminator = Model(discriminator_input, discriminator_output)

# Define the Generator
generator_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(64, activation='relu')(generator_input)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
generator_output = layers.Dense(num_features, activation='sigmoid')(x)
generator = Model(generator_input, generator_output)

# Define the GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

# Compile models
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train GAN
batch_size = 64
epochs = 100
for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    synthetic_samples = generator.predict(noise)
    real_samples_idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_samples = X_train.iloc[real_samples_idx]  # Use iloc for indexing DataFrame by integer positions

    X_combined = np.concatenate([real_samples, synthetic_samples])
    y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

    X_combined = np.array(X_combined, dtype=np.float32)
    y_combined = np.array(y_combined, dtype=np.float32)

    discriminator_loss = discriminator.train_on_batch(X_combined, y_combined)

    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    y_gan = np.ones((batch_size, 1))

    gan_loss = gan.train_on_batch(noise, y_gan)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}')


# Generate synthetic data
num_synthetic_samples = 1000
# noise = np.random.normal(0, 1, size=(num_synthetic_samples, latent_dim))
# synthetic_data = generator.predict(noise)

# Generate synthetic data
noise = np.random.normal(0, 1, size=(num_synthetic_samples, latent_dim))
synthetic_samples = generator.predict(noise)
fake_labels = np.ones(num_synthetic_samples)  # Assign a label of 1 to the synthetic data

# Train the classifier

# Evaluation using downstream classification task
classifier = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6)
# Measure training time
start_time = time.time()
classifier.fit(np.concatenate([X_train, synthetic_samples]), np.concatenate([y_train, fake_labels]))
training_time = time.time() - start_time
print("Training Time:", training_time, "seconds")
print(X_train.shape)
print(np.concatenate([X_train, synthetic_samples]).shape)
#classifier.fit(synthetic_data, np.zeros(num_synthetic_samples))  # Fake labels since it's synthetic data

print(X_test.shape)
start_time = time.time()
y_pred = classifier.predict(X_test)
prediction_time = time.time() - start_time
print("Prediction Time:", prediction_time, "seconds")
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on synthetic data: {accuracy}')
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)