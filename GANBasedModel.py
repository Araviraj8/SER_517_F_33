import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model

# Read data
csvpath = 'Combined.csv'
data = pd.read_csv(csvpath, engine='python')

# Drop columns with high missing values or irrelevant for classification
columns_to_drop = ['sVid', 'dTos', 'dDSb', 'dTtl', 'dHops', 'SrcGap', 'DstGap',
                   'SrcWin', 'DstWin', 'dVid', 'SrcTCPBase', 'DstTCPBase',
                   'Attack Type', 'Attack Tool']
data.drop(columns_to_drop, axis=1, inplace=True)

# Impute missing values
data["sTos"].fillna(0, inplace=True)
data["sDSb"].fillna("cs0", inplace=True)
data["sTtl"].fillna(data["sTtl"].mean(), inplace=True)
data["sHops"].fillna(data["sHops"].mean(), inplace=True)

# Encode the target variable 'Label'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])  # 1 is malicious, 0 is benign now

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Proto', 'sDSb', 'Cause', 'State'], dtype='int')

# Split data into features and target
X = data.drop('Label', axis=1)
y = data['Label']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Generator
latent_dim = 32

generator_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(128)(generator_input)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(X_train.shape[1], activation='tanh')(x)

generator = Model(generator_input, x)

# Define the Discriminator
discriminator_input = layers.Input(shape=(X_train.shape[1],))
x = layers.Dense(512)(discriminator_input)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = Model(discriminator_input, x)

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the combined model (GAN)
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training
batch_size = 64
epochs = 1000

for epoch in range(epochs):
    # Train Discriminator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_data = generator.predict(noise)
    real_data = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

    X_combined_batch = np.concatenate([real_data, generated_data])
    y_combined_batch = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

    discriminator_loss = discriminator.train_on_batch(X_combined_batch, y_combined_batch)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    y_misleading = np.ones((batch_size, 1))

    generator_loss = gan.train_on_batch(noise, y_misleading)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

# Generate synthetic data
num_samples = X_test.shape[0]
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data = generator.predict(noise)

# Train CatBoost classifier on synthetic data
clf = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass', random_seed=42)
clf.fit(synthetic_data, y_train, verbose=100)

# Measure prediction time
start_time = time.time()
# Predict on the test data
y_pred = clf.predict(X_test)
prediction_time = time.time() - start_time
print("Prediction Time:", prediction_time, "seconds")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred).ravel()
print("Confusion Matrix:")
print(conf_matrix)
