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



# Generate synthetic data
num_synthetic_samples = 1000
noise = np.random.normal(0, 1, size=(num_synthetic_samples, latent_dim))
synthetic_data = generator.predict(noise)

# Split data into features and target
X = data.drop('Label', axis=1)
y = data['Label']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine real and synthetic data
X_combined = pd.concat([X_train, pd.DataFrame(synthetic_data, columns=X.columns)])
y_combined = np.concatenate([y_train, np.ones(num_synthetic_samples)])  # Use 1 as label for synthetic data

# Train XGBoost classifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_combined, y_combined)

# Predict on test set
y_pred = xgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")
