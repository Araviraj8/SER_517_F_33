import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical

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
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])  # Encode labels numerically

# Split data into features and target
X = data.drop('Label', axis=1)
y = data['Label']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Proto', 'sDSb', 'Cause', 'State'], dtype='int')

# Convert target labels to one-hot encoding
y = to_categorical(y)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

