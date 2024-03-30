import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from ann_initialize import Ann  # Assuming Ann is the class for your model
from torch_optimizer import RAdam

# Load data
csvpath = 'C:/Users/waghs/Desktop/ser517/cleanedforAnn.csv'
data = pd.read_csv(csvpath, engine='python')

# Separate features and target
X = data.iloc[:, :-1]
y = data['Label']

# Train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

# Convert data to PyTorch tensors
X_tensor_train = torch.tensor(x_train.values, dtype=torch.float)
y_tensor_train = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)
X_tensor_test = torch.tensor(x_test.values, dtype=torch.float)
y_tensor_test = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

# Define L1 and L2 regularization parameters
l1_lambda = 0.001
l2_lambda = 0.001

# Create model
model = Ann()  # Assuming Ann is the class for your model

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Define optimizer with L2 regularization (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_lambda)

# Or if you want to use a different optimizer like RAdam, you can adjust it similarly:
# optimizer = RAdam(model.parameters(), lr=0.01, weight_decay=l2_lambda)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_tensor_train, y_tensor_train = X_tensor_train.to(device), y_tensor_train.to(device)

num_epochs = 1000
batch_size = 1000000

train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add L1 regularization to the loss
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        
        loss = loss + l1_lambda * l1_reg
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")

# Evaluate the model
model.eval()
X_tensor_test, y_tensor_test = X_tensor_test.to(device), y_tensor_test.to(device)
with torch.no_grad():
    outputs = model(X_tensor_test)
    predicted_labels = torch.round(torch.sigmoid(outputs))  # Round the output to get binary predictions
    accuracy = (predicted_labels == y_tensor_test).sum().item() / len(y_tensor_test)
    print(f"Accuracy on test set: {accuracy:.2f}")

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')
