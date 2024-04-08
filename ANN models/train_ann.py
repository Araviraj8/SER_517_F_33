# Import packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder          
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from ann_initialize import Ann
from torch_optimizer import RAdam
from torch.utils.data import TensorDataset, DataLoader
csvpath = 'C:/Users/waghs/Desktop/ser517/processed_multiclass.csv'
data = pd.read_csv(csvpath, engine='python')
data.drop('Label', axis = 1, inplace = True)

# X = data.iloc[:, :-1]
# y = data['Label']

X = data.iloc[:, :-9]  # Features (selecting all columns except last 6)
y = data.iloc[:, -9:]  # Labels (selecting last 6 columns)

# print("y label ", y)

# # train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

print("xtrain shape", x_train.shape)
print("ytrain shape", y_train.shape)

#cuda
print("cuda",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_tensor_train = torch.tensor(x_train.values , dtype=torch.float32)
y_tensor_train = torch.tensor(y_train.values, dtype=torch.float32)
# y_tensor_train = y_tensor_train.reshape(-1, 9)

X_tensor_train.to('cuda')
y_tensor_train.to('cuda')

# print("y tensor size dimensions", y_tensor_train.size())
# print("x tensor size dimensions", X_tensor_train.size())


# print(type(X_tensor_train))

# Define L2 regularization parameter (lambda)
weight_decay = 0.001  # Adjust this parameter as needed



# Load the model weights
model = Ann()  # Assuming Ann is the class for your model
model.to('cuda')
model.load_state_dict(torch.load('model_weights_multi.pth'))



# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay = weight_decay)
#optimizer = RAdam(model.parameters(), lr=0.001)


eval_time = time.time()
#eval loop
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     outputs = model(X_tensor_train[0])  
#     # predicted_labels = torch.round(torch.sigmoid(outputs))
#     print("time for inference", time.time() - eval_time)



# # Training loop
# start_time_train = time.time()

# num_epochs = 100000
# print("total time training for epochs:{num_epochs}")
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(X_tensor_train.to("cuda"))
#     loss = criterion(outputs, y_tensor_train.to("cuda"))
    
#     # # L2 regularization term
#     # l2_reg = torch.tensor(0., requires_grad=True)
#     # for param in model.parameters():
#     #     l2_reg += torch.norm(param)**2
        
#     # # Add L2 regularization term to the loss function
#     # loss += weight_decay * l2_reg

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


    
#     # Print progress
#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# total_time_training = time.time() - start_time_train

# print(total_time_training)

# print("total time training for epochs:{num_epochs}", total_time_training)
# # Convert test data to PyTorch tensors
# X_tensor_test = torch.tensor(x_test.values, dtype=torch.float).to(device)
# y_tensor_test = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1).to(device)

# # Evaluate the model
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     outputs = model(X_tensor_test)
#     predicted_labels = torch.round(torch.sigmoid(outputs))  # Round the output to get binary predictions

#     # Calculate accuracy
#     correct = (predicted_labels == y_tensor_test).sum().item()
#     total = y_tensor_test.size(0)
#     accuracy = correct / total

# print(f'Accuracy on test set: {accuracy:.2f}')
        
# # Save the model weights
# torch.save(model.state_dict(), 'model_weights.pth')

    # Define L1 and L2 regularization parameters
l1_lambda = 0.001
l2_lambda = 0.001

num_epochs = 3000
batch_size = 1000
print("xtensor_train",X_tensor_train.size())
print(y_tensor_train.size())
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
# model.eval()
# X_tensor_test, y_tensor_test = X_tensor_test.to(device), y_tensor_test.to(device)
# with torch.no_grad():
#     outputs = model(X_tensor_test)
#     predicted_labels = torch.round(torch.sigmoid(outputs))  # Round the output to get binary predictions
#     accuracy = (predicted_labels == y_tensor_test).sum().item() / len(y_tensor_test)
#     print(f"Accuracy on test set: {accuracy:.2f}")
 

# Save the model weights
torch.save(model.state_dict(), 'model_weights_multi.pth')

