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
from ann_initialize_3 import Ann
from torch_optimizer import RAdam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score




def convert_tensor_to_one_hot(tensor):
    """
    Convert a tensor such that the highest value in the tensor is set to 1,
    and the rest of the elements are set to 0.

    Parameters:
    tensor (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Transformed tensor with highest value set to 1 and the rest set to 0.
    """
    # Find the index of the maximum value in the tensor
    max_index = torch.argmax(tensor)
    
    # Create a new tensor initialized with zeros
    result_tensor = torch.zeros_like(tensor)
    
    # Set the value at the index of the maximum value to 1
    result_tensor[max_index] = 1
    
    return result_tensor



def calculate_f1_scores(output_tensor, ground_truth_tensor):
    f1_scores = []
    for i in range(output_tensor.shape[0]):  # Iterate over each sample
        sample_f1_scores = []
        for j in range(output_tensor.shape[1]):  # Iterate over each class
            # Flatten the tensors for this sample and class
            y_true = ground_truth_tensor[i, j].detach().cpu().numpy().flatten()
            y_pred = output_tensor[i, j].detach().cpu().numpy().flatten()
            

            print(y_true.size,"y true size")
            print(y_pred.size,"y pred size")

            # Calculate F1 score for this sample and class
            f1 = f1_score(y_true, y_pred)
            sample_f1_scores.append(f1)
        f1_scores.append(sample_f1_scores)
    return f1_scores

def convert_to_one_hot_2d(input_tensor):
    max_indices = torch.argmax(input_tensor, dim=1)
    converted_tensor = torch.zeros_like(input_tensor)
    for i in range(input_tensor.shape[0]):
        converted_tensor[i, max_indices[i]] = 1
    return converted_tensor

def split_into_columns(input_tensor):
    column_tensors = []
    for col_idx in range(input_tensor.shape[1]):
        column_tensor = input_tensor[:, col_idx]
        column_tensors.append(column_tensor)
    return column_tensors



csvpath = 'C:/Users/waghs/Desktop/ser517/processed_multiclass2.csv'
data = pd.read_csv(csvpath, engine='python')
data.drop('Label', axis = 1, inplace = True)

X = data.iloc[:, :-9]  # Features (selecting all columns except last 6)
y = data.iloc[:, -9:]  # Labels (selecting last 6 columns)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_tensor_train = torch.tensor(x_train.values , dtype=torch.float32)
y_tensor_train = torch.tensor(y_train.values, dtype=torch.float32)

X_tensor_train.to('cuda')
y_tensor_train.to('cuda')

weight_decay = 0.001  

model = Ann()  
model.to('cuda')
model.load_state_dict(torch.load('model_weights_multi_new.pth'))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay = weight_decay)

l1_lambda = 0.001
l2_lambda = 0.001

num_epochs = 10000
batch_size = 10000

train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        
        loss = loss + l1_lambda * l1_reg
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")



torch.save(model.state_dict(), 'model_weights_multi_new.pth')


