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




class Ann(nn.Module):
    def __init__(self):
        super(Ann,self).__init__()
        self.fc1 = nn.Linear(67,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, 64)   # Second hidden layer (64 nodes) to third hidden layer (64 nodes)
        self.fc4 = nn.Linear(64, 32)   # Second hidden layer (64 nodes) to third hidden layer (64 nodes)

        self.fc5 = nn.Linear(32, 1)    # Third hidden layer (64 nodes) to output layer (2 nodes)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function to the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function to the second hidden layer
        x = F.relu(self.fc3(x))  # Apply ReLU activation function to the third hidden layer
        x = F.relu(self.fc4(x))  # Apply ReLU activation function to the third hidden layer

        x = self.fc4(x)          # Output layer, no activation function applied
        return x



