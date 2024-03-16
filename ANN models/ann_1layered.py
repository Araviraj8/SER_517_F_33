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
        self.fc3 = nn.Linear(64, 1)    # Third hidden layer (64 nodes) to output layer (2 nodes)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function to the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function to the second hidden layer
        x = self.fc3(x)          # Output layer, no activation function applied
        return x





if __name__ == "__main__":
    # Code here will only execute if the script is run directly,
    # not when it's imported as a module


    # Read data
    #url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'   
    csvpath = 'C:/Users/waghs/Desktop/ser517/Combined.csv'
    data = pd.read_csv(csvpath, engine='python')
    data.drop('sVid', axis = 1, inplace = True)
    data.drop('dTos', axis = 1, inplace = True)
    data.drop('dDSb', axis = 1, inplace = True)
    data.drop('dTtl', axis = 1, inplace = True)
    data.drop('dHops', axis = 1, inplace = True)
    data.drop('SrcGap', axis = 1, inplace = True)
    data.drop('DstGap', axis = 1, inplace = True)
    data.drop('SrcWin', axis = 1, inplace = True)
    data.drop('DstWin', axis = 1, inplace = True)
    data.drop('dVid', axis = 1, inplace = True)
    data.drop('SrcTCPBase', axis = 1, inplace = True)
    data.drop('DstTCPBase', axis = 1, inplace = True)
    data.drop('Attack Type', axis = 1, inplace = True)
    data.drop('Attack Tool', axis = 1, inplace = True)

    