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

csvpath = 'C:/Users/waghs/Desktop/ser517/cleanedforAnn.csv'
data = pd.read_csv(csvpath, engine='python')

X = data.iloc[:, :-1]
y = data['Label']

# print("y label ", y)

# # train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

# print("xtrain shape", x_train.shape)
# print("ytrain shape", y_train.shape)



#cuda
print("cuda",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_tensor_train = torch.tensor(x_train.values , dtype=torch.float)
y_tensor_train = torch.tensor(y_train.values, dtype=torch.float)
y_tensor_train = y_tensor_train.view(-1, 1)

X_tensor_train.to('cuda')
y_tensor_train.to('cuda')

# print("y tensor size dimensions", y_tensor_train.size())
# print("x tensor size dimensions", X_tensor_train.size())


# print(type(X_tensor_train))

# Define L2 regularization parameter (lambda)
weight_decay = 0.001  # Adjust this parameter as needed
