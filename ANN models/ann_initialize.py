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

data["sTos"].fillna(0, inplace=True)
data["sDSb"].fillna("cs0", inplace=True)
data["sTtl"].fillna(data["sTtl"].mean(), inplace=True)
data["sHops"].fillna(data["sHops"].mean(), inplace=True)


# Assuming 'target_column' is the name of your target column in your DataFrame
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])   # 1 is malicious, 0 is benign now

data = pd.get_dummies(data, columns = ['Proto', 'sDSb', 'Cause', 'State'], dtype = 'int') 
# print(one_hot_encoded_data)

print("unique values of label column", data['Label'].unique())
# # Check missing values
print(data.isnull().sum())
# # Drop columns with too many missing values
# data.drop('Cabin', axis=1, inplace=True)


# # Drop data with missing values 
# data.dropna(inplace=True)

# Inspect data
data.head()
print(data.shape)
# Transfrom attribute
# le = LabelEncoder()
# data['Sex'] = le.fit_transform(data['Sex'])


# Export DataFrame to CSV file
#data.to_csv('processed.csv', index=False)  # Set index=False if you don't want to export the index


# # Decide variables to use
# X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
# y = data['Survived']
print(data)

X = data.iloc[:, :-1]
y = data['Label']

print("y label ", y)


# Assuming 'df' is your DataFrame
data.to_csv('cleanedforAnn.csv', index=False)





# # train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

print("xtrain shape", x_train.shape)
print("ytrain shape", y_train.shape)





# Assuming df is your DataFrame with features
correlation_matrix = data.corr()



# # Plotting the heatmap correlation matrix
# plt.figure(figsize=(70, 20))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

# data[data.columns[:-1]].corr()['Label'][:]

#only accuracy- shiv
start_time = time.time()

#cuda
print("cuda",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_tensor_train = torch.tensor(x_train.values , dtype=torch.float)
y_tensor_train = torch.tensor(y_train.values, dtype=torch.float)
y_tensor_train = y_tensor_train.view(-1, 1)

X_tensor_train.to('cuda')
y_tensor_train.to('cuda')

print("y tensor size dimensions", y_tensor_train.size())
print("x tensor size dimensions", X_tensor_train.size())


print(type(X_tensor_train))


class Ann(nn.Module):
    def __init__(self):
        super(Ann,self).__init__()
        self.fc1 = nn.Linear(67,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, 64)   # Second hidden layer (64 nodes) to third hidden layer (64 nodes)
        self.fc4 = nn.Linear(64, 1)    # Third hidden layer (64 nodes) to output layer (2 nodes)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function to the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function to the second hidden layer
        x = F.relu(self.fc3(x))  # Apply ReLU activation function to the third hidden layer
        x = self.fc4(x)          # Output layer, no activation function applied
        return x

# Create an instance of the neural network
model = Ann()
model.to('cuda')

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor_train.to("cuda"))
    loss = criterion(outputs, y_tensor_train.to("cuda"))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Inference
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     test_input = torch.tensor(np.random.rand(1, 40).astype(np.float32))  # Example input data for inference
#     output = model(test_input)
#     print('Inference Output:', output)
        











# Convert test data to PyTorch tensors
X_tensor_test = torch.tensor(x_test.values, dtype=torch.float).to(device)
y_tensor_test = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1).to(device)

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(X_tensor_test)
    predicted_labels = torch.round(torch.sigmoid(outputs))  # Round the output to get binary predictions

    # Calculate accuracy
    correct = (predicted_labels == y_tensor_test).sum().item()
    total = y_tensor_test.size(0)
    accuracy = correct / total

print(f'Accuracy on test set: {accuracy:.2f}')
        
# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')


