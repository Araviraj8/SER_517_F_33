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
from ann_initialize_3_binary import Ann
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








csvpath = 'C:/Users/waghs/Desktop/ser517/processed_multiclass2.csv'
data = pd.read_csv(csvpath, engine='python')
# data.drop('Label', axis = 1, inplace = True)

data.drop('AttackType_HTTPFlood', axis = 1, inplace = True)
data.drop('AttackType_ICMPFlood', axis = 1, inplace = True)
data.drop('AttackType_SYNFlood', axis = 1, inplace = True)
data.drop('AttackType_SYNScan', axis = 1, inplace = True)
data.drop('AttackType_SlowrateDoS', axis = 1, inplace = True)
data.drop('AttackType_TCPConnectScan', axis = 1, inplace = True)
data.drop('AttackType_UDPFlood', axis = 1, inplace = True)
data.drop('AttackType_UDPScan', axis = 1, inplace = True)



#AttackType_HTTPFlood,AttackType_ICMPFlood,AttackType_SYNFlood,AttackType_SYNScan,AttackType_SlowrateDoS,AttackType_TCPConnectScan,AttackType_UDPFlood,AttackType_UDPScan




X = data.iloc[:, :-1]
y = data['AttackType_Benign']


# # train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)


#cuda
print("cuda",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_tensor_train = torch.tensor(x_train.values , dtype=torch.float32)
y_tensor_train = torch.tensor(y_train.values, dtype=torch.float32)

X_tensor_train.to('cuda')
y_tensor_train.to('cuda')





# Define L2 regularization parameter (lambda)
weight_decay = 0.001  # Adjust this parameter as needed



# Load the model weights 


model = Ann()  # Assuming Ann is the class for your model
model.to('cuda')
model.load_state_dict(torch.load('model_weights_binaryfinal.pth'))


# Define loss function and optimizer
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay = weight_decay)



eval_time = time.time()

l1_lambda = 0.001
l2_lambda = 0.001

num_epochs = 100
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
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze()

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
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)



csvpath = 'C:/Users/waghs/Desktop/ser517/data_test1.csv'
data = pd.read_csv(csvpath, engine='python')

data.drop('AttackType_HTTPFlood', axis = 1, inplace = True)
data.drop('AttackType_ICMPFlood', axis = 1, inplace = True)
data.drop('AttackType_SYNFlood', axis = 1, inplace = True)
data.drop('AttackType_SYNScan', axis = 1, inplace = True)
data.drop('AttackType_SlowrateDoS', axis = 1, inplace = True)
data.drop('AttackType_TCPConnectScan', axis = 1, inplace = True)
data.drop('AttackType_UDPFlood', axis = 1, inplace = True)
data.drop('AttackType_UDPScan', axis = 1, inplace = True)

X = data.iloc[:, :-1]
y = data['AttackType_Benign']



X = torch.tensor(X.values , dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

model.to('cpu')
outputs = model(X)





model.eval()
X_tensor_train = X.to('cpu')
outputs = model(X_tensor_train)

outputs = model(X_tensor_train)

predicted_class = torch.argmax(outputs)
probabilities = F.softmax(outputs, dim=0)
threshold = 0.5

# Convert probabilities to binary values
binary_values = (probabilities > threshold).int()

print("binary values", binary_values)

print("f1 score returned first class",f1_score(binary_values, y, average='macro'), "f1 score returned first class")

print("precision score returned first class", precision_score(binary_values, y, average='macro'))

print("recall score returned 4 class",recall_score(binary_values, y, average='macro'))



# Save the model weights
torch.save(model.state_dict(), 'model_weights_binaryfinal.pth')

##########################

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






