import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size2, 1)  # Output layer with one node

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.output_layer(x)
        return x

# Example usage:
input_size = 10  # Example input size
hidden_size1 = 20
hidden_size2 = 10
model = NeuralNetwork(input_size, hidden_size1, hidden_size2)
print(model)
