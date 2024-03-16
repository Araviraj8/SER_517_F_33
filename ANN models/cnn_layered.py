import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'relu'], ['conv2', 'relu']], inplace=True)

    def train_model(self, train_loader, num_epochs=10, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Enable quantization for the model
        self.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self, inplace=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters())

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs.squeeze(), labels.float())

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update the running loss
                running_loss += loss.item()

            # Print the average loss for the epoch
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Convert the model to quantized form
        torch.quantization.convert(self, inplace=True)

        # Save the quantized model
        torch.save(self.state_dict(), "quantized_cnn.pth")
