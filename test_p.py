import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Initialize the model and move it to GPUs (DataParallel will handle parallelism)
model = SimpleModel()

# Wrap the model in DataParallel to use multiple GPUs
model = nn.DataParallel(model)

# Move the model to the device (CUDA)
model = model.to(device)

# Create random input data
input_data = torch.randn(64, 10).to(device)

# Create a simple optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example forward pass with loss computation and backward pass
optimizer.zero_grad()
output = model(input_data)
target = torch.randn(64, 10).to(device)  # Random target tensor
loss = nn.MSELoss()(output, target)
loss.backward()

# Step the optimizer
optimizer.step()

print(f"Training completed. Loss: {loss.item()}")
