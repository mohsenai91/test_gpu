import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)  # Large input/output to maximize GPU usage
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model and move it to GPUs (DataParallel will handle parallelism)
model = SimpleModel()

# Wrap the model in DataParallel to use multiple GPUs
model = nn.DataParallel(model)

# Move the model to the device (CUDA)
model = model.to(device)

# Create a large input tensor to increase memory usage
batch_size = 1024  # Increase batch size to use more memory
input_data = torch.randn(batch_size, 1024).to(device)

# Create a simple optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example forward pass with loss computation and backward pass
for epoch in range(10):  # Loop over 10 epochs for training
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_data)
    
    # Random target tensor to calculate loss (MSELoss)
    target = torch.randn(batch_size, 10).to(device)  
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Step the optimizer
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

print("Training completed.")
