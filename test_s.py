import torch
import torch.nn as nn

# Check if GPUs are available
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Initialize the model and move it to GPU 0
model = SimpleModel().to(device0)

# Generate random data and send to GPU 0
input_data0 = torch.randn(32, 10).to(device0)

# Generate random data for GPU 1
input_data1 = torch.randn(32, 10).to(device1)

# Forward pass on GPU 0
output0 = model(input_data0)

# Move the model to GPU 1 and run a forward pass
model = model.to(device1)
output1 = model(input_data1)

print(f"Output from GPU 0: {output0}")
print(f"Output from GPU 1: {output1}")
