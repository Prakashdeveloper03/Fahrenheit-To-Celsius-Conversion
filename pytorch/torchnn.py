import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LinearRegression(nn.Module):
    """
    A simple linear regression model with a single input and output.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        """
        Compute the forward pass through the model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, 1).

        Returns:
        - y (torch.Tensor): Output tensor of shape (batch_size, 1).
        """
        return self.linear(x)

# Define the training data
c = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=np.float32)
f = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=np.float32)

# Create a PyTorch dataset and dataloader for the training data
train_ds = TensorDataset(torch.from_numpy(c), torch.from_numpy(f))
train_dl = DataLoader(train_ds, batch_size=len(train_ds))

# Create a model, loss function, and optimizer
model = LinearRegression()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(500):
    for xb, yb in train_dl:
        pred = model(xb.unsqueeze(1))
        loss = loss_fn(pred, yb.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Test the trained model
print("Finished training the model")
celsius = float(input("Enter the Celsius : "))
print(f"Fahrenheit conversion by model : {model(torch.tensor([celsius])).item()}")

# Calculate the Fahrenheit temperature by hand and compare with the model's prediction
fahrenheit_by_hand = (celsius * 1.8) + 32
print(f"Fahrenheit conversion by hand : {fahrenheit_by_hand}")