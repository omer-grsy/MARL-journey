import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from torch.nn import ReLU

x = torch.linspace(-3.14, 3.14, 100).unsqueeze(1)
y = torch.sin(x)

model = nn.Sequential(nn.Linear(1,32),
                      nn.ReLU(),
                      nn.Linear(32,1))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
losses = []

for epoch in range(1000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


plt.scatter(x.numpy(), y.numpy())
plt.title("sin(x) data")
plt.show()

plt.plot(losses)
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

x_np = x.detach().numpy()   #x'i detach etmek lazım plt tensor kabul etmiyor
y_np = y.detach().numpy()
y_pred_np = y_pred.detach().numpy()

plt.plot(x_np, y_np, 'b', label="True")
plt.plot(x_np, y_pred_np, 'r', label="Pred")
plt.legend()
plt.show()
