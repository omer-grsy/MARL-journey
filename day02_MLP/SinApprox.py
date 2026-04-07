import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# veri
x = torch.linspace(-3.14, 3.14, 100).unsqueeze(1)
y = torch.sin(x)

# model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),   # burada Tanh daha iyi
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []

# TRAINING
model.train()

for epoch in range(500):

    # forward
    y_pred = model(x)

    # loss
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# EVALUATION
model.eval()

with torch.no_grad():
    y_pred = model(x)

# GRAFİK
x_np = x.detach().numpy()
y_np = y.detach().numpy()
y_pred_np = y_pred.detach().numpy()

plt.plot(x_np, y_np, 'b', label="True")
plt.plot(x_np, y_pred_np, 'r', label="Pred")
plt.legend()
plt.title("Sin(x) Approximation")
plt.show()

# LOSS GRAPH
plt.plot(losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()