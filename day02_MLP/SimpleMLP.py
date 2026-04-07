import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 🔹 Veri oluştur (y = 2x + 1)
X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 2 * X + 1 + 0.2 * torch.randn(X.size())

# 🔹 Model (MLP)
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 🔹 Loss ve optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 🔹 Training
losses = []

for epoch in range(100):
    y_pred = model(X)

    loss = criterion(y_pred, y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 🔹 Grafik
plt.plot(losses)
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()