import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.close('all')
plt.clf()


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(16, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)  #Pytorch'un computation graph ile x'in tüm geçmişi korunur
        return x

x = torch.linspace(-3.14, 3.14, 100).unsqueeze(1)
y = torch.sin(x)

model = MyModule()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
losses = []
model.train()
for epoch in range(1000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


model.eval()
with torch.no_grad():
    y_pred = model(x)

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