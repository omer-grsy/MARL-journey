import torch
import torch.nn as nn
import torch.optim as optim

# veri (tek nokta)
# X = torch.tensor([[1.0]])
# y = torch.tensor([[3.0]])  # y = 2x + 1
X = torch.tensor([[-1.0], [0.0], [1.0]])
y = 2 * X + 1

# model
model = nn.Linear(1, 1)

# loss & optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # her 10 adımda yazdır
    if epoch % 10 == 0:
        print(f"\nEpoch {epoch}")
        for name, param in model.named_parameters():
            print(name, param.data.item())



# X = torch.tensor([[1.0]])
# y = torch.tensor([[3.0]])  # hedef: y = 2x + 1 → 3
#
# # basit model (tek layer)
# model = nn.Linear(1, 1)
#
# # loss ve optimizer
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# # başlangıç weight ve bias
# print("Başlangıç:")
# for name, param in model.named_parameters():
#     print(name, param.data)
#
# # forward
# y_pred = model(X)
# loss = criterion(y_pred, y)
#
# # backward
# loss.backward()
#
# print("\nGradientler:")
# for name, param in model.named_parameters():
#     print(name, param.grad)
#
# # update
# optimizer.step()
#
# print("\nGüncellenmiş değerler:")
# for name, param in model.named_parameters():
#     print(name, param.data)