import torch
import matplotlib.pyplot as plt
# veri
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [5.0], [7.0]])

# parametreler (öğrenecek)
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.01
losses = []
for epoch in range(100):

    # forward
    y_pred = w * x + b

    # loss (MSE)
    loss = ((y_pred - y)**2).mean()
    losses.append(loss.item())
    # backward
    loss.backward()

    # update (gradient descent)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # grad sıfırla (çok önemli!)
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: w={w.item():.4f}, b={b.item():.4f}, loss={loss.item():.4f}")

print("\nFinal:", w.item(), b.item())
plt.plot(losses)
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()