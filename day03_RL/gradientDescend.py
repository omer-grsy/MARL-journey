import random

# veri
X = [-1, 0, 1]
y = [-1, 1, 3]  # 2x + 1

# başlangıç (rastgele)
w = random.random()
b = random.random()

lr = 0.1  # learning rate

# eğitim
for epoch in range(50):
    total_loss = 0

    dw = 0
    db = 0

    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        # forward
        y_pred = w * x_i + b

        # loss
        loss = (y_pred - y_i) ** 2
        total_loss += loss

        # türevler
        dw += 2 * (y_pred - y_i) * x_i
        db += 2 * (y_pred - y_i)

    # ortalama gradient
    dw /= len(X)
    db /= len(X)

    # update
    w = w - lr * dw
    b = b - lr * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss:.4f} | w: {w:.4f} | b: {b:.4f}")