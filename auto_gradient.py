import torch

x_data = [float(i) for i in range(101)]
y_data = [float(i*2) for i in x_data]
w = torch.tensor([1.0], requires_grad=True)

# our model forward pass


def forward(x):
    return x * w

# Loss function


def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2


# Before training
print("Prediction (before training)",  4, forward(4).item())

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)  # 1) Forward pass
        l = loss(y_pred, y_val)  # 2) Compute loss
        l.backward()  # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.item())
        w.data = w.data - 0.001 * w.grad.item()

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)",  4, forward(4).item())
