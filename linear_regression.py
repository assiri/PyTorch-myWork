from torch import nn
import torch
from torch import tensor

x_data = tensor([[i] for i in torch.arange(101)], dtype=torch.float32)
y_data = tensor([[i * 2] for i in x_data], dtype=torch.float32)


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss()  # (reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Training loop
model.train()
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
model.eval()
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, y_pred)
