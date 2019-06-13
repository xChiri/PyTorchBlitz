import numpy as np
import torch

device = torch.device("cpu")
# device = torch.device("cuda:0") # uncomment this line to run on GPU

# N is the batch size, D_in is the input dimension, H is the hidden dimension, D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random input and output data
x = torch.randn(N, D_in, device=device, dtype=torch.float)
y = torch.randn(N, D_out, device=device, dtype=torch.float)

# randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=torch.float)
w2 = torch.randn(H, D_out, device=device, dtype=torch.float)

learning_rate = 1e-6
for t in range(500):
    # forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2