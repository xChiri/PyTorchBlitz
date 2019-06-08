import numpy as np

# N is the batch size, D_in is the input dimension, H is the hidden dimension, D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 =