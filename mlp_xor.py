"""A simple MLP to solve XOR."""
import numpy as np
import math
import sys
from scipy.special import expit


this = sys.modules[__name__]

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
this.X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
this.y = np.array([[0], [1], [1], [0]])

this.X_size = this.X.shape[0]

this.z_2 = np.empty((4, 4))
this.z_3 = np.empty((4, 1))

this.a_1 = np.empty((4, 3))
this.a_2 = np.empty((4, 4))
this.a_3 = np.empty((4, 1))


def relu(x):
    """Calculate the RELU activation."""
    # return np.maximum(input_value, 0)
    return 1. * (x > 0)


def sigmoid(s):
    """Calculate sigmoid activation values."""
    # return 1 / (1 + np.exp(-s))
    return expit(s)


def derivative_sigmoid(x):
    """Calculate the derivative of sigmoid activation function."""
    return sigmoid(x) * (1 - sigmoid(x))


def forward_pass(W_1, W_2):
    """Forward propagation through the network."""
    this.a_1 = this.X
    this.z_2 = np.dot(this.a_1, W_1)
    this.a_2 = sigmoid(this.z_2)

    this.z_3 = np.dot(this.a_2, W_2)
    this.a_3 = sigmoid(this.z_3)

    return this.a_3


def backprop(y_hat, W_1, W_2):
    """Backpropagation algorithm."""
    error_3 = np.subtract(this.y, y_hat)
    delta_3 = np.multiply(error_3, derivative_sigmoid(this.z_3))
    # delta_3 = np.multiply(error_3, derivative_sigmoid(this.a_3))

    error_2 = delta_3.dot(W_2.T)
    delta_2 = np.multiply(error_2, derivative_sigmoid(this.z_2))
    # delta_2 = np.multiply(error_2, derivative_sigmoid(this.a_2))

    dj_w1 = this.a_1.T.dot(delta_2)
    dj_w2 = this.a_2.T.dot(delta_3)

    return [dj_w1, dj_w2]


def loss(predicted, actual):
    return (np.abs(predicted - actual)).mean()


def derivative_relu(x):
    return 1. * (x > 0)


def gd():
    this.w_1 = np.random.uniform(0, 1, (3, 4))
    this.w_2 = np.random.uniform(0, 1, (4, 1))

    i = 0

    while i < 60000:
        this.fp = forward_pass(this.w_1, this.w_2)

        if i % 10000 == 0:
            print(loss(this.fp, this.y))

        [dw_1, dw_2] = backprop(this.fp, this.w_1, this.w_2)

        this.w_1 += dw_1
        this.w_2 += dw_2

        i += 1


gd()
