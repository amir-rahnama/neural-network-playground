"""A simple MLP to solve XOR."""
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.special import expit
from numpy import linalg as LA
import math


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

this.learning_rates = [0.1, 0.001, 10, 100]


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


def predict(X, W_1, W_2):
    """Forward propagation through the network."""
    this.a_1 = X

    this.z_2 = np.dot(this.a_1, W_1)
    this.a_2 = sigmoid(this.z_2)

    this.z_3 = np.dot(this.a_2, W_2)
    this.a_3 = sigmoid(this.z_3)

    return this.a_3


def backprop(y_hat, W_1, W_2):
    """Backpropagation algorithm."""
    error_3 = np.subtract(y_hat, this.y)
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


def gd(learn_rates, epochs):
    this.w_1_rate_0 = np.random.uniform(0, 1, (3, 4))
    this.w_2_rate_0 = np.random.uniform(0, 1, (4, 1))

    this.alpha = []
    this.beta = []

    i = 0

    this.w_1_t = np.zeros((epochs, 3, 4))
    this.w_1_t[i] = this.w_1_rate_0

    this.w_2_t = np.zeros((epochs, 4, 1))
    this.w_2_t[i] = this.w_2_rate_0

    while i < epochs:
        this.fp_rate_0 = forward_pass(this.w_1_rate_0, this.w_2_rate_0)
        [dw_1_rate_0, dw_2_rate_0] = backprop(this.fp_rate_0, this.w_1_rate_0,
                                              this.w_2_rate_0)

        this.w_1_rate_0_new = learn_rates[0] * dw_1_rate_0
        this.w_1_t[i] = this.w_1_rate_0_new
        this.w_1_rate_0 -= this.w_1_rate_0_new

        this.w_2_rate_0_new = learn_rates[0] * dw_2_rate_0
        this.w_2_t[i] = this.w_2_rate_0_new
        this.w_2_rate_0 -= this.w_2_rate_0_new

        new_fp = forward_pass(this.w_1_rate_0, this.w_2_rate_0)
        loss_val = loss(new_fp, this.y)

        if i % 1000:
            print(loss_val)

        i += 1

    return [this.w_1_rate_0, this.w_2_rate_0]


def plot(W_1, W_2):
    alpha = []
    beta = []

    diff_u_w_1 = np.subtract(this.w_1_t, this.w_1_t[0])
    diff_u_w_2 = np.subtract(this.w_2_t, this.w_2_t[0])

    u_w_1_direction = np.subtract(W_1, this.w_1_t[0])
    u_w_2_direction = np.subtract(W_2, this.w_2_t[0])

    u_w_1 = u_w_1_direction / LA.norm(u_w_1_direction)
    u_w_2 = u_w_2_direction / LA.norm(u_w_2_direction)

    alpha_w_1 = np.einsum('ijk,km->ijm', np.swapaxes(diff_u_w_1, 1, 2), u_w_1)
    alpha_w_2 = np.einsum('ijk,km->ijm', np.swapaxes(diff_u_w_2, 1, 2), u_w_2)

    # alpha.append(LA.norm(alpha_w_1, axis=(1, 2)) + LA.norm(alpha_w_2, axis=(1, 2)))
    alpha.append(LA.norm(alpha_w_2, axis=(1, 2)))

    diff_v_w_1 = this.w_1_t - this.w_1_t[0] - np.einsum('km,ijm->ikm', u_w_1, alpha_w_1)
    diff_v_w_2 = this.w_2_t - this.w_2_t[0] - np.einsum('km,ijm->ikm', u_w_2, alpha_w_2)

    v_w_1 = diff_v_w_1 / LA.norm(diff_v_w_1)
    v_w_2 = diff_v_w_2 / LA.norm(diff_v_w_2)

    beta_w_1 = np.einsum('ijk,ikm->imj', diff_v_w_1, np.swapaxes(v_w_1, 1, 2))
    beta_w_2 = np.einsum('ijk,ikm->imj', diff_v_w_2, np.swapaxes(v_w_2, 1, 2))

    beta.append(LA.norm(beta_w_1, axis=(1, 2)) + LA.norm(beta_w_2, axis=(1, 2)))
    #beta.append(LA.norm(beta_w_1, axis=(1, 2)))

    plt.scatter(np.arange(0, 600), alpha)
    plt.show()


if __name__ == '__main__':
    [w_1, w_2] = gd([10], epochs=600)
    plot(w_1, w_2)
