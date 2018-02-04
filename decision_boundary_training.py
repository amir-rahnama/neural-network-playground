"""A simple MLP to solve XOR."""
import matplotlib.pyplot as plt
import numpy as np
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

this.learning_rates = [0.1, 0.001, 10, 100]

plt.ion()

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(18.5, 10.5)

mng = plt.get_current_fig_manager()


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
    this.loss_map = {}
    this.weight_map = {}

    this.w_1_rate_0 = np.random.uniform(0, 1, (3, 4))
    this.w_2_rate_0 = np.random.uniform(0, 1, (4, 1))

    this.w_1_rate_1 = np.random.uniform(0, 1, (3, 4))
    this.w_2_rate_1 = np.random.uniform(0, 1, (4, 1))

    this.w_1_rate_2 = np.random.uniform(0, 1, (3, 4))
    this.w_2_rate_2 = np.random.uniform(0, 1, (4, 1))

    this.w_1_rate_3 = np.random.uniform(0, 1, (3, 4))
    this.w_2_rate_3 = np.random.uniform(0, 1, (4, 1))

    i = 0

    while i < epochs:
        this.fp_rate_0 = forward_pass(this.w_1_rate_0, this.w_2_rate_0)
        [dw_1_rate_0, dw_2_rate_0] = backprop(this.fp_rate_0, this.w_1_rate_0, this.w_2_rate_0)
        this.w_1_rate_0 -= learn_rates[0] * dw_1_rate_0
        this.w_2_rate_0 -= learn_rates[0] * dw_2_rate_0

        this.fp_rate_1 = forward_pass(this.w_1_rate_1, this.w_2_rate_1)
        [dw_1_rate_1, dw_2_rate_1] = backprop(this.fp_rate_1, this.w_1_rate_1, this.w_2_rate_1)
        this.w_1_rate_1 -= learn_rates[1] * dw_1_rate_1
        this.w_2_rate_1 -= learn_rates[1] * dw_2_rate_1

        this.fp_rate_2 = forward_pass(this.w_1_rate_2, this.w_2_rate_2)
        [dw_1_rate_2, dw_2_rate_2] = backprop(this.fp_rate_2, this.w_1_rate_2, this.w_2_rate_2)
        this.w_1_rate_2 -= learn_rates[2] * dw_1_rate_2
        this.w_2_rate_2 -= learn_rates[2] * dw_2_rate_2

        this.fp_rate_3 = forward_pass(this.w_1_rate_3, this.w_2_rate_3)
        [dw_1_rate_3, dw_2_rate_3] = backprop(this.fp_rate_3, this.w_1_rate_3, this.w_2_rate_3)
        this.w_1_rate_3 -= learn_rates[3] * dw_1_rate_3
        this.w_2_rate_3 -= learn_rates[3] * dw_2_rate_3

        i += 1

        if i % 500 == 0:
            plot_decision_boundary(learn_rates[0], this.w_1_rate_0,
                                   this.w_2_rate_0)
            plot_decision_boundary(learn_rates[1], this.w_1_rate_1,
                                   this.w_2_rate_1)
            plot_decision_boundary(learn_rates[2], this.w_1_rate_2,
                                   this.w_2_rate_2)
            plot_decision_boundary(learn_rates[3], this.w_1_rate_3,
                                   this.w_2_rate_3)


def plot_decision_boundary(learn_rate, w_1, w_2):
    index = this.learning_rates.index(learn_rate)

    if index == 0:
        i = 0
        j = 0
        title = 'Learning rate: 0.1'
    elif index == 1:
        i = 0
        j = 1
        title = 'Learning rate: 0.001'
    elif index == 2:
        i = 1
        j = 0
        title = 'Learning rate: 10'
    else:
        i = 1
        j = 1
        title = 'Learning rate: 100'

    x_min = y_min = 0
    x_max = y_max = 1

    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    t = np.ones(xx.ravel().shape[0])
    x = np.c_[xx.ravel(), yy.ravel(), t]
    Z = predict(x, w_1, w_2)

    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    ax[i, j].contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax[i, j].scatter(this.X[:, 0], this.X[:, 1], cmap=plt.cm.Paired,
                     c=this.y.ravel())
    ax[i, j].set_title(title)

    fig.canvas.draw()


if __name__ == '__main__':
    gd(this.learning_rates, epochs=70000)
