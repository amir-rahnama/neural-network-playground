"""A simple MLP to solve XOR."""
import matplotlib.pyplot as plt
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


def gd(losses, epchos):
    this.loss_map = {}
    this.weight_map = {}

    for loss in losses:
        this.loss_map[loss] = []
        this.weight_map[rate] = []
        print('training with rate: ', rate)

        this.w_1 = np.random.uniform(0, 1, (3, 4))
        this.w_2 = np.random.uniform(0, 1, (4, 1))

        this.delta_w_1 = np.zeros((3, 4))
        this.delta_w_2 = np.zeros((4, 1))

        i = 0

        while i < epchos:
            this.fp = forward_pass(this.w_1, this.w_2)
            [dw_1, dw_2] = backprop(this.fp, this.w_1, this.w_2)

            this.w_1 -= rate * dw_1
            this.w_2 -= rate * dw_2

            this.weight_map[rate] = [this.w_1, this.w_2]

            loss_value = loss(this.fp, this.y)
            this.loss_map[rate].append(loss_value)

            i += 1

            if i % 10000 == 0:
                print(loss_value)
    return [this.loss_map, this.weight_map]


def get_weights():
    return this.weight_map


def plot():
    this.epochs = 70000
    this.learning_rates = [0.1, 0.01, 0.001, 0.0001, 10, 100]
    this.loss, this.weight = gd(this.learning_rates, this.epochs)

    training_epochs = np.arange(0, this.epochs)

    plt.plot(training_epochs, this.loss[0.1])
    plt.plot(training_epochs, this.loss[0.01])
    plt.plot(training_epochs, this.loss[0.001])
    plt.plot(training_epochs, this.loss[0.0001])
    plt.plot(training_epochs, this.loss[10])
    plt.plot(training_epochs, this.loss[100])

    plt.legend(['Learning rate = 0.1', 'Learning rate = 0.01',
                'Learning rate = 0.001',
                'Learning rate = 0.0001', 'Learning rate = 10',
                'Learning rate = 100'], loc='upper left')

    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.axis([1, this.epochs, 0, 1])

    plt.show()


if __name__ == '__main__':
    plot()
