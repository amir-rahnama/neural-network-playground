import matplotlib.pyplot as plt
import numpy as np
import sys
import mlp_xor

this = sys.modules[__name__]

this.initial_w1 = np.random.uniform(0, 1, (3, 4))
this.initial_w2 = np.random.uniform(0, 1, (4, 1))


def plot_decision_boundary():
    X = mlp_xor.X
    y = mlp_xor.y

    x_min = y_min = 0
    x_max = y_max = 1

    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    [this.optimal_w_1, this.optimal_w_2] = mlp_xor.gd([0.1], 70000)[1][0.1]

    # Predict the function value for the whole gid
    t = np.ones(xx.ravel().shape[0])
    x = np.c_[xx.ravel(), yy.ravel(), t]
    Z = mlp_xor.predict(x, this.optimal_w_1, this.optimal_w_2)

    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.Paired, c=y.ravel())
    plt.show()


if __name__ == '__main__':
    plot_decision_boundary()
