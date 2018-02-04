"""2-d visualization of restricted loss."""
import matplotlib.pyplot as plt
import numpy as np
import sys
import mlp_xor

this = sys.modules[__name__]

this.initial_w1 = np.random.uniform(0, 1, (3, 4))
this.initial_w2 = np.random.uniform(0, 1, (4, 1))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(left=0.2, wspace=0.6)


scores = mlp_xor.gd([0.1], 70000)
[this.optimal_w_1, this.optimal_w_2] = mlp_xor.get_weights()[0.1]

alphas = []
scores = []

for alpha in np.linspace(-2, 2, num=200):
    alphas.append(alpha)

    new_w_1 = (1 - alpha) * this.initial_w1 + alpha * this.optimal_w_1
    new_w_2 = (1 - alpha) * this.initial_w2 + alpha * this.optimal_w_2

    fp = mlp_xor.forward_pass(new_w_1, new_w_2)
    loss_value = mlp_xor.loss(fp, mlp_xor.y)

    scores.append(loss_value)

ax1.plot(alphas, scores)

alphas = []
scores = []

for alpha in np.linspace(0, 1, num=200):
    alphas.append(alpha)

    new_w_1 = (1 - alpha) * this.initial_w1 + alpha * this.optimal_w_1
    new_w_2 = (1 - alpha) * this.initial_w2 + alpha * this.optimal_w_2

    fp = mlp_xor.forward_pass(new_w_1, new_w_2)
    loss_value = mlp_xor.loss(fp, mlp_xor.y)

    scores.append(loss_value)

ax2.plot(alphas, scores)

alphas = []
scores = []

for alpha in np.linspace(0, 0.1, num=200):
    alphas.append(alpha)

    new_w_1 = (1 - alpha) * this.initial_w1 + alpha * this.optimal_w_1
    new_w_2 = (1 - alpha) * this.initial_w2 + alpha * this.optimal_w_2

    fp = mlp_xor.forward_pass(new_w_1, new_w_2)
    loss_value = mlp_xor.loss(fp, mlp_xor.y)

    scores.append(loss_value)

ax3.plot(alphas, scores)

alphas = []
scores = []

for alpha in np.linspace(.99, 1, num=200):
    alphas.append(alpha)

    new_w_1 = (1 - alpha) * this.initial_w1 + alpha * this.optimal_w_1
    new_w_2 = (1 - alpha) * this.initial_w2 + alpha * this.optimal_w_2

    fp = mlp_xor.forward_pass(new_w_1, new_w_2)
    loss_value = mlp_xor.loss(fp, mlp_xor.y)

    scores.append(loss_value)


ax4.plot(alphas, scores)
plt.show()
