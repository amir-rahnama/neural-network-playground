"""2-d visualization of restricted loss."""
import matplotlib.pyplot as plt
import numpy as np
import sys
import mlp_xor

this = sys.modules[__name__]

this.initial_w1 = np.random.uniform(0, 1, (3, 4))
this.initial_w2 = np.random.uniform(0, 1, (4, 1))


fig = plt.figure()
ax = fig.add_subplot(111)

var = 1
alpha = 0.1

scores = mlp_xor.gd([0.1], 70000)
[optimal_w_1, optimal_w_2] = mlp_xor.get_weights()


for k in np.linspace(-1, 1, num=100):
    new_weights = []

    scores = []
    t_ranges = np.linspace(-1, 1, num=100)

    for t in t_ranges:
        new_w_1 = (1 - alpha) * this.initial_w1 + alpha * optimal_w_1
        new_w_2 = (1 - alpha) * this.initial_w2 + alpha * optimal_w_2

        fp = mlp_xor.forward_pass(new_w_1, new_w_2)
        loss_value = mlp_xor.loss(fp, mlp_xor.y)
        scores.append(loss_value)

    ax.clear()
    ax.plot(t_ranges, scores)

plt.show()
