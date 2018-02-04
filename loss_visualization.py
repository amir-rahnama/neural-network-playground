"""2-d visualization of restricted loss."""
import matplotlib.pyplot as plt
import numpy as np
import sys
import mlp_xor


this = sys.modules[__name__]

this.initial_w1 = np.random.uniform(0, 1, (3, 4))
this.initial_w2 = np.random.uniform(0, 1, (4, 1))


var = 1
alpha = 0.1

plt.ion()

fig = plt.figure()

ax = fig.add_subplot(111)

t_ranges = np.linspace(-1, 1, num=100)
line, = ax.plot(t_ranges)

for k in range(1, 1000):
    new_weights = []
    random_w_1 = var*np.random.normal(size=(3, 4))
    random_w_2 = var*np.random.normal(size=(4, 1))

    this.scores = []

    for t in t_ranges:
        new_w_1 = this.initial_w1 + t * random_w_1
        new_w_2 = this.initial_w2 + t * random_w_2

        fp = mlp_xor.forward_pass(new_w_1, new_w_2)
        loss_value = mlp_xor.loss(fp, mlp_xor.y)

        this.scores.append(loss_value)

    line.set_ydata(this.scores)
    # res, = plt.plot(t_ranges, scores)
    fig.canvas.draw()
