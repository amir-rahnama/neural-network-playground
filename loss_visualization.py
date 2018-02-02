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
for k in range(3):
    new_weights = []
    random_w_1 = var*np.random.normal(size=(3, 4))
    random_w_2 = var*np.random.normal(size=(4, 1))

    scores = []
    t_ranges = np.linspace(-1, 1, num=100)

    for t in t_ranges:
        new_w_1 = this.initial_w1 + t * random_w_1
        new_w_2 = this.initial_w2 + t * random_w_2

        fp = mlp_xor.forward_pass(new_w_1, new_w_2)
        loss_value = mlp_xor.loss(fp, mlp_xor.y)
        scores.append(loss_value)

    ax.clear()
    ax.plot(t_ranges, scores)

plt.show()
