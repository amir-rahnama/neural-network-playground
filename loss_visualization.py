"""2-d visualization of restricted loss."""
import matplotlib.pyplot as plt
import numpy as np
import sys
import mlp_xor

this = sys.modules[__name__]

this.initial_w1 = np.random.uniform(0, 1, (3, 4))
this.initial_w2 = np.random.uniform(0, 1, (4, 1))


fig, ax = plt.subplots()
ax = fig.add_subplot(111)

var = 1
alpha = 0.1

t_ranges = np.linspace(-1, 1, num=100)
line, = plt.plot(t_ranges)
plt.ion()
plt.show()


for k in range(1, 100):
    new_weights = []
    random_w_1 = var*np.random.normal(size=(3, 4))
    random_w_2 = var*np.random.normal(size=(4, 1))

    scores = []
    # t_ranges = np.linspace(-1, 1, num=100)

    for t in t_ranges:
        new_w_1 = this.initial_w1 + t * random_w_1
        new_w_2 = this.initial_w2 + t * random_w_2

        fp = mlp_xor.forward_pass(new_w_1, new_w_2)
        loss_value = mlp_xor.loss(fp, mlp_xor.y)

        this.scores.append(loss_value)

    ax.clear()
    line.set_ydata(scores)
    plt.draw()
    time.sleep(0.1)
    plt.pause(0.0001)
    # ax.plot(t_ranges, scores)

plt.show()


def animate(scores):
    line.set_ydata(scores)  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init(scores):
    line.set_ydata(scores)
    return line,


ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True)
plt.show()
