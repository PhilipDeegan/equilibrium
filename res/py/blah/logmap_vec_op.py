import numpy as np
import equilibrium as eq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %matplotlib inline
plt.ioff()  # do not show image also
plt.style.use("dark_background")

x_min, x_max = [1, 4]
y_min, y_max = [-0.1, 1]
N = 300
W = N + 1
N = 100
W = 2000
X = np.linspace(x_min, x_max, W)


# array based
class E:
    def __init__(self):
        self.Y = np.repeat(0.1, W)
        self.Y[0] = 0.5

    def __call__(self, i):
        Y = self.Y
        self.Y = Y * X * (1 - Y)
        p.set_data(X, Y)
        return ax


# index based
class Ei:
    def __init__(self):
        self.Y = np.repeat(-10.0, W)  # outside plot
        self.Y[0] = 0.5

    def __call__(self, j):
        i = j + 1
        Y = self.Y
        self.Y[i] = Y[j] * X[j] * (1 - Y[j])
        p.set_data(X, Y)
        return ax


e = E()
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
(p,) = ax.plot(X, e.Y, "^", color="white", alpha=0.4, markersize=0.113)
eq.plot.Animation(fig=fig, func=e, frames=N, interval=50).show()
