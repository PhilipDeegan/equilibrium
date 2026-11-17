import numpy as np
import equilibrium as eq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %matplotlib inline
plt.ioff()  # do not show image also
plt.style.use("dark_background")

x_min, x_max = [1, 4]
y_min, y_max = [1, 4]
N = 300
W = N + 1
X = np.repeat(1.0, W)  # np.linspace(x_min, x_max, W)
X[0] = 1
d = np.array([1, 4])
g = 9.8
t = 0.01


class E:
    def __init__(self):
        self.v = d.copy() + 1
        self.a = d.copy() + 4
        self.m = d.copy()
        self.X = X.copy()
        self.Y = X.copy()

        self.mov(0)

    def mov(self, i):
        v, a = self.v, self.a
        s = v * t - 0.5 * a * (t**2)
        # self.X[i], self.Y[i] = s
        self.X[i] = self.X[i - 1] + s[0]
        self.Y[i] = self.Y[i - 1] + s[1]

    def __call__(self, j):
        i = j + 1
        X = self.X
        Y = self.Y
        self.mov(i)
        # Y = self.Y
        # self.Y[i] = Y[j] * X[j] * (1 - Y[j])
        p.set_data(self.X, self.Y)
        return ax


e = E()
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
(p,) = ax.plot(e.X, e.Y, "^", color="white", alpha=0.4, markersize=0.313)
eq.plot.Animation(fig=fig, func=e, frames=N, interval=50).show()
