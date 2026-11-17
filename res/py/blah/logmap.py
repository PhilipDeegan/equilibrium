# https://blbadger.github.io/logistic-map.html
import numpy as np
import equilibrium as eq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %matplotlib inline
plt.ioff()  # do not show image also
plt.style.use("dark_background")

x_min, x_max = [0.5, 5]
y_min, y_max = [-0.5, 3]

N = 4000
# N = 100000
Y = np.zeros(N + 1).astype("double")
X = np.zeros(N + 1).astype("double")
X[0], Y[0] = 1.0, 0.5


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
(p,) = ax.plot(X, Y, "^", color="white", alpha=0.8, markersize=0.513)


def animate(i):
    print("i", i)
    x = X[i]
    y = Y[i]
    i += 1
    y1 = y * x * (1 - y)
    print("y1", y1)
    Y[i] = y * x * (1 - y)
    X[i] = x + 0.001
    p.set_data(X[:i], Y[:i])
    return ax


eq.plot.Animation(fig=fig, func=animate, frames=N, interval=50).show()
