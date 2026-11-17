import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def sub_plot(fig, p, elev=30, azim=50, roll=0):
    fig.sub_plot_3d(p, elev, azim, roll)


def main():
    origin = np.array([0, 0, 0]).astype("double")
    sp = eq.math.geo.sphere.Sphere(origin, 1, 5, 3)
    fig = eq.plot.Fig3D(cols=2, rows=2, scale=3)
    p = sp.points()

    sub_plot(fig, p, 90, 0)
    sub_plot(fig, p, 00, 40, 0)
    sub_plot(fig, p, 00)
    sub_plot(fig, p, 20, 0)
    plt.show()


if __name__ == "__main__":
    main()
