import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def sub_plot(fig, sp, elev=30, azim=50, roll=0):
    fig.sub_plot_3d(sp.v, elev, azim, roll)


def main():
    origin = np.array([0, 0, 0]).astype("double")
    sp = eq.math.geo.sphere.Sphere(origin, 1, 5, 5)
    fig = eq.plot.Fig3D(cols=2, rows=2, scale=3)

    sub_plot(fig, sp, 90, 0)
    sub_plot(fig, sp, 00, 40, 0)
    sub_plot(fig, sp, 00)
    sub_plot(fig, sp, 20, 0)
    plt.show()


if __name__ == "__main__":
    main()
