import argparse

import numpy as np

import equilibrium as eq

PI = np.longdouble("3.141592653589793238462643383279502884")


def initial():
    p = np.zeros([4, 3])
    p[0, 0], p[1, 0], p[2, 1], p[3, 1] = 1, -1, 1, -1
    v = np.zeros_like(p)
    v[:, 2] = PI / 100
    return p, v


def draw(axs, p, bound):
    for ax in axs:
        ax.cla()
        ax.invert_yaxis()
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound))
        ax.scatter(p[:, 0], p[:, 1], p[:, 2])


def animation(frames=200, views=1, bound=3):
    p, v = initial()
    if views == 4:
        fig = eq.plot.Fig3D(4, 1, 3)
        axs = fig.axes_view()
    else:
        fig = eq.plot.Fig3D(2, 2, 2)
        axs = [fig.sub_plot_3d(elev=30, azim=50, roll=0)]
    draw(axs, p, bound)

    def update(_):
        p[:] += np.cross(p, v)
        draw(axs, p, bound)

    return eq.plot.Animation(fig=fig, func=update, frames=frames, interval=50)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--views", type=int, choices=(1, 4), default=1)
    a = ap.parse_args(argv)
    return animation(a.frames, a.views).show()


if __name__ == "__main__":
    main()
