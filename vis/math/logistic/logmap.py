# https://blbadger.github.io/logistic-map.html
import argparse

import numpy as np

import equilibrium as eq


def animation(frames=300, r0=1.0, y0=0.2, dr=0.001, xlim=(0.9, 4), ylim=(-0.1, 4)):
    eq.plot.style.use("dark_background")
    R = np.zeros(frames + 1)
    Y = np.zeros(frames + 1)
    R[0], Y[0] = r0, y0
    fig = eq.plot.Fig()
    ax = fig.sub_plot(ax_set={"xlim": xlim, "ylim": ylim, "xlabel": "r", "ylabel": "y"})
    (p,) = ax.plot(R, Y, "^", color="white", alpha=0.4, markersize=0.113)

    def update(i):
        Y[i + 1] = Y[i] * R[i] * (1 - Y[i])
        R[i + 1] = R[i] + dr
        p.set_data(R, Y)
        return (p,)

    return eq.plot.Animation(
        fig=fig, init_func=lambda: (p,), func=update, frames=frames, interval=50
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args(argv)
    return animation(a.frames).show()


if __name__ == "__main__":
    main()
