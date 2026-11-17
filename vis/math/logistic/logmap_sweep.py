import argparse

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def animation(frames=100, width=2000, r_min=1.0, r_max=4.0):
    eq.plot.style.use("dark_background")
    r = np.linspace(r_min, r_max, width)
    y = np.repeat(0.1, width)
    y[0] = 0.5
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set(xlim=(r_min, r_max), ylim=(-0.1, 1))
    (p,) = ax.plot(r, y, "^", color="white", alpha=0.4, markersize=0.113)

    def update(_):
        y[:] = y * r * (1 - y)
        p.set_data(r, y)
        return (p,)

    return eq.plot.Animation(fig=fig, func=update, frames=frames, interval=50)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="logistic map iterated in parallel across r"
    )
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args(argv)
    return animation(a.frames).show()


if __name__ == "__main__":
    main()
