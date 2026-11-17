# https://www.kaggle.com/code/miguelrodriguezolmos/plot-the-logistic-map-with-python-matplotlib
import argparse

import matplotlib.pyplot as plt
import numpy as np


def bifurcation(r_min=3.5, r_max=4.0, y0=0.01, iters=1000, keep=0.8, n=3000):
    r = np.linspace(r_min, r_max, n)
    y = np.full(n, y0)
    last = round(keep * iters)
    rs, ys = [], []
    for i in range(iters):
        y = r * y * (1 - y)
        if i >= iters - last:
            rs.append(r)
            ys.append(y)
    return np.concatenate(rs), np.concatenate(ys)


def figure(r_min=3.5, r_max=4.0, n=3000, **kw):
    r, y = bifurcation(r_min, r_max, n=n, **kw)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="steelblue")
    ax.set(xlim=(r_min, r_max), ylim=(0, 1))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    ax.scatter(r, y, s=(3 * 72 / n) ** 2, alpha=0.8, c="darkred", marker=".")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-min", type=float, default=3.5)
    ap.add_argument("--r-max", type=float, default=4.0)
    ap.add_argument("--save")
    a = ap.parse_args(argv)
    fig = figure(a.r_min, a.r_max)
    if a.save:
        fig.savefig(a.save, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
