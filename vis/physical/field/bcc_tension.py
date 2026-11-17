import argparse

import matplotlib.pyplot as plt
import numpy as np


def bcc_nodes(size=6, scale=1.0):
    r = np.arange(size)
    corners = np.stack(np.meshgrid(r, r, r), axis=-1).reshape(-1, 3) * scale
    return np.vstack([corners, corners + scale / 2])


def tension(nodes, centre):
    d = nodes - centre
    r = np.linalg.norm(d, axis=1, keepdims=True)
    r[r == 0] = 0.1
    return -d / r**3


def figure(size=6, centre=(2.5, 2.5, 2.5)):
    nodes = bcc_nodes(size)
    v = tension(nodes, np.asarray(centre))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*nodes.T, color="blue", s=2, alpha=0.3)
    ax.quiver(*nodes.T, *v.T, length=0.4, normalize=True, color="red", alpha=0.6)
    ax.set(
        title="BCC nodes, 1/r² pull toward a central knot",
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=6)
    a = ap.parse_args(argv)
    figure(a.size, centre=((a.size - 1) / 2,) * 3)
    plt.show()


if __name__ == "__main__":
    main()
