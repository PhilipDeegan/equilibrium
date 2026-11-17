import argparse

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def gaussian(n=100, extent=3):
    s = np.linspace(-extent, extent, n)
    x, y = np.meshgrid(s, s)
    return np.exp(-(x**2 + y**2))


def laplacian(phi, dx=1.0):
    return (
        np.roll(phi, 1, 0)
        + np.roll(phi, -1, 0)
        + np.roll(phi, 1, 1)
        + np.roll(phi, -1, 1)
        - 4 * phi
    ) / dx**2


def dV(phi, lam=1.0, a=1.0):
    return 4 * lam * phi * (phi**2 - a**2)


def energy(phi, pi, dx=1.0, lam=1.0, a=1.0):
    gx = (np.roll(phi, -1, 0) - phi) / dx
    gy = (np.roll(phi, -1, 1) - phi) / dx
    return (
        0.5 * np.sum(pi**2)
        + 0.5 * np.sum(gx**2 + gy**2)
        + np.sum(lam * (phi**2 - a**2) ** 2)
    )


def snap(phi, threshold, a=1.0):
    phi[phi > threshold] = a
    phi[phi < -threshold] = -a
    phi[np.abs(phi) <= threshold] = 0


def step(phi, pi, dt=0.01, threshold=None, a=1.0):
    pi += dt * (laplacian(phi) - dV(phi, a=a))
    phi += dt * pi
    if threshold is not None:
        snap(phi, threshold, a)


def run(steps=500, threshold=None, n=100):
    phi = gaussian(n)
    pi = np.zeros_like(phi)
    history = []
    for _ in range(steps):
        step(phi, pi, threshold=threshold)
        history.append(energy(phi, pi))
    return phi, history


def figure(phi, history, threshold=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    kw = (
        {"cmap": "bwr", "vmin": -1, "vmax": 1}
        if threshold is not None
        else {"cmap": "coolwarm"}
    )
    fig.colorbar(a.imshow(phi, origin="lower", **kw), ax=a, label="Field φ")
    a.set_title(
        "Final field" + (f" (snap {threshold})" if threshold is not None else "")
    )
    a.axis("off")
    b.plot(history)
    b.set(xlabel="Step", ylabel="Total Energy", title="Energy over time")
    b.grid(True)
    return fig


def animation(frames=111, threshold=None, n=100):
    phi = gaussian(n)
    pi = np.zeros_like(phi)
    fig = eq.plot.Fig(cols=1, rows=1, scale=5)
    ax = fig.fig.gca()
    im = ax.imshow(phi, cmap="coolwarm", origin="lower")
    fig.fig.colorbar(im, ax=ax, label="Field φ")

    def update(_):
        step(phi, pi, threshold=threshold)
        im.set_data(phi)
        im.autoscale()
        return (im,)

    return eq.plot.Animation(fig=fig, func=update, frames=frames, interval=155)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="2D phi^4 field, optional threshold snapping"
    )
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument(
        "--threshold", type=float, help="snap |phi|>t to +-a, else 0 (e.g. 0.2)"
    )
    ap.add_argument("--animate", action="store_true")
    a = ap.parse_args(argv)
    if a.animate:
        return animation(a.steps, a.threshold).show()
    phi, history = run(a.steps, a.threshold)
    figure(phi, history, a.threshold)
    plt.show()


if __name__ == "__main__":
    main()
