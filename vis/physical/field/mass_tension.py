import argparse

import matplotlib.pyplot as plt
import numpy as np

TRIANGLE = [(-3, 0, 5), (3, 0, 5), (0, 4, 5)]
CROSS = [(-6, 0, 5), (6, 0, 5), (0, 6, 5), (0, -6, 5)]


def field(masses, extent=10, n=200, soften=0.1):
    s = np.linspace(-extent, extent, n)
    x, y = np.meshgrid(s, s)
    f = np.zeros_like(x)
    for mx, my, m in masses:
        f += m / (np.hypot(x - mx, y - my) + soften)
    return x, y, f


def tension_panel(ax, x, y, f, masses=None, every=10):
    gy, gx = np.gradient(-f)
    cs = ax.contourf(x, y, f, levels=100, cmap="viridis")
    ax.figure.colorbar(cs, ax=ax, label="Field Tension Magnitude")
    s = slice(None, None, every)
    ax.quiver(x[s, s], y[s, s], gx[s, s], gy[s, s], color="white")
    for mx, my, m in masses or []:
        ax.plot(mx, my, "ro")
        ax.text(mx + 0.3, my + 0.3, f"m={m}", color="white")
    ax.set(title="Dynamic Field Tension", xlabel="x", ylabel="y")


def energy_panel(ax, x, y, f):
    gy, gx = np.gradient(-f)
    cs = ax.contourf(x, y, np.hypot(gx, gy) * f, levels=100, cmap="plasma")
    ax.figure.colorbar(cs, ax=ax, label="Inductive Energy Density")
    ax.set(title="Inductive Interaction Density", xlabel="x", ylabel="y")


def figure(masses=TRIANGLE, energy=True, label_masses=False):
    x, y, f = field(masses)
    if energy:
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
        tension_panel(a, x, y, f, masses if label_masses else None)
        energy_panel(b, x, y, f)
    else:
        fig, a = plt.subplots(figsize=(10, 8))
        tension_panel(a, x, y, f, masses if label_masses else None)
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("triangle", "cross"), default="triangle")
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--label-masses", action="store_true")
    a = ap.parse_args(argv)
    plt.style.use("dark_background")
    masses = TRIANGLE if a.layout == "triangle" else CROSS
    figure(masses, not a.no_energy, a.label_masses)
    plt.show()


if __name__ == "__main__":
    main()
