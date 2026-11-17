import argparse

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def helix(t, radius=1e-9, pitch=1e-9, angular_velocity=2 * np.pi * 1e12, axis="x"):
    theta = angular_velocity * t
    along = pitch / (2 * np.pi) * theta
    if axis == "z":
        return radius * np.cos(theta), radius * np.sin(theta), along
    return along, radius * np.sin(theta), radius * np.cos(theta)


def helix_figure(axis="x"):
    x, y, z = helix(np.linspace(0, 1e-12, 1000), axis=axis)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set(title="Photon Spiral Around Origin", xlabel="X", ylabel="Y", zlabel="Z")
    return fig


def restoring_field_figure(wavelength=1e-7, turns=5, radius=0.5e-8, n=1000):
    t = np.linspace(0, 2 * np.pi * turns, n)
    x = np.linspace(0, wavelength, n)
    X, Y, Z = np.meshgrid(
        np.linspace(0, wavelength, 10),
        np.linspace(-2 * radius, 2 * radius, 10),
        np.linspace(-2 * radius, 2 * radius, 10),
    )
    rho = np.sqrt(Y**2 + Z**2) + 1e-12
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(
        X,
        Y,
        Z,
        0 * X,
        -Y / rho,
        -Z / rho,
        length=1e-8,
        normalize=True,
        color="lightblue",
        alpha=0.6,
    )
    ax.plot(
        x,
        radius * np.cos(t),
        radius * np.sin(t),
        color="red",
        linewidth=2,
        label="Photon Spiral Path",
    )
    ax.set(
        title="Photon Spiral Propagation through Vector Field",
        xlabel="X (propagation)",
        ylabel="Y",
        zlabel="Z",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def oscillating_field_animation(frames=100):
    s = np.linspace(-2, 2, 5)
    x, y, z = np.meshgrid(s, s, s)
    vx = np.ones_like(x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    q = [ax.quiver(x, y, z, vx, -0.2 * y, -0.2 * z, length=0.3, normalize=True)]
    ax.set(
        xlim=(-2, 2),
        ylim=(-2, 2),
        zlim=(-2, 2),
        title="Restoring Vector Field Animation",
    )

    def update(i):
        phase = np.sin(2 * np.pi * i / frames)
        q[0].remove()
        q[0] = ax.quiver(
            x,
            y,
            z,
            vx,
            0.2 * y * np.cos(phase),
            0.2 * z * np.sin(phase),
            length=0.3,
            normalize=True,
        )
        return q

    return eq.plot.Animation(fig=fig, func=update, frames=frames, interval=50)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode", choices=("helix", "field", "animate"), nargs="?", default="field"
    )
    ap.add_argument("--axis", choices=("x", "z"), default="x")
    a = ap.parse_args(argv)
    if a.mode == "animate":
        return oscillating_field_animation().show()
    helix_figure(a.axis) if a.mode == "helix" else restoring_field_figure()
    plt.show()


if __name__ == "__main__":
    main()
