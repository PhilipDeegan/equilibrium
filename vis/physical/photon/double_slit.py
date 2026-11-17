import argparse

import matplotlib.pyplot as plt
import numpy as np


def busy_wall(
    n_photons=1000, n_atoms=100, emission_interval=1e-6, processing_time=1e-4
):
    busy = np.zeros(n_atoms)
    hits = np.zeros(n_atoms)
    centre = n_atoms // 2
    order = [centre] + [
        i
        for d in range(1, n_atoms)
        for i in (centre - d, centre + d)
        if 0 <= i < n_atoms
    ]
    for _ in range(n_photons):
        free = next((i for i in order if busy[i] == 0), None)
        if free is not None:
            hits[free] += 1
            busy[free] = processing_time
        busy = np.maximum(busy - emission_interval, 0)
    return hits


def capacitance_walk(
    n_photons=5000,
    shape=(100, 200),
    slit_width=5,
    slit_spacing=10,
    capacity=3,
    steps=100,
    seed=None,
):
    rng = np.random.default_rng(seed)
    slits = np.ones(shape)
    c, top, bottom = shape[1] // 2, shape[0] // 3, 2 * shape[0] // 3
    slits[top : top + slit_width, c - slit_spacing] = 0
    slits[bottom : bottom + slit_width, c + slit_spacing] = 0
    photons = np.zeros((n_photons, 2), dtype=int)
    photons[:, 0] = shape[0] - 1
    photons[:, 1] = c + rng.integers(-10, 10, n_photons)
    counts = np.zeros(shape)
    for _ in range(steps):
        for i in range(n_photons):
            x, y = photons[i]
            if x > 0:
                nx, ny = x - 1, np.clip(y + rng.choice([-1, 0, 1]), 0, shape[1] - 1)
                if slits[nx, ny] == 0 or counts[nx, ny] < capacity:
                    photons[i] = nx, ny
                    counts[nx, ny] += 1
    return counts


def least_energy_path(
    n_photons=5000,
    size=100,
    slit_width=5,
    slit_spacing=10,
    slit_height=30,
    increment=1e-6,
):
    material = np.ones((size, size))
    sx, y0 = size // 2, (size - slit_height) // 2
    material[
        y0 : y0 + slit_height,
        sx - slit_spacing // 2 : sx - slit_spacing // 2 + slit_width,
    ] = 0
    material[
        y0 : y0 + slit_height,
        sx + slit_spacing // 2 - slit_width : sx + slit_spacing // 2,
    ] = 0
    material[:, -1] = 2
    energy = np.zeros((size, size))
    for _ in range(n_photons):
        x, y = 5, size // 2
        while x < size - 1:
            energy[y, x] += increment
            moves = [
                (x + 1, yy)
                for yy in (y, y - 1, y + 1)
                if 0 <= yy < size and material[yy, x + 1] != 1
            ]
            if not moves:
                break
            x, y = min(moves, key=lambda m: energy[m[1], m[0]])
    return energy


def interference(n=500, wavelength=50, slit_distance=100, screen_distance=400):
    x = np.linspace(-n // 2, n // 2, n)
    y = np.linspace(0, screen_distance, n)
    X, Y = np.meshgrid(x, y)
    d1 = np.hypot(X + slit_distance // 2, Y)
    d2 = np.hypot(X - slit_distance // 2, Y)
    return (
        np.cos(2 * np.pi * d1 / wavelength) + np.cos(2 * np.pi * d2 / wavelength)
    ) ** 2


def figure(mode, **kw):
    if mode == "wave":
        n = kw.get("n", 500)
        fig, ax = plt.subplots(figsize=(10, 6))
        extent = [-n // 2, n // 2, 0, kw.get("screen_distance", 400)]
        im = ax.imshow(interference(**kw), cmap="inferno", extent=extent, aspect="auto")
        fig.colorbar(im, ax=ax, label="Intensity")
        ax.set(
            title="Two-source wave interference (reference)",
            xlabel="Position",
            ylabel="Distance from slits",
        )
        return fig
    if mode == "wall":
        fig, ax = plt.subplots(figsize=(10, 5))
        hits = busy_wall(**kw)
        ax.bar(range(len(hits)), hits, color="blue")
        ax.set(
            xlabel="Back Wall Position",
            ylabel="Photon Hits",
            title="Photon Distribution on Back Wall",
        )
        return fig
    grid = capacitance_walk(**kw) if mode == "walk" else least_energy_path(**kw)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.colorbar(ax.imshow(grid, cmap="hot", origin="lower"), ax=ax)
    ax.set_title(
        {"walk": "Photon absorption counts", "path": "Least-energy photon paths"}[mode]
    )
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="finite-capacity double-slit toys")
    ap.add_argument("mode", choices=("wave", "wall", "walk", "path"))
    ap.add_argument("--n-photons", type=int)
    a = ap.parse_args(argv)
    figure(
        a.mode,
        **({"n_photons": a.n_photons} if a.n_photons and a.mode != "wave" else {})
    )
    plt.show()


if __name__ == "__main__":
    main()
