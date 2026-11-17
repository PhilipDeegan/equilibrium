import argparse

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def coulomb(r, k=1.0, sigma=0.3):
    return 0 if r == 0 else -k * (sigma / r) ** 2


def lennard_jones(r, epsilon=0.01, sigma=0.3):
    return 0 if r == 0 else 24 * epsilon * (2 * (sigma / r) ** 13 - (sigma / r) ** 7)


FORCES = {"coulomb": coulomb, "lj": lennard_jones}


def forces(pos, pair=coulomb, box=None, wall_k=2.0, sigma=0.3):
    f = np.zeros_like(pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            r = pos[j] - pos[i]
            d = np.linalg.norm(r)
            if d > 0:
                fv = pair(d) / d * r
                f[i] += fv
                f[j] -= fv
    if box is not None:
        for i, j in np.ndindex(pos.shape):
            f[i, j] -= coulomb(pos[i, j], wall_k, sigma)
            f[i, j] += coulomb(box - pos[i, j], wall_k, sigma)
    return f


def initial(n, box, seed=42, pair_gap=None):
    if pair_gap is not None:
        mid = box / 2
        return np.array([[mid - pair_gap, mid], [mid + pair_gap, mid]]), np.zeros(
            (2, 2)
        )
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2) * box, (rng.rand(n, 2) - 0.5) * 2


def animation(
    n=14,
    force="coulomb",
    box=2.0,
    walls=True,
    frames=1500,
    dt=0.01,
    seed=42,
    pair_gap=None,
):
    pos, vel = initial(n, box, seed, pair_gap)
    pair = FORCES[force]
    fig, ax = plt.subplots()
    ax.set(xlim=(0, box), ylim=(0, box))
    (dots,) = ax.plot([], [], "bo", markersize=10)

    def update(_):
        vel[:] += forces(pos, pair, box if walls else None) * dt
        pos[:] += vel * dt
        dots.set_data(pos[:, 0], pos[:, 1])
        return (dots,)

    return eq.plot.Animation(
        fig=fig, func=update, frames=frames, interval=10, blit=True
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--force", choices=FORCES, default="coulomb")
    ap.add_argument("--box", type=float, default=2.0)
    ap.add_argument("--no-walls", action="store_true")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pair-gap", type=float, help="two particles at rest, +-gap from centre"
    )
    a = ap.parse_args(argv)
    return animation(
        a.n, a.force, a.box, not a.no_walls, a.frames, seed=a.seed, pair_gap=a.pair_gap
    ).show()


if __name__ == "__main__":
    main()
