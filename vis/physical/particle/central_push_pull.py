import argparse

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def particles(n=50, k_pull=500, seed=None):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(5.0, 10.0, n)
    pos = r[:, None] * np.stack([np.cos(a), np.sin(a)], axis=1)
    vel = (np.sqrt(k_pull / r) * 0.5)[:, None] * np.stack(
        [-np.sin(a), np.cos(a)], axis=1
    )
    return pos, vel


def step(pos, vel, dt=0.05, k_pull=500, k_push=10):
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    r_hat = np.divide(pos, r, out=np.zeros_like(pos), where=r > 0)
    f = -k_pull * r_hat / np.where(r > 0, r, 1) ** 2 + k_push * r_hat / np.where(
        r > 0, r, 1
    )
    vel += f * dt
    pos += vel * dt


def animation(n=50, steps=500, dt=0.05, k_pull=500, k_push=10, seed=None, lim=20):
    plt.style.use("dark_background")
    pos, vel = particles(n, k_pull, seed)
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=10)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")

    def update(_):
        step(pos, vel, dt, k_pull, k_push)
        scat.set_offsets(pos)
        return (scat,)

    return eq.plot.Animation(fig=fig, func=update, frames=steps, interval=50, blit=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--k-pull", type=float, default=500)
    ap.add_argument("--k-push", type=float, default=10)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    return animation(a.n, a.steps, k_pull=a.k_pull, k_push=a.k_push, seed=a.seed).show()


if __name__ == "__main__":
    main()
