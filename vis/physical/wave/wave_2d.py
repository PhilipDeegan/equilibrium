import argparse

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def step(E, E_prev, c=1.0, dt=0.1, dx=1.0):
    lap = (
        np.roll(E, 1, 0)
        + np.roll(E, -1, 0)
        + np.roll(E, 1, 1)
        + np.roll(E, -1, 1)
        - 4 * E
    ) / dx**2
    return 2 * E - E_prev + (c * dt) ** 2 * lap


def run(n=100, steps=1000):
    E = np.zeros((n, n))
    E[n // 2, n // 2] = 1.0
    E_prev = np.zeros_like(E)
    frames = []
    for _ in range(steps):
        E, E_prev = step(E, E_prev), E
        frames.append(E.copy())
    return frames


def animation(frames):
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0], cmap="RdBu", vmin=-0.1, vmax=0.1, animated=True)

    def update(i):
        img.set_array(frames[i])
        return [img]

    return eq.plot.Animation(
        fig=fig, func=update, frames=len(frames), interval=50, blit=True
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--save", help="write mp4 instead of showing")
    a = ap.parse_args(argv)
    ani = animation(run(a.n, a.steps))
    if a.save:
        return ani.write_to_file(a.save, fps=10)
    return ani.show()


if __name__ == "__main__":
    main()
