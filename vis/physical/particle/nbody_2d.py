import argparse

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def ring(n=3, radius=10.0, speed=1.5):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = radius * np.stack([np.cos(a), np.sin(a)], axis=1)
    vel = speed * np.stack([-np.sin(a), np.cos(a)], axis=1)
    return pos, vel


def forces(pos, external_pressure=0.0):
    f = np.zeros_like(pos)
    for i in range(len(pos)):
        for j in range(len(pos)):
            r = pos[i] - pos[j]
            d = np.linalg.norm(r)
            if i != j and d:
                f[i] -= r / d**2
        d = np.linalg.norm(pos[i])
        if external_pressure and d:
            f[i] -= external_pressure * pos[i] / d
    return f


def step(pos, vel, dt=0.05, external_pressure=0.0):
    for i in range(len(pos)):
        vel[i] += forces(pos, external_pressure)[i] * dt
        pos[i] += vel[i] * dt


def animation(n=3, steps=500, dt=0.05, external_pressure=0.0, lim=15):
    plt.style.use("dark_background")
    pos, vel = ring(n)
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=50, c="red")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")

    def update(_):
        step(pos, vel, dt, external_pressure)
        scat.set_offsets(pos)
        return (scat,)

    return eq.plot.Animation(fig=fig, func=update, frames=steps, interval=50, blit=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--external-pressure", type=float, default=0.0)
    a = ap.parse_args(argv)
    return animation(a.n, a.steps, external_pressure=a.external_pressure).show()


if __name__ == "__main__":
    main()
