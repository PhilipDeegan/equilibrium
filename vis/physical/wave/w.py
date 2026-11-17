# https://beltoforion.de/en/recreational_mathematics/2d-wave-equation.php
import random

import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq

dimx = 300  # width of the simulation domain
dimy = 300  # height of the simulation domain


def update(u, alpha):
    u[2] = u[1]
    u[1] = u[0]

    # Version 2: Much faster by eliminating loops
    u[0, 1 : dimx - 1, 1 : dimy - 1] = (
        alpha[1 : dimx - 1, 1 : dimy - 1]
        * (
            u[1, 0 : dimx - 2, 1 : dimy - 1]
            + u[1, 2:dimx, 1 : dimy - 1]
            + u[1, 1 : dimx - 1, 0 : dimy - 2]
            + u[1, 1 : dimx - 1, 2:dimy]
            - 4 * u[1, 1 : dimx - 1, 1 : dimy - 1]
        )
        + 2 * u[1, 1 : dimx - 1, 1 : dimy - 1]
        - u[2, 1 : dimx - 1, 1 : dimy - 1]
    )

    # Not part of the wave equation but I need to remove energy from the system.
    # The boundary conditions are closed. Energy cannot leave and the simulation keeps adding energy.
    u[0, 1 : dimx - 1, 1 : dimy - 1] *= 0.995


def add_raindrop(u):
    x = random.randrange(5, dimx - 5)
    y = random.randrange(5, dimy - 5)
    u[0, x - 2 : x + 2, y - 2 : y + 2] = 120


def place_raindrops(u):
    if random.random() < 0.02:
        add_raindrop(u)


def clip(u, pixeldata):
    pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
    pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128, 0, 255)
    pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128, 0, 255)


def e(i, u, alpha, pixeldata, ax):
    place_raindrops(u)
    update(u, alpha)
    clip(u, pixeldata)
    ax.set_array(pixeldata)
    return [ax]


def main():
    h = 1  # spatial step width
    k = 1  # time step width
    N = 1000  # frames - ignored?

    u = np.zeros((3, dimx, dimy))  # The three dimensional simulation grid
    c = 0.5  # The "original" wave propagation speed
    alpha = np.zeros(
        (dimx, dimy)
    )  # wave propagation velocities of the entire simulation domain
    alpha[0:dimx, 0:dimy] = ((c * k) / h) ** 2  # will be set to a constant value of tau

    add_raindrop(u)
    update(u, alpha)
    pixeldata = np.zeros((dimx, dimy, 3), dtype=np.uint8)
    clip(u, pixeldata)
    fig = eq.plot.Fig(3, 3, 3)
    im = plt.imshow(pixeldata)
    eq.plot.Animation(
        fig=fig, func=e, fargs=(u, alpha, pixeldata, im), frames=N, interval=250
    ).write_to_file("wave.mp4")


if __name__ == "__main__":
    main()
