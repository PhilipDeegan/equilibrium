import numpy as np

import equilibrium as eq


def main():
    # Define grid
    mid = 0
    C = 50
    dx, dy, _dz = 0.1, 0.1, 0.1
    L = C * dx
    mid = L / 2
    print("mid", mid)
    x = np.arange(0, L, 0.1)
    y = np.arange(0, L, 0.1)

    X, Y = np.meshgrid(x, y, indexing="ij")

    X -= mid + dx * 2
    Y -= mid + dy * 2

    R = np.sqrt(X**2 + Y**2)
    eps = 1e-8
    Ux = -X / (R + eps)
    Uy = -Y / (R + eps)

    wavelength1 = 0.1
    k1 = 2 * np.pi / wavelength1
    frequency1 = 1.0
    omega1 = 2 * np.pi * frequency1
    phi1 = k1 * R + omega1

    A = np.cos(phi1)
    U = A * Ux
    V = A * Uy

    # Compute direction toward center (0,0)
    # U = -X  # Negative x-component (points toward center)
    # V = -Y  # Negative y-component (points toward center)

    # # Normalize vectors to unit length
    # magnitude = np.sqrt(U**2 + V**2) + 1e-5  # Avoid division by zero

    # # Normalize vectors (unit length)
    # U /= magnitude
    # V /= magnitude

    U *= 0.1
    V *= 0.1

    # U[~mask] = 0
    # V[~mask] = 0

    # Plot
    fig = eq.plot.Fig(cols=1, rows=1, scale=11)

    def sub_plot(data):
        return fig.sub_plot(data)

    print(U[np.nonzero(U)])
    ax = sub_plot(None)
    ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=11)

    fig.show()


if __name__ == "__main__":
    main()
