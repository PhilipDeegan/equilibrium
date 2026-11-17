import numpy as np

import equilibrium as eq


def main():
    # Define grid
    x = np.linspace(-30, 30)  # Square grid (bigger than radius 25)
    y = np.linspace(-30, 30)
    X, Y = np.meshgrid(x, y)

    print(X.shape)

    # Compute direction toward center (0,0)
    U = -X  # Negative x-component (points toward center)
    V = -Y  # Negative y-component (points toward center)

    # Normalize vectors to unit length
    magnitude = np.sqrt(U**2 + V**2) + 1e-5  # Avoid division by zero

    # Normalize vectors (unit length)
    U /= magnitude
    V /= magnitude

    # Define circular mask (radius = 25)
    radius = 25
    mask = X**2 + Y**2 <= (radius + 2) ** 2

    # Apply mask (set vectors outside the circle to zero)
    U[~mask] = 0
    V[~mask] = 0

    # second mask
    mask = X**2 + Y**2 >= (radius - 2) ** 2
    U[~mask] = 0
    V[~mask] = 0

    # Plot
    fig = eq.plot.Fig(cols=1, rows=1, scale=11)

    def sub_plot(data):
        return fig.sub_plot(data)

    print(U)
    sub_plot(None).quiver(X, Y, U, V)

    fig.show()


if __name__ == "__main__":
    main()
