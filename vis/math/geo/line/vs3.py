import numpy as np

import equilibrium as eq


def sub_plot(fig, data, elev=30, azim=50, roll=0):
    return fig.sub_plot_3d(data, elev, azim, roll)


def main():
    # Define grid
    x = np.linspace(-30, 30, 20)  # Square grid (bigger than radius 25)
    y = np.linspace(-30, 30, 20)
    z = np.linspace(-30, 30, 20)
    X, Y, Z = np.meshgrid(x, y, z)

    print(X.shape)

    # Compute direction toward center (0,0)
    U = -X  # Negative x-component (points toward center)
    V = -Y  # Negative y-component (points toward center)
    W = -Z

    # Normalize vectors to unit length
    magnitude = np.sqrt(U**2 + V**2 + W**2) + 1e-5  # Avoid division by zero

    # Normalize vectors (unit length)
    U /= magnitude
    V /= magnitude
    W /= magnitude

    # Define circular mask (radius = 25)
    radius = 25
    mask = X**2 + Y**2 + Z**2 <= (radius + 2) ** 2

    # Apply mask (set vectors outside the circle to zero)
    U[~mask] = 0
    V[~mask] = 0
    W[~mask] = 0

    # second mask
    mask = X**2 + Y**2 + Z**2 >= (radius - 2) ** 2
    U[~mask] = 0
    V[~mask] = 0
    W[~mask] = 0

    # Plot
    fig = eq.plot.Fig3D(cols=1, rows=1, scale=11)

    print(U)
    sub_plot(fig, None).quiver(X, Y, Z, U, V, W, length=1.5)

    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(projection="3d")
    # ax.quiver(X, Y, Z, U, V, W)  # Quiver plot (vector field)
    # ax.gca().add_patch(circle)
    # plt.xlim([-30, 30])
    # plt.ylim([-30, 30])
    # plt.ylim([-30, 30])
    # plt.gca().set_aspect("equal")  # Keep aspect ratio
    # plt.title("Vector Field with Circular Mask")
    fig.show()


if __name__ == "__main__":
    main()
