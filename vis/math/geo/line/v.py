import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    np.set_printoptions(threshold=sys.maxsize, precision=4)

    # Define 3D grid between -1 and 1 with 21 points
    x = np.linspace(-0.2, 1.2, 15)
    y = np.linspace(-0.2, 1.2, 14)
    z = np.linspace(-0.2, 1.2, 14)
    X, Y, Z = np.meshgrid(x, y, z)

    print(X.shape)
    c = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T.reshape(
        (len(x), len(y), len(z), 3)
    )
    c -= 0.5
    # a = a * -1
    c = c.reshape([15 * 14 * 14, 3])
    a = c.copy()
    # X, Y, Z = a.reshape((len(x), len(y), len(z), 3))
    X = a[:, 0].reshape(len(x), len(y), len(z))
    Y = a[:, 1].reshape(len(x), len(y), len(z))
    Z = a[:, 2].reshape(len(x), len(y), len(z))

    # print(a[:, 0])
    # a[:, 0] = x
    # a[:, 1] = y
    # a[:, 2] = z

    # Compute vector components pointing toward the origin (0,0,0)
    U = X * -1  # Negative X direction
    V = Y * -1  # Negative Y direction
    W = Z * -1  # Negative Z direction

    # Compute magnitude for normalization
    magnitude = np.sqrt(U**2 + V**2 + W**2) + 1e-5  # Avoid division by zero

    # Normalize vectors (unit length)
    U /= magnitude
    V /= magnitude
    W /= magnitude

    radius = 0.5
    diff = 0.05
    mask = X**2 + Y**2 + Z**2 <= (radius + diff) ** 2

    # Apply mask (set vectors outside the circle to zero)
    U[~mask] = 0
    V[~mask] = 0
    W[~mask] = 0

    # second mask
    mask = X**2 + Y**2 + Z**2 >= (radius - diff) ** 2
    U[~mask] = 0
    V[~mask] = 0
    W[~mask] = 0

    # print("U", U)
    # U *= 0.00001
    # V *= 0.00001
    # W *= 0.00001
    # print("U", U)

    # c[:, 0] = U.reshape(15 * 14 * 14)
    # c[:, 1] = V.reshape(15 * 14 * 14)
    # c[:, 2] = W.reshape(15 * 14 * 14)
    # c = c * 0.001
    # print(c)

    # Plot 3D Vector Field
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Quiver plot (3D arrows)
    ax.quiver(X, Y, Z, U, V, W, length=0.05, normalize=True, color="b")

    # Set axis limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # Keep aspect ratio
    ax.set_title("3D Vector Field Pointing to Center")

    plt.show()


if __name__ == "__main__":
    main()
