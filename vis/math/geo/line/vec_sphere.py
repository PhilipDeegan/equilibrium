import matplotlib.pyplot as plt
import numpy as np

# #

# import sys
# import numpy as np
# import equilibrium as eq
# import matplotlib.pyplot as plt


# np.set_printoptions(threshold=sys.maxsize, precision=4)

# origin = np.array([0, 0, 0]).astype("double")
# fig = eq.plot.Fig3D(cols=2, rows=2, scale=3)
# V = eq.math.geo.line.Vector

# sp = eq.math.geo.sphere.Sphere(origin, 1, 3, 1)
# fig = eq.plot.Fig3D(cols=2, rows=2, scale=3)

# ps = sp.points()[0]


# def sub_plot(data, elev=30, azim=50, roll=0):
#     fig.sub_plot_3d(data, elev, azim, roll)


# r = 0.4
# # print(-1 / (x + 0.1))
# #
# # a = np.linspace((-0.65, -0.65, -0.65), (0.65, 0.65, 0.65), 11)
# # i = a.copy()

# # i[np.where((a > 0) & (a < r))] /= 0.00001
# # i[np.where((a < 0) & (a > -r))] /= 0.00001

# # i[np.where(a > (r * 2))] = 0
# # i[np.where(a < -r * 2)] = 0

# # i = 1 / i
# # i[np.where(i == np.inf)] = 0

# x = np.linspace(-0.65, 0.65, 5)
# y = np.linspace(-0.65, 0.65, 5)
# z = np.linspace(-0.65, 0.65, 15)

# # X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# X, Y = np.meshgrid(x, y, indexing="ij")

# # print(X, Y)
# a = np.array([X.flatten(), Y.flatten()]).T.reshape(len(x), len(y), 2)

# # print(a)
# for x in a:
#     for y in x:
#         print(y, end=" ")
#     print()

# # a = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T.reshape(
# #     (len(x), len(y), len(z), 3)
# # )
# # print(a.shape)
# # a = a.reshape([5**3, 3])

# # s = np.array((0.65, 0.65, 0.65))
# # print(a[-1][-1][-1])
# # l = a[np.equal(a[:, [0, 1]], 1).all(axis=1)]
# # print(l)
# # print(a[:, 0])


# # i = a.copy()
# # i[np.where((a > 0) & (a < r))] = 0
# # i[np.where((a < 0) & (a > -r))] = 0

# # i[np.where(a > (r * 2))] = 0
# # i[np.where(a < -r * 2)] = 0

# # i = 0.001 / i
# # i[np.where(i == np.inf)] = 0
# # a = i.copy()

# # a = a.reshape([5**3, 3])
# # vecs = [V(e) for e in a]

# # print(a)
# # sub_plot(vecs)
# # plt.show()


def main():
    # Define grid
    x = np.linspace(-30, 30, 50)  # Square grid (bigger than radius 25)
    y = np.linspace(-30, 30, 50)
    X, Y = np.meshgrid(x, y)

    # # Define vector field (example: radial field)
    # U = Y  # Example X-component (e.g., circular field)
    # # U = 1 / Y
    # V = -X  # Example Y-component
    # # V = 1 / X

    # Compute direction toward center (0,0)
    U = -X  # Negative x-component (points toward center)
    V = -Y  # Negative y-component (points toward center)

    # Normalize vectors to unit length
    magnitude = np.sqrt(U**2 + V**2)
    U = U / (magnitude + 1e-5)  # Avoid division by zero
    V = V / (magnitude + 1e-5)

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

    # V = 1 / V
    # U = 1 / U

    # Plot
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V)  # Quiver plot (vector field)
    circle = plt.Circle(
        (0, 0), radius, color="r", fill=False, linewidth=2
    )  # Circle outline
    plt.gca().add_patch(circle)
    plt.xlim([-30, 30])
    plt.ylim([-30, 30])
    plt.gca().set_aspect("equal")  # Keep aspect ratio
    plt.title("Vector Field with Circular Mask")
    plt.show()


if __name__ == "__main__":
    main()
