import matplotlib.pyplot as plt
import numpy as np


def main():
    # Constants
    v_sw = 400e3  # solar wind speed (m/s)
    omega = 2.7e-6  # solar rotation rate (rad/s)
    r0 = 6.96e8  # solar radius (m)

    # Parameters for grid
    r_max = 1.5e11  # 1 AU in meters
    n_lines = 20  # number of field lines
    n_points = 500  # resolution of each line

    # Generate theta and phi values
    theta = np.pi / 2  # Equatorial plane
    r = np.linspace(r0, r_max, n_points)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Generate multiple spiral lines with varying phi0
    for i in range(n_lines):
        dphi0 = i * 2 * np.pi / n_lines
        phi = omega * r / v_sw + dphi0
        # Convert spherical to Cartesian
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)  # remains constant in equator

        ax.plot(x, y, z, lw=1.5)

    # Sun as a red sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    xs = r0 * np.cos(u) * np.sin(v)
    ys = r0 * np.sin(u) * np.sin(v)
    zs = r0 * np.cos(v)
    ax.plot_surface(xs, ys, zs, color="orange", alpha=0.6)

    # Formatting
    ax.set_title("3D Parker Spiral Magnetic Field Lines", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 0.5])  # flatten Z
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
