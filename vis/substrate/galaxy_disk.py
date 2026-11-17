import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def main():
    global star_theta

    # 1. Simulation parameters
    num_stars = 400
    num_frames = 1000
    grid_size = 200

    # Create the underlying substrate map (X, Y coordinates)
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)

    # Calculate substrate bit-depth (L_local / L_max)
    # Smoothly transitions from compressed core to the rigid Void Wall (1.0) at the boundaries
    substrate_density = 1.0 - (0.8 / (1.0 + (R_grid / 2.5) ** 2))

    # 2. Initialize the periodic star loops
    # Randomly distribute stars across the galactic disk radius
    star_r = np.random.uniform(0.8, 8.5, num_stars)
    star_theta = np.random.uniform(0, 2 * np.pi, num_stars)

    # Apply the flat rotation curve (DST edge gradient acceleration)
    # Normal Newtonian drops off as 1/sqrt(r), but the Void Wall forces a flat velocity profile
    v_base = 0.4  # Maximum rotation speed
    star_v = v_base * (1.0 - np.exp(-star_r / 1.2))  # Smoothly ramps up and stays flat!
    star_omega = star_v / star_r  # Angular velocity

    # 3. Set up the plotting environment
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.set_facecolor("black")

    # Plot the underlying compressed substrate grid
    im = ax.imshow(
        substrate_density,
        cmap="plasma",
        extent=[-10, 10, -10, 10],
        origin="lower",
        alpha=0.6,
        vmin=0.2,
        vmax=1.0,
    )

    # Initialize the star scatter plot
    stars_plot = ax.scatter(
        [],
        [],
        color="white",
        s=3,
        alpha=0.8,
        edgecolors="none",
        # shadow=True if "shadow" in locals() else None,
    )

    # Add subtle UI indicators for the physicalist ledger
    ax.text(
        -9.5,
        9.0,
        "Substrate Base: 14-Neighbor Kelvin Lattice",
        color="#00ffcc",
        fontsize=10,
        fontfamily="monospace",
    )
    ax.text(
        -9.5,
        8.4,
        "Outer Boundary Floor: L_max = 1.0 (Integer Max)",
        color="#ffcc00",
        fontsize=10,
        fontfamily="monospace",
    )

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axis("off")

    # 5. Compile and save the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=30,
        blit=True,
        fargs=(stars_plot, im, star_r, star_omega),
    )

    # Save as a local GIF file to view the rotation
    ani.save("galaxy_disk.mp4", writer=animation.FFMpegWriter(fps=30))
    plt.close()


# 4. Animation update loop
def update(frame, stars_plot, im, star_r, star_omega):
    global star_theta
    # Advance the loop phases based on their local cell-gradient velocity
    star_theta += star_omega * 0.05

    # Reconstruct Cartesian positions from the advancing phase angles
    x_stars = star_r * np.cos(star_theta)
    y_stars = star_r * np.sin(star_theta)

    # Update scatter plot data
    stars_plot.set_offsets(np.c_[x_stars, y_stars])
    return stars_plot, im


if __name__ == "__main__":
    main()
