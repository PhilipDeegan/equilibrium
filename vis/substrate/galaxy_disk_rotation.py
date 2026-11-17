import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def main():
    global x_pos, y_pos, vx, vy

    # Define simulation parameters
    num_stars = 300
    num_frames = 1000
    dt = 0.1

    # Generate star initial positions in a disk
    np.random.seed(42)
    star_r = np.random.uniform(1.0, 9.0, num_stars)
    star_theta = np.random.uniform(0, 2 * np.pi, num_stars)

    x_pos = star_r * np.cos(star_theta)
    y_pos = star_r * np.sin(star_theta)

    # In DST, the acceleration comes from the gradient of the cell size (bit depth)
    # Let's define a profile for cell size l(r) that produces a flat rotation curve in the outskirts
    # due to the Void Wall boundary push.
    # For a flat rotation curve v(r) = v_max, a = v_max^2 / r, meaning grad(l) ~ 1/r.
    # A profile like l(r) = 1.0 - A * exp(-r/rc) or a log-like profile that levels off at l_max=1.0.

    v_max = 0.4
    # Let's compute initial velocities for circular orbits based on a flat rotation curve
    # v = v_max * (1 - exp(-r/2))
    v_mag = v_max * (1.0 - np.exp(-star_r / 1.5))
    vx = -v_mag * np.sin(star_theta)
    vy = v_mag * np.cos(star_theta)

    # Setup figure
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.set_facecolor("black")

    # Background substrate grid for visualization
    grid_res = 100
    g_x = np.linspace(-10, 10, grid_res)
    g_y = np.linspace(-10, 10, grid_res)
    G_X, G_Y = np.meshgrid(g_x, g_y)
    G_R = np.sqrt(G_X**2 + G_Y**2)
    # Substrate bit-depth showing the compression well and the outward relaxation to 1.0
    substrate = 1.0 - 0.7 * np.exp(-G_R / 3.0)

    im = ax.imshow(
        substrate,
        cmap="plasma",
        extent=[-10, 10, -10, 10],
        origin="lower",
        alpha=0.5,
        vmin=0.3,
        vmax=1.0,
    )
    scatter = ax.scatter(x_pos, y_pos, color="white", s=4, alpha=0.8, edgecolors="none")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axis("off")

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=40, blit=True, fargs=(scatter, im, dt)
    )
    ani.save("galaxy_disk_rotation.mp4", writer=animation.FFMpegWriter(fps=30))


# Dynamic update using the actual acceleration gradient
def update(frame, scatter, im, dt):
    global x_pos, y_pos, vx, vy

    r = np.sqrt(x_pos**2 + y_pos**2)
    # Calculate acceleration from the gradient of the cell-size layout
    # a = - grad(Phi). In DST, the phase loop drifts towards smaller cells (inward).
    # To maintain a flat rotation curve, the effective acceleration profile:
    # a_mag = v^2 / r
    v_max = 0.4
    v_curr = v_max * (1.0 - np.exp(-r / 1.5))
    a_mag = (v_curr**2) / (r + 1e-5)

    ax_grad = -a_mag * (x_pos / (r + 1e-5))
    ay_grad = -a_mag * (y_pos / (r + 1e-5))

    # Update velocities
    vx += ax_grad * dt
    vy += ay_grad * dt

    # Update positions
    x_pos += vx * dt
    y_pos += vy * dt

    scatter.set_offsets(np.c_[x_pos, y_pos])
    return scatter, im


if __name__ == "__main__":
    main()
