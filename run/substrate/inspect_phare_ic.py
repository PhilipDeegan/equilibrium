"""Visualise the raw PHARE ic_b.npz to determine the flux ring orientation."""

if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="path to ic_b.npz")
    parser.add_argument("--out", default="phare_ic_slices.png")
    args = parser.parse_args()

    d = np.load(args.npz)
    x, y, z = d["x"], d["y"], d["z"]
    bx, by, bz = d["bx"], d["by"], d["bz"]
    bmag = np.sqrt(bx**2 + by**2 + bz**2)

    # Grid is (Nx, Ny, Nz) — find centre indices
    cx = len(x) // 2
    cy = len(y) // 2
    cz = len(z) // 2

    print(f"Grid: {bx.shape}, x=[{x[0]:.2f},{x[-1]:.2f}], spacing={x[1]-x[0]:.3f}")
    print(
        f"B_max={bmag.max():.4f}  at index {np.unravel_index(bmag.argmax(), bmag.shape)}"
    )
    print(f"Centre indices: ({cx},{cy},{cz})")

    # --- 6-panel: |B| and in-plane streamlines for each slice ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("PHARE ic_b.npz — |B| + in-plane field lines", fontsize=13)

    # XY slice at z=cz
    ax = axes[0, 0]
    im = ax.pcolormesh(x, y, bmag[:, :, cz].T, cmap="inferno", shading="auto")
    plt.colorbar(im, ax=ax)
    # streamlines need 2D arrays on a regular grid
    skip = 4
    ax.streamplot(
        x[::skip],
        y[::skip],
        bx[::skip, ::skip, cz].T,
        by[::skip, ::skip, cz].T,
        color="white",
        linewidth=0.5,
        density=1.0,
        arrowsize=0.7,
    )
    ax.set_title(f"XY  z={z[cz]:.2f}  — |B|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # XZ slice at y=cy
    ax = axes[0, 1]
    im = ax.pcolormesh(x, z, bmag[:, cy, :].T, cmap="inferno", shading="auto")
    plt.colorbar(im, ax=ax)
    ax.streamplot(
        x[::skip],
        z[::skip],
        bx[::skip, cy, ::skip].T,
        bz[::skip, cy, ::skip].T,
        color="white",
        linewidth=0.5,
        density=1.0,
        arrowsize=0.7,
    )
    ax.set_title(f"XZ  y={y[cy]:.2f}  — |B|")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    # YZ slice at x=cx
    ax = axes[0, 2]
    im = ax.pcolormesh(y, z, bmag[cx, :, :].T, cmap="inferno", shading="auto")
    plt.colorbar(im, ax=ax)
    ax.streamplot(
        y[::skip],
        z[::skip],
        by[cx, ::skip, ::skip].T,
        bz[cx, ::skip, ::skip].T,
        color="white",
        linewidth=0.5,
        density=1.0,
        arrowsize=0.7,
    )
    ax.set_title(f"YZ  x={x[cx]:.2f}  — |B|")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    # Bottom row: Bx, By, Bz each at the slice with max |B| to see dominant component
    for col, (label, comp) in enumerate([("Bx", bx), ("By", by), ("Bz", bz)]):
        ax = axes[1, col]
        # use the XZ slice (y=cy) to show variation in x and z
        vm = np.abs(comp[:, cy, :]).max()
        im = ax.pcolormesh(
            x, z, comp[:, cy, :].T, cmap="seismic", vmin=-vm, vmax=vm, shading="auto"
        )
        plt.colorbar(im, ax=ax)
        ax.set_title(f"XZ  y={y[cy]:.2f}  — {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved → {args.out}")
