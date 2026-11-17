"""Reconstruct + plot PHARE's spheromak B-field YZ slice (~/tmp/EM_B.vtkhdf,
a VTK Overlapping-AMR HDF5 export, x-slice at i=51, 101x101 cells) at a
handful of representative timesteps, for visual comparison against the DST
phi-reconstructed vector field (plot_phi_vector_per_level.py).

File layout (reverse-engineered, no VTKHDF reader library used):
  VTKHDF/Steps/Values                     -- (n_steps,) time stamps
  VTKHDF/Level0/AMRBox                    -- (total_boxes, 6) [imin,imax,jmin,jmax,kmin,kmax]
                                              CELL index ranges (SAMRAI patch boxes)
  VTKHDF/Steps/Level0/{AMRBoxOffset,NumberOfAMRBox}  -- per-step slice into AMRBox
  VTKHDF/Steps/Level0/PointDataOffset/data           -- per-step offset into PointData
  VTKHDF/Level0/PointData/data            -- (total_points, 3) flattened point data (Bx,By,Bz),
                                              concatenated box-by-box, step-by-step

Point counts per box are (cell_range+1) in ALL 3 dims (standard VTK point/cell
convention), including the degenerate x dimension (imin==imax always here --
a single-cell-thick slice exported as 2 x-planes). Point storage order within
a box is i-fastest (x), then j (y), then k (z) -- standard VTK structured
order. The two x-planes are numerically identical for 6/7 boxes per step
(confirmed max|plane0-plane1|=0 in the coarser boxes); the finest box shows a
small (~0.15, ~few% of field scale) difference, presumably a real x-gradient
sampled at the two bounding faces of that single fine cell -- plane i=0 is
used throughout, not averaged.

Level1 in this file is empty (0 boxes at every step) -- Level0 alone gives
full (102,102) point coverage of the y in [0,101], z in [0,101] slice, no gaps.

Run:
    PYTHONPATH=$PWD python3 exp/.../plot_phare_b_slice.py --out-dir plots/PHARE_reference
"""

import argparse
import os
import sys

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py


def load_step_grid(f, step):
    amrbox = f["VTKHDF/Level0/AMRBox"]
    boxoff = f["VTKHDF/Steps/Level0/AMRBoxOffset"][:]
    nbox = f["VTKHDF/Steps/Level0/NumberOfAMRBox"][:]
    pdoff = f["VTKHDF/Steps/Level0/PointDataOffset/data"][:]
    pdata = f["VTKHDF/Level0/PointData/data"]

    b0, nb = boxoff[step], nbox[step]
    boxes = amrbox[b0 : b0 + nb]
    cursor = pdoff[step]

    grid = np.full((102, 102, 3), np.nan)
    for bx in boxes:
        imin, imax, jmin, jmax, kmin, kmax = (int(v) for v in bx)
        nx, ny, nz = imax - imin + 2, jmax - jmin + 2, kmax - kmin + 2
        n = nx * ny * nz
        chunk = pdata[cursor : cursor + n]
        cursor += n
        arr = chunk.reshape(nz, ny, nx, 3)  # k, j, i, 3 (i fastest)
        plane = arr[:, :, 0, :]  # i=0 face
        grid[kmin : kmax + 2, jmin : jmax + 2, :] = plane
    return grid  # grid[k, j, :] = (Bx, By, Bz) at (y=j, z=k)


def plot_step(f, step, t, out_png):
    grid = load_step_grid(f, step)
    By, Bz, Bx = grid[:, :, 1], grid[:, :, 2], grid[:, :, 0]
    n = grid.shape[0]
    yi = np.arange(n) - (n - 1) / 2.0
    zi = np.arange(n) - (n - 1) / 2.0
    Yi, Zi = np.meshgrid(yi, zi)  # Yi[k,j], Zi[k,j]

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    im = ax.pcolormesh(
        Yi,
        Zi,
        Bx,
        cmap="RdBu_r",
        shading="auto",
        vmin=-np.nanmax(np.abs(Bx)),
        vmax=np.nanmax(np.abs(Bx)),
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Bx (out-of-plane)")

    speed = np.sqrt(By**2 + Bz**2)
    lw = 0.5 + 2.5 * (speed / np.nanmax(speed))
    ax.streamplot(
        Yi,
        Zi,
        np.nan_to_num(By),
        np.nan_to_num(Bz),
        color="k",
        density=1.6,
        linewidth=np.nan_to_num(lw, nan=0.5),
        arrowsize=1.0,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("y (cell index, centred)")
    ax.set_ylabel("z (cell index, centred)")
    ax.set_title(
        f"PHARE EM_B  step={step}  t={t:.3f}  Bx (bg) + in-plane (By,Bz) field lines"
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[step {step}] t={t:.3f} -> {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.expanduser("~/tmp/EM_B.vtkhdf"))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help="explicit step indices; default: 5 steps spread over the run",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with h5py.File(args.path, "r") as f:
        values = f["VTKHDF/Steps/Values"][:]
        n_steps = len(values)
        steps = args.steps or sorted(
            set(int(round(x)) for x in np.linspace(0, n_steps - 1, 6))
        )
        for step in steps:
            out_png = os.path.join(args.out_dir, f"phare_b_step_{step:03d}.png")
            plot_step(f, step, values[step], out_png)


if __name__ == "__main__":
    main()
