"""
Extract |B|² from PHARE EM_B.vtkhdf (all-primal, t=3, two AMR levels) and
produce a radial lam profile for DST initialisation in explore_taylor.py.

The vtkhdf stores primal nodes for each AMR box sequentially.
Level0: dl=0.1 (base),  Level1: dl=0.05 (2x refinement around core).
Domain centre: (5.05, 5.05, 5.05) for cells=(101,101,101), dl=0.1.

Output npz:
    r_bins   : (Nr,)  — radial bin centres in PHARE physical units
    b2_radial: (Nr,)  — mean |B|² per radial bin
    r_phys   : (N,)   — raw radial distances (all nodes, both levels)
    b2_raw   : (N,)   — raw |B|² at each node

Run:
    PYTHONPATH=$PWD python3 run/substrate/phare_b_to_lam.py
    PYTHONPATH=$PWD python3 run/substrate/phare_b_to_lam.py --vtkhdf /path/to/EM_B.vtkhdf
"""

import argparse
from pathlib import Path

import numpy as np

DEFAULT_VTKHDF = Path("/home/deegan/git/phare/master/EM_B.vtkhdf")
CENTRE = np.array([5.05, 5.05, 5.05])  # domain centre for cells=101, dl=0.1


def read_level(f, level_name, dl, step=1):
    """
    Read all boxes for one AMR level at the given step index (0 or 1).
    Returns (positions, b2) arrays — one entry per primal node.

    AMRBox layout: [lo_x, hi_x, lo_y, hi_y, lo_z, hi_z] in cell units.
    Primal node count per dim: hi - lo + 2.
    Data is ordered x-fastest (Fortran/VTK convention) within each box.
    """
    boxes_all = f[f"VTKHDF/{level_name}/AMRBox"][:]
    data_all = f[f"VTKHDF/{level_name}/PointData/data"][:]
    nbox_per = f[f"VTKHDF/Steps/{level_name}/NumberOfAMRBox"][:]
    pt_off = f[f"VTKHDF/Steps/{level_name}/PointDataOffset/data"][:]

    box_start = nbox_per[:step].sum()
    box_end = box_start + nbox_per[step]
    boxes = boxes_all[box_start:box_end]
    data = data_all[
        pt_off[step] : pt_off[step]
        + sum((b[1] - b[0] + 2) * (b[3] - b[2] + 2) * (b[5] - b[4] + 2) for b in boxes)
    ]

    pos_list = []
    b2_list = []
    ptr = 0

    for box in boxes:
        nx = box[1] - box[0] + 2
        ny = box[3] - box[2] + 2
        nz = box[5] - box[4] + 2
        n = nx * ny * nz

        # physical node coordinates (x varies fastest — Fortran/VTK order)
        xi = (box[0] + np.arange(nx)) * dl
        yj = (box[2] + np.arange(ny)) * dl
        zk = (box[4] + np.arange(nz)) * dl
        XX, YY, ZZ = np.meshgrid(xi, yj, zk, indexing="ij")  # (nx, ny, nz)

        # data for this box: (n, 3), stored x-fastest → reshape (nx, ny, nz, 3)
        b = data[ptr : ptr + n].reshape(nx, ny, nz, 3)
        b2 = (b**2).sum(axis=-1)  # (nx, ny, nz)

        pos_list.append(np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1))
        b2_list.append(b2.ravel())
        ptr += n

    return np.concatenate(pos_list), np.concatenate(b2_list)


def extract(vtkhdf=DEFAULT_VTKHDF, out_npz="phare_b2_radial.npz", n_bins=200):
    import h5py

    print(f"Reading {vtkhdf}")
    with h5py.File(vtkhdf, "r") as f:
        times = f["VTKHDF/Steps/Values"][:]
        print(f"  timestamps: {times}")
        step = 1  # second entry = t=3

        pos0, b2_0 = read_level(f, "Level0", dl=0.1, step=step)
        pos1, b2_1 = read_level(f, "Level1", dl=0.05, step=step)

    print(f"  Level0 nodes: {len(b2_0)}  |B|² max: {b2_0.max():.4f}")
    print(f"  Level1 nodes: {len(b2_1)}  |B|² max: {b2_1.max():.4f}")

    pos = np.concatenate([pos0, pos1])
    b2 = np.concatenate([b2_0, b2_1])
    r = np.linalg.norm(pos - CENTRE, axis=1)

    # radial profile (Level1 nodes dominate the core — higher resolution wins)
    r_max = r.max()
    edges = np.linspace(0, r_max, n_bins + 1)
    bins = np.digitize(r, edges) - 1
    b2_rad = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bins == i
        if mask.sum() > 0:
            b2_rad[i] = b2[mask].mean()
            counts[i] = mask.sum()
    r_bins = 0.5 * (edges[:-1] + edges[1:])

    np.savez_compressed(
        out_npz,
        r_bins=r_bins,
        b2_radial=b2_rad,
        r_phys=r,
        b2_raw=b2,
        centre=CENTRE,
    )
    print(f"Saved → {out_npz}")
    print(f"  r range: [0, {r_max:.3f}]  peak |B|² at r={r_bins[b2_rad.argmax()]:.3f}")
    print(f"  |B|² peak: {b2_rad.max():.4f}  at r=0+: {b2_rad[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtkhdf", default=str(DEFAULT_VTKHDF))
    parser.add_argument("--out", default="phare_b2_radial.npz")
    parser.add_argument("--n-bins", type=int, default=200)
    args = parser.parse_args()
    extract(Path(args.vtkhdf), args.out, args.n_bins)
