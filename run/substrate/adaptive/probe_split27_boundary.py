"""
Coarse-fine boundary test for the 27-child split, in pure Python -- no C++
changes. Builds on probe_split27.py's single-cell refinement (one L0 cell ->
27 L1 children, surrounded by unrefined L0 neighbours) and asks the two
questions that matter before touching adaptive_lattice.hpp/bcc_geometry.hpp:

  1. PROLONGATION (coarse -> fine): given only the parent cell's own value
     (and its neighbours', for a gradient reconstruction), how should the 27
     children be initialised? Compare constant injection vs a gradient-
     corrected fill against a known analytic field, and check the
     injection is conservative (children's volume-mean reproduces the
     parent's value exactly).

  2. ACCUMULATION / flux matching at the interface (the "reflux" problem):
     each of the parent's 14 coarse neighbours can compute its own outward
     flux across the shared face using ONE naive coarse-resolution estimate
     (area/distance at level 0, against some single aggregate child value),
     while the children compute 27-way finer flux across the same physical
     face independently. These two do NOT match in general -- same issue
     already fixed once for the 8/9-child split (see project_subcycle_amr_
     reflux memory: "reflux fix verified"). This script quantifies the
     mismatch ("leak") for the naive scheme, and shows the reflux-consistent
     fix (reuse the fine-computed sum, negated, as the coarse cell's actual
     contribution) is exact by construction.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_split27_boundary.py --R 10
"""

import argparse

import numpy as np

from run.substrate.adaptive.probe_split27 import (
    ALL_DELTA,
    build_bcc_sphere_idx,
    split27,
)

H = 1.0 / np.sqrt(3.0)  # coarse h-unit physical size, d1=1
A0 = 1.0
W_HEX, W_SQ = (
    2.0 / 3.0,
    1.0 / 3.0,
)  # face-area proxies, from kelvin.py's Laplacian weights
D_HEX, D_SQ = 1.0, 2.0 / np.sqrt(3.0)  # d1, d2 (coarse)


def face_kind(k):
    return "hex" if k < 8 else "sq"


def coarse_area(k):
    return A0 * (W_HEX if k < 8 else W_SQ)


def coarse_dist(k):
    return D_HEX if k < 8 else D_SQ


def fine_area(k):
    return coarse_area(k) / 9.0  # (L_child/L_parent)^2 = (1/3)^2


def u_field(pos, center, sigma):
    r2 = np.sum((pos - center) ** 2, axis=-1)
    return np.exp(-r2 / (2.0 * sigma**2))


def gradient_fit(offsets, du):
    """Least-squares gradient from 14 neighbour offsets/value-differences."""
    D = offsets  # (14,3)
    g, *_ = np.linalg.lstsq(D, du, rcond=None)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--cell", type=int, nargs=3, default=(0, 0, 0))
    ap.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=(3.0, 0.0, 0.0),
        help="centre of the Gaussian test field, physical d1 units",
    )
    ap.add_argument("--sigma", type=float, default=2.5)
    args = ap.parse_args()
    center = np.array(args.center)

    idx0 = build_bcc_sphere_idx(args.R)
    key_to_row = {tuple(row): i for i, row in enumerate(idx0.tolist())}
    i0 = key_to_row[tuple(args.cell)]
    p = idx0[i0]
    pos_parent = p * H

    present, slots = [], []
    for k, d in enumerate(ALL_DELTA):
        q = tuple((p + d).tolist())
        if q in key_to_row:
            present.append(idx0[key_to_row[q]])
            slots.append(k)
    if len(present) != 14:
        raise SystemExit(
            f"cell {args.cell} is not fully interior ({len(present)}/14 nb) -- pick another"
        )

    children, _ = split27(p, present, window=args.window)
    pos_children = children.astype(np.float64) * (H / 3.0)
    my_set = {tuple(c) for c in children.tolist()}

    # true (analytic) field, sampled directly -- the "exact fine solution" we
    # compare prolongation against, and use for the flux/accumulation test.
    u_true_child = u_field(pos_children, center, args.sigma)
    u_parent = float(u_field(pos_parent[None, :], center, args.sigma)[0])

    # ------------------------------------------------------------------
    # 1. Prolongation
    # ------------------------------------------------------------------
    print("=== Prolongation (coarse -> 27 children) ===")
    u_inj = np.full(27, u_parent)

    nb_offsets = np.array([n * H - pos_parent for n in present])  # (14,3) physical
    u_nb = u_field(np.array([n * H for n in present]), center, args.sigma)
    grad = gradient_fit(nb_offsets, u_nb - u_parent)
    u_grad = u_parent + pos_children @ grad - pos_parent @ grad

    rms_inj = float(np.sqrt(np.mean((u_inj - u_true_child) ** 2)))
    rms_grad = float(np.sqrt(np.mean((u_grad - u_true_child) ** 2)))
    print(f"  parent u = {u_parent:.6f}")
    print(
        f"  constant-injection : RMS error vs true = {rms_inj:.6e}   mean(children) = {u_inj.mean():.6f}"
    )
    print(
        f"  gradient-corrected : RMS error vs true = {rms_grad:.6e}   mean(children) = {u_grad.mean():.6f}"
    )
    print(
        f"  (mean(children) should equal parent u={u_parent:.6f} exactly for a conservative "
        f"prolongation -- both do, by the 27-child set's inversion symmetry)"
    )

    # ------------------------------------------------------------------
    # 2. Accumulation / flux matching at the interface
    # ------------------------------------------------------------------
    print("\n=== Accumulation / interface flux matching (per coarse face) ===")
    print(
        f"{'k':>3} {'kind':>4} {'n_ext':>5} {'F_naive':>12} {'F_fine_sum':>12} {'leak=F_n+F_f':>13}"
    )
    leaks = []
    for n_row, k in zip(present, slots):
        pos_nb = n_row * H
        u_n = float(u_field(pos_nb[None, :], center, args.sigma)[0])

        ext_children = []
        for c in children.tolist():
            nb_pt = tuple((np.array(c) + ALL_DELTA[k]).tolist())
            if nb_pt not in my_set:
                ext_children.append(c)
        idxs = [i for i, c in enumerate(children.tolist()) if c in ext_children]
        u_face_mean = float(u_true_child[idxs].mean()) if idxs else u_parent

        # naive: neighbour's own coarse-resolution view of the shared face
        F_naive = coarse_area(k) * (u_face_mean - u_n) / coarse_dist(k)

        # fine: sum of each externally-facing child's own flux to the neighbour
        F_fine_sum = 0.0
        for i in idxs:
            d_phys = float(np.linalg.norm(pos_children[i] - pos_nb))
            F_fine_sum += fine_area(k) * (u_n - u_true_child[i]) / d_phys

        leak = F_naive + F_fine_sum
        leaks.append(leak)
        print(
            f"{k:>3} {face_kind(k):>4} {len(idxs):>5} {F_naive:>12.6f} {F_fine_sum:>12.6f} {leak:>13.6f}"
        )

    leaks = np.array(leaks)
    print(
        f"\nmax |leak| (naive, independent-side flux calc) = {np.abs(leaks).max():.6e}"
    )
    print(
        "-> nonzero: the neighbour's naive coarse-side estimate and the sum of the "
        "children's fine-side estimates do NOT match -- same 'reflux' mismatch already "
        "fixed once for the 8/9-child split."
    )
    print(
        "reflux-consistent fix: define the neighbour's actual contribution as "
        "-F_fine_sum (reuse the fine value, negated) instead of recomputing F_naive "
        "independently -> leak == 0 exactly, by construction, for every face."
    )


if __name__ == "__main__":
    main()
