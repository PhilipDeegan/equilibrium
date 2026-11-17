"""
Diff between a baseline phi and a settled snapshot phi_N.

Baseline is either:
  - a fresh PHARE IC projection (default, requires --npz)
  - a previous HDF5 snapshot (--baseline-snapshot), for consecutive-step diffs

Convergence check: run repeatedly with consecutive snapshots as baseline;
diff should shrink toward zero as the system reaches local equilibrium.

Usage (IC vs snapshot):
  PYTHONPATH=$PWD python3 run/substrate/adaptive/diff_ic_snapshot.py \
      shell_profile_R100_s10.h5 \
      --npz /home/p/git/phare/master/ic_b.npz \
      --R 100 --phare-scale 10.0 --out ic_diff.png

Usage (snapshot A vs snapshot B):
  PYTHONPATH=$PWD python3 run/substrate/adaptive/diff_ic_snapshot.py \
      snap_2000.h5 --baseline-snapshot snap_1000.h5 --out diff_1000_2000.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import init_phare_ic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", help="HDF5 snapshot (phi after N steps)")
    parser.add_argument(
        "--baseline-snapshot",
        type=str,
        default=None,
        help="compare two snapshots instead of IC vs snapshot",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="path to ic_b.npz (required if no --baseline-snapshot)",
    )
    parser.add_argument("--R", type=float, default=100.0)
    parser.add_argument("--phare-scale", type=float, default=10.0)
    parser.add_argument("--lam", type=float, default=9.0)
    parser.add_argument("--n-threads", type=int, default=10)
    parser.add_argument("--thickness", type=float, default=2.0)
    parser.add_argument("--out", default="ic_diff.png")
    args = parser.parse_args()

    gridsize = 250

    # ---- phi_0: baseline ---------------------------------------------------
    if args.baseline_snapshot:
        print(f"Loading baseline {args.baseline_snapshot} ...", flush=True)
        alat0 = _lat.AdaptiveLattice.load(args.baseline_snapshot)
        cl0 = alat0[0]
        phi0 = np.array(cl0.leaf_phi).T  # (N, 14)
        pos = np.array(cl0.leaf_pos)  # (N, 3)
        baseline_label = args.baseline_snapshot
        print(f"  N={cl0.n_leaves}", flush=True)
    else:
        assert args.npz, "Provide --npz (IC) or --baseline-snapshot"
        print("Building IC sphere ...", flush=True)
        alat0 = _lat.AdaptiveLattice()
        alat0.set_n_threads(args.n_threads)
        cl0 = alat0[0]
        _lat.build_bcc_sphere(cl0, args.R, d1=1.0, u0=0.0, lam0=0)
        alat0.rebuild(0)
        init_phare_ic(
            cl0,
            args.npz,
            scale=args.phare_scale,
            lam_target=args.lam,
            u_max=1.0,
            amplitude=1.0,
        )
        phi0 = np.array(cl0.leaf_phi).T  # (N, 14)
        pos = np.array(cl0.leaf_pos)  # (N, 3)
        baseline_label = f"PHARE IC (scale={args.phare_scale})"
        print(f"  N={cl0.n_leaves}", flush=True)

    # ---- phi_N: loaded snapshot --------------------------------------------
    print(f"Loading {args.snapshot} ...", flush=True)
    alatN = _lat.AdaptiveLattice.load(args.snapshot)
    clN = alatN[0]
    phiN = np.array(clN.leaf_phi).T  # (N, 14)
    print(f"  N={clN.n_leaves}", flush=True)

    assert (
        phi0.shape == phiN.shape
    ), f"Shape mismatch: baseline {phi0.shape} vs snapshot {phiN.shape}"

    rms_diff = float(np.sqrt(np.mean((phiN - phi0) ** 2)))
    print(f"  RMS(phi_N - phi_0) = {rms_diff:.6e}", flush=True)

    # ---- diff quantities ---------------------------------------------------
    dphi = phiN - phi0  # (N, 14)  signed change per face
    dphi_mag = np.sqrt((dphi**2).sum(axis=1))  # (N,)     magnitude of change
    phi0_mag = np.sqrt((phi0**2).sum(axis=1))
    phiN_mag = np.sqrt((phiN**2).sum(axis=1))
    dmag = phiN_mag - phi0_mag  # (N,)     net gain/loss of |phi|

    t = args.thickness

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"phi diff:  {args.snapshot}  −  {baseline_label}  RMS={rms_diff:.3e}",
        fontsize=11,
    )

    slices = [
        ("XY (z≈0)", np.abs(pos[:, 2]) < t, 0, 1, "x", "y"),
        ("XZ (y≈0)", np.abs(pos[:, 1]) < t, 0, 2, "x", "z"),
        ("YZ (x≈0)", np.abs(pos[:, 0]) < t, 1, 2, "y", "z"),
    ]

    for col, (title, mask, ix, iy, xl, yl) in enumerate(slices):
        px, py = pos[mask, ix], pos[mask, iy]

        # Row 0: magnitude of change |dphi|
        ax = axes[0, col]
        hb = ax.hexbin(
            px,
            py,
            C=dphi_mag[mask],
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap="hot",
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — |Δphi|")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

        # Row 1: signed net gain/loss of phi magnitude
        ax = axes[1, col]
        vm = np.abs(dmag[mask]).max() or 1.0
        hb = ax.hexbin(
            px,
            py,
            C=dmag[mask],
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap="seismic",
            vmin=-vm,
            vmax=vm,
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — |phi_N| − |phi_0|  (red=gain, blue=loss)")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
