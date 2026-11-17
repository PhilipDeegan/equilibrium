"""
Compensated IC: phi_corrected = phi_0 + alpha * (phi_N - phi_0)

Builds a fresh IC from the npz, loads a settled snapshot, applies the
extrapolation, and saves the result as a new HDF5 for use with --load.

  alpha = 1.0  → just use phi_N (the settled snapshot phi as-is)
  alpha > 1.0  → overshoot toward predicted equilibrium
  alpha = 2.0  → pre-apply two rounds worth of settling in one shot

u and lam are taken from the settled snapshot (already partially relaxed).

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/compensate_ic.py \
      shell_profile_R100_s10.h5 \
      --npz /home/p/git/phare/master/ic_b.npz \
      --R 100 --phare-scale 10.0 --alpha 2.0 \
      --out compensated_ic.h5
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import init_phare_ic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", help="settled HDF5 snapshot (phi_N)")
    parser.add_argument("--npz", required=True, help="path to ic_b.npz")
    parser.add_argument("--R", type=float, default=100.0)
    parser.add_argument("--phare-scale", type=float, default=10.0)
    parser.add_argument("--lam", type=float, default=9.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="extrapolation factor (1=just use snapshot, 2=double-step)",
    )
    parser.add_argument("--n-threads", type=int, default=10)
    parser.add_argument("--out", default="compensated_ic.h5")
    args = parser.parse_args()

    # ---- phi_0: fresh IC ---------------------------------------------------
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
    print(f"  N={cl0.n_leaves}", flush=True)

    # ---- phi_N: settled snapshot -------------------------------------------
    print(f"Loading {args.snapshot} ...", flush=True)
    alatN = _lat.AdaptiveLattice.load(args.snapshot)
    clN = alatN[0]
    phiN = np.array(clN.leaf_phi).T  # (N, 14)
    print(f"  N={clN.n_leaves}", flush=True)

    assert (
        phi0.shape == phiN.shape
    ), f"Shape mismatch: IC {phi0.shape} vs snapshot {phiN.shape}"

    # ---- compensate --------------------------------------------------------
    diff = phiN - phi0
    phi_c = phi0 + args.alpha * diff  # (N, 14)

    rms_before = float(np.sqrt(np.mean(diff**2)))
    rms_after = float(np.sqrt(np.mean((phi_c - phi0) ** 2)))
    print(f"  RMS(phi_N - phi_0)         = {rms_before:.6e}")
    print(f"  RMS(phi_comp - phi_0)      = {rms_after:.6e}  (alpha={args.alpha})")

    # ---- write compensated phi into snapshot lattice and save --------------
    # u and lam kept from settled snapshot (already partially relaxed)
    clN.set_leaf_phi(phi_c.T)  # expects (14, N)
    alatN.save(args.out)
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
