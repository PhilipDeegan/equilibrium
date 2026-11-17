"""
diff_lattice.py — compare two saved lattice .npz snapshots (e.g. IC vs final).

Usage:
    PYTHONPATH=$PWD python3 run/substrate/diff_lattice.py \\
        tmp/lattice/rad_ic.npz tmp/lattice/rad.npz

Prints per-field stats and plots radial profiles of Δu, Δlam, Δphi_rms.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ic", help="IC snapshot (.npz)")
    parser.add_argument("final", help="final snapshot (.npz)")
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="save figure to this path (e.g. tmp/lattice/diff.png); omit to show interactively",
    )
    parser.add_argument("--n-bins", type=int, default=80)
    args = parser.parse_args()

    ic = np.load(args.ic)
    final = np.load(args.final)

    du = final["u"] - ic["u"]
    dlam = final["lam"] - ic["lam"]
    dphi = final["phi"] - ic["phi"]  # shape (N, 14)

    pos = ic["pos"]
    r = np.linalg.norm(pos, axis=1)
    R = float(ic["R"])

    print(f"R={R}  N={len(r)}")
    print(
        f"\nΔu   : min={du.min():.4f}  max={du.max():.4f}  mean={du.mean():.4e}  rms={np.sqrt(np.mean(du**2)):.4e}"
    )
    print(
        f"Δlam : min={dlam.min():.4f}  max={dlam.max():.4f}  mean={dlam.mean():.4e}  sum={dlam.sum():.6f}"
    )
    print(
        f"Δphi : rms={np.sqrt(np.mean(dphi**2)):.4e}  max|Δphi|={np.abs(dphi).max():.4f}"
    )

    # radial bins
    edges = np.linspace(0, R, args.n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    du_r = np.zeros(args.n_bins)
    dlam_r = np.zeros(args.n_bins)
    dphi_r = np.zeros(args.n_bins)
    cnt = np.zeros(args.n_bins, dtype=int)

    idx = np.searchsorted(edges[1:], r, side="left").clip(0, args.n_bins - 1)
    for b in range(args.n_bins):
        m = idx == b
        if m.any():
            du_r[b] = du[m].mean()
            dlam_r[b] = dlam[m].mean()
            dphi_r[b] = np.sqrt(np.mean(dphi[m] ** 2))
            cnt[b] = m.sum()

    import matplotlib

    if args.plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"IC → final diff  (R={R})")

    axes[0].plot(centres, du_r)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("<Δu>")
    axes[0].set_title("Δu radial mean")

    axes[1].plot(centres, dlam_r, color="red")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("<Δλ>")
    axes[1].set_title("Δλ radial mean")

    axes[2].plot(centres, dphi_r, color="green")
    axes[2].set_xlabel("r")
    axes[2].set_ylabel("Δφ rms")
    axes[2].set_title("Δφ rms radial")

    plt.tight_layout()
    if args.plot:
        plt.savefig(args.plot, dpi=120)
        print(f"saved → {args.plot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
