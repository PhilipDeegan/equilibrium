"""
Far-field flat-tail check (doc/mems/project_poynting_perf_r150_farfield.md next step):
does phi_rms(r) outside the ring resolve into real 1/r-family decay with more
domain room, or does it stay flat all the way to the sponge regardless of R?

Same radial_phi_profile/fit_power_law pattern as
exp/2026/08/20/18_00_cycling_ring_charge_test/probe_radial_profile.py, applied to a
saved AdaptiveLattice checkpoint instead of a fresh short run.

Usage:
    python3 exp/2026/09/08/12_15_poynting_time_evolution/probe_farfield_radial_profile.py \
        --state <path.h5> [--r-min 30] [--r-max 85] [--n-bins 25] [--axis z] [--mode octagon] [--hops 8]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--state", type=str, required=True)
parser.add_argument("--axis", default="z")
parser.add_argument("--mode", type=str, default="octagon", choices=["square", "octagon"])
parser.add_argument("--hops", type=int, default=8)
parser.add_argument("--r-min", type=float, default=30.0)
parser.add_argument("--r-max", type=float, default=None, help="default: 0.95 * max radius present")
parser.add_argument("--n-bins", type=int, default=25)
parser.add_argument("--out-prefix", type=str, default=None, help="default: <state stem>_farfield_profile")
args = parser.parse_args()


def radial_phi_profile(pos, phi_rms, ring_centre, r_min, r_max, n_bins):
    r_all = np.linalg.norm(pos - ring_centre, axis=1)
    edges = np.linspace(r_min, r_max, n_bins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    totals = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    idx_all = ((r_all - r_min) / (r_max - r_min) * n_bins).astype(int)
    valid = (r_all >= r_min) & (r_all < r_max) & (idx_all >= 0) & (idx_all < n_bins)
    np.add.at(totals, idx_all[valid], phi_rms[valid])
    np.add.at(counts, idx_all[valid], 1)
    mask = counts > 0
    mean_prof = np.where(mask, totals / np.maximum(counts, 1), np.nan)
    return r_mid, mean_prof, counts


def fit_power_law(r_mid, profile):
    mask = np.isfinite(profile) & (profile > 0) & (r_mid > 0)
    if mask.sum() < 3:
        return None, None, mask.sum()
    lr = np.log(r_mid[mask])
    ly = np.log(profile[mask])
    coeffs = np.polyfit(lr, ly, 1)
    return float(np.exp(coeffs[1])), float(coeffs[0]), mask.sum()


def main():
    out_prefix = args.out_prefix or args.state.rsplit(".h5", 1)[0] + "_farfield_profile"

    alat = _lat.AdaptiveLattice.load(args.state)
    cl = alat[0]
    pos = np.array(cl.leaf_pos)
    r_all = np.linalg.norm(pos, axis=1)
    r_max_present = float(r_all.max())
    r_max = args.r_max if args.r_max is not None else 0.95 * r_max_present

    ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode=args.mode, hops=args.hops)
    if ring_js is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    phi_full = np.asarray(cl.leaf_phi)  # (14, N)
    phi_rms = np.sqrt((phi_full**2).mean(axis=0))  # (N,)

    print(
        f"N={cl.n_leaves} ring_r={ring_r:.3f} r_max_present={r_max_present:.2f} "
        f"profiling r in [{args.r_min:.1f}, {r_max:.1f}] cells over {args.n_bins} bins",
        flush=True,
    )

    r_mid, profile, counts = radial_phi_profile(pos, phi_rms, ring_centre, args.r_min, r_max, args.n_bins)
    A, slope, n_pts = fit_power_law(r_mid, profile)
    print(f"power-law fit over {n_pts} bins: A={A}  slope={slope}  (flat tail => slope~0; dipole 1/r^3 => slope~-3)")
    for r0, p, c in zip(r_mid, profile, counts):
        print(f"  r={r0:7.2f}  phi_rms={p:.4e}  n={c}")

    np.savez(f"{out_prefix}.npz", r_mid=r_mid, profile=profile, counts=counts, A=A, slope=slope, ring_r=ring_r)
    print(f"saved {out_prefix}.npz")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.isfinite(profile) & (profile > 0)
    ax.loglog(r_mid[mask], profile[mask], "o", color="tab:blue", label=f"slope={slope:.3f}" if slope is not None else "data")
    if A is not None:
        rr = np.linspace(r_mid[mask].min(), r_mid[mask].max(), 50)
        ax.loglog(rr, A * rr**slope, "--", color="tab:blue", alpha=0.6, label="power-law fit")
    ax.set_xlabel("r (distance from ring centre, cells)")
    ax.set_ylabel("phi_rms(r)")
    ax.legend()
    ax.set_title(f"Far-field phi falloff, {args.state}")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=130)
    print(f"saved {out_prefix}.png")


if __name__ == "__main__":
    main()
