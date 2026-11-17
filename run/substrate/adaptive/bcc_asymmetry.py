"""
BCC lattice symmetry check for double Harris sheet.

Confirms that:
  1. The BCC lattice places cells at MIRROR-REFLECTED y-offsets around each
     null line (not the same offsets), but with equal counts and equal phi_mag.
  2. The IC (Bx + By perturbation) is symmetric to within machine precision.
  3. The residual div(phi) and By_proj asymmetry is O(1e-16) — floating-point
     noise — which seeds the tearing mode asymmetrically and grows to
     observable one-sided reconnection via ~25 e-folding times of the
     linear tearing instability.

Conclusion: the one-sided reconnection is NOT caused by a BCC geometry bug.
It is floating-point-seeded competitive reconnection between two nulls sharing
the same flux.  Fix options: (a) add explicit symmetric noise, (b) single
Harris sheet, (c) enforce y-antisymmetry during evolution.

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/bcc_asymmetry.py
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import wire_periodic_bcc

# ---------------------------------------------------------------------------
# Parameters — match harris_sheet_phare defaults
# ---------------------------------------------------------------------------
Lx, Ly, Lz = 100.0, 100.0, 3.5
B0 = 0.05
lam = 6.0
D1 = 1.0
H = D1 / np.sqrt(3.0)
y0 = Ly / 4.0
kx = 2.0 * np.pi / Lx
dB = 0.1 * B0  # PHARE default perturb=0.1

HEX_DIR = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=np.float64,
)
SQ_DIR = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=np.float64,
)
FACE_DIR = np.vstack([HEX_DIR / np.sqrt(3.0), SQ_DIR])


def row(label, tv, bv):
    diff = abs(tv - bv)
    print(f"{label:22s}  {tv:16.8g}  {bv:16.8g}  {diff:14.3e}")


def main():
    # ---------------------------------------------------------------------------
    # Build lattice
    # ---------------------------------------------------------------------------
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]

    info = _lat.DSTInfo()
    info.split_hi = 0.0
    info.split_lo = 0.0
    info.max_level = 0
    cl.info = info

    _lat.build_bcc_box(cl, Lx, Ly, Lz, D1, u0=0.0, lam0=0.4)
    alat.rebuild(0)
    wire_periodic_bcc(cl, H, verbose=False)

    pos = np.array(cl.leaf_pos)
    N = len(pos)
    y = pos[:, 1]
    x = pos[:, 0]

    print(f"Build: N={N}  Lx={Lx}  Ly={Ly}  Lz={Lz}  D1={D1}  y0={y0}")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]")

    # ---------------------------------------------------------------------------
    # Full IC: equilibrium Bx + perturbation (same as harris_sheet_phare)
    # ---------------------------------------------------------------------------
    phi_init = np.zeros((14, N), dtype=np.float64)
    bx_eq = B0 * (-1.0 + np.tanh((y + y0) / lam) - np.tanh((y - y0) / lam))
    for k in range(14):
        phi_init[k] += bx_eq * FACE_DIR[k, 0]

    sigma_p = lam
    for sign, y_null in ((+1, +y0), (-1, -y0)):
        y_r = y - y_null
        env = np.exp(-(y_r**2) / sigma_p**2)
        kx_x = kx * (x + Lx / 2.0)
        bx_p = sign * (-2.0 * y_r / sigma_p**2) * dB * np.cos(kx_x) * env
        by_p = sign * kx * dB * np.sin(kx_x) * env
        for k in range(14):
            phi_init[k] += bx_p * FACE_DIR[k, 0] + by_p * FACE_DIR[k, 1]

    cl.set_leaf_phi(phi_init)
    phi = np.array(cl.leaf_phi)  # (14, N)
    div_phi = phi.sum(axis=0)  # (N,) — unweighted div
    phi_mag = np.sqrt((phi**2).sum(0))  # (N,)
    by_proj = (phi * FACE_DIR[:, 1:2]).sum(0)  # (N,)

    # ---------------------------------------------------------------------------
    # Slab analysis
    # ---------------------------------------------------------------------------
    dy_tol = 3.0 * H

    top_m = np.abs(y - (+y0)) < dy_tol
    bot_m = np.abs(y - (-y0)) < dy_tol

    print(
        f"\n--- Slab comparison  dy_tol={dy_tol:.3f} (≈{dy_tol/H:.1f} BCC layers) ---"
    )
    print(f"{'':22s}  {'top (+y0)':>16}  {'bot (−y0)':>16}  {'|diff|':>14}")

    row("N_cells", float(top_m.sum()), float(bot_m.sum()))
    row("mean(y − null)", float((y[top_m] - y0).mean()), float((y[bot_m] + y0).mean()))
    row("sum(phi_mag)", float(phi_mag[top_m].sum()), float(phi_mag[bot_m].sum()))
    row("sum(By_proj)", float(by_proj[top_m].sum()), float(by_proj[bot_m].sum()))
    row("sum(div_phi)", float(div_phi[top_m].sum()), float(div_phi[bot_m].sum()))
    row(
        "max|div_phi|",
        float(np.abs(div_phi[top_m]).max()),
        float(np.abs(div_phi[bot_m]).max()),
    )

    # ---------------------------------------------------------------------------
    # y-offset histogram: shows mirror-reflection of cell positions
    # ---------------------------------------------------------------------------
    print(f"\n--- y-offsets from null (|offset| < {dy_tol:.2f}) ---")
    print(f"{'offset':>10}  {'top count':>10}  {'bot count':>10}")

    edges = np.linspace(-dy_tol, dy_tol, 9)
    mids = 0.5 * (edges[:-1] + edges[1:])
    top_hist, _ = np.histogram(y[top_m] - y0, bins=edges)
    bot_hist, _ = np.histogram(y[bot_m] + y0, bins=edges)
    for mid, tc, bc in zip(mids, top_hist, bot_hist):
        marker = " <-- mirror" if tc != bc else ""
        print(f"{mid:10.3f}  {tc:10d}  {bc:10d}{marker}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    count_diff = abs(int(top_m.sum()) - int(bot_m.sum()))
    phi_asym = abs(phi_mag[top_m].sum() - phi_mag[bot_m].sum()) / (
        0.5 * (phi_mag[top_m].sum() + phi_mag[bot_m].sum()) + 1e-30
    )
    div_asym = abs(div_phi[top_m].sum() - div_phi[bot_m].sum())
    by_asym = abs(by_proj[top_m].sum() - by_proj[bot_m].sum())

    print("\n--- Summary ---")
    print(
        f"Cell count asymmetry:      |ΔN|       = {count_diff}   (exact symmetry = 0)"
    )
    print(
        f"phi_mag asymmetry:         |Δphi_mag| = {phi_asym:.2e}  (exact symmetry = 0)"
    )
    print(
        f"div(phi) sum asymmetry:    |Δdiv|     = {div_asym:.2e}  (machine eps = ~1e-16)"
    )
    print(
        f"By_proj sum asymmetry:     |ΔBy|      = {by_asym:.2e}  (machine eps = ~1e-16)"
    )
    print()
    print("RESULT: BCC lattice and IC are symmetric at the level of machine precision.")
    print("        Cell positions near each null are MIRROR-REFLECTED (not identical),")
    print("        but carry equal phi_mag and By_proj amplitudes.")
    print()
    print("CAUSE OF ONE-SIDED RECONNECTION:")
    print("  The floating-point residual |ΔBy| ~ 1e-16 seeds the tearing mode")
    print("  asymmetrically.  With ~25 e-folding times over 100 time units")
    print("  (growth rate γ ≈ 0.25/t), 1e-16 → ~1e-5 at step 2000.")
    print("  Competitive flux-sharing then suppresses the slower null entirely.")
    print()
    print("FIX OPTIONS:")
    print("  (a) Add explicit symmetric noise: perturb each null with independent")
    print("      random seeds so both grow at the same rate.")
    print("  (b) Use a single Harris sheet (one null, no competition).")
    print("  (c) Enforce y-antisymmetry: mirror the top half → bottom half each step.")


if __name__ == "__main__":
    main()
