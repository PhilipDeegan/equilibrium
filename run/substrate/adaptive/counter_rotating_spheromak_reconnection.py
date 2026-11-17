"""
Counter-rotating spheromak collision — DST magnetic reconnection testbed.

Two Taylor spheromaks with opposite spin (rigid-body rotation about z) are
placed at ±z0. Their opposing toroidal phi wakes create an antiparallel-field
interface at the midplane. As phi diffuses inward, a current sheet (div-phi
spike) builds at z=0 and the topology reorganises — reconnection without
imposed resistivity.

Physical picture:
  spin+1 (at +z0, omega>0): phi_tor ∝ +(-y, x, 0)/rho   (CCW from above)
  spin-1 (at -z0, omega<0): phi_tor ∝ -(-y, x, 0)/rho   (CW — antiparallel)

Current sheet: when opposing phi wakes meet at z=0, div-phi spikes (J in DST).
Reconnection:  phi flux tubes rearrange topology → J drops, energy released.

Wikipedia (Spheromak): "They are particularly useful for studying magnetic
reconnection events, when two or more spheromaks collide."

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/counter_rotating_spheromak_reconnection.py

  Larger run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/counter_rotating_spheromak_reconnection.py \\
      --R 80 --scale 16 --z-sep 20 --omega 0.1 --n-steps 30000
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.discrete import LAM_MAX
from run.substrate.adaptive.lam_phi_drive import step_lam_relax
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    apply_toroidal_rotation,
    init_spheromak_ic,
)

# BCC face geometry — matches adaptive_flux_tube._FACE_NORM
_HEX_DIR = np.array(
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
_SQ_DIR = np.array(
    [
        [-1, 0, 0],
        [+1, 0, 0],
        [0, -1, 0],
        [0, +1, 0],
        [0, 0, -1],
        [0, 0, +1],
    ],
    dtype=np.float64,
)
_FACE_NORM = np.vstack([_HEX_DIR / np.sqrt(3), _SQ_DIR])  # (14, 3)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=60.0)
parser.add_argument(
    "--scale",
    type=float,
    default=12.0,
    help="DST units per spheromak unit; knot core radius ≈ scale DST units",
)
parser.add_argument(
    "--z-sep",
    type=float,
    default=16.0,
    help="centre-to-centre separation in DST units (spheromaks at ±z_sep/2)",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=0.7,
    help="Gaussian envelope in spheromak units (avoids sharp IC boundary)",
)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument(
    "--omega",
    type=float,
    default=0.1,
    help="toroidal rotation amplitude; +omega at +z0, -omega at -z0",
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument(
    "--lam-relax-rate",
    type=float,
    default=0.1,
    help="local lam relaxation rate toward alpha*|phi|  (0 = disabled)",
)
parser.add_argument(
    "--lam-alpha",
    type=float,
    default=1.0,
    help="phi→lam coupling: lam_target = alpha * |phi|",
)
parser.add_argument("--cross-scale", type=float, default=0.005)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--plot-out", type=str, default="tmp/reconnection.png")
parser.add_argument("--no-plot", action="store_true")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def current_sheet_j(phi_arr, z, dz):
    """Mean |div-phi| in midplane slab |z| < dz. Proxy for current density J."""
    mid = np.abs(z) < dz
    if not mid.any():
        return 0.0
    return float(np.abs(phi_arr.sum(axis=0)[mid]).mean())


def phi_rms_zone(phi_arr, mask):
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(phi_arr[:, mask] ** 2)))


def lam_mean_zone(lam_arr, mask):
    if not mask.any():
        return 0.0
    return float(lam_arr[mask].mean())


def circ_z_com(phi_arr, pos, z_bins):
    """z-COM of azimuthal phi circulation (from SG experiment). Returns scalar."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    rho = np.sqrt(x**2 + y**2)
    valid = rho > 0.5
    phi_hat_x = np.where(valid, -y / np.where(rho > 0, rho, 1.0), 0.0)
    phi_hat_y = np.where(valid, x / np.where(rho > 0, rho, 1.0), 0.0)
    proj = (
        _FACE_NORM[:, 0:1] * phi_hat_x[None, :]
        + _FACE_NORM[:, 1:2] * phi_hat_y[None, :]
    )
    circ_cell = np.where(valid, (phi_arr * proj).sum(axis=0), 0.0)
    n_bins = len(z_bins) - 1
    bin_idx = np.clip(np.digitize(z, z_bins) - 1, 0, n_bins - 1)
    circ_v = np.zeros(n_bins)
    np.add.at(circ_v, bin_idx, circ_cell)
    z_mids = 0.5 * (z_bins[:-1] + z_bins[1:])
    abs_circ = np.abs(circ_v)
    total = float(abs_circ.sum())
    return float((abs_circ * z_mids).sum() / total) if total > 1e-30 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    pos = np.array(cl.leaf_pos)  # (N, 3)
    z = pos[:, 2]
    r_all = np.linalg.norm(pos, axis=1)
    r_max = float(r_all.max())
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0,
            damp_max=args.damp_max,
            r_all=r_all,
            r_inner=r_max * (1.0 - args.sponge),
        )
    )

    z0 = args.z_sep / 2.0
    print(
        f"N={cl.n_leaves}  R={args.R}  scale={args.scale}  z0=±{z0:.1f}  omega={args.omega}"
    )
    print(f"sigma={args.sigma}  lam_target={args.lam_target}  u_max={args.u_max}")

    # ------------------------------------------------------------------
    # IC: two counter-rotating Taylor spheromaks at ±z0
    #
    # lam and u from the Taylor force-free state (shape only, no phi).
    # Phi comes entirely from opposite rigid-body rotation:
    #   spin+1 at +z0: phi_tor ∝ +(-y,x,0)/rho  (CCW)
    #   spin-1 at -z0: phi_tor ∝ -(-y,x,0)/rho  (CW — antiparallel at midplane)
    # This directly creates the antiparallel-field interface at z=0.
    # ------------------------------------------------------------------

    print(f"\n[sph+1] Taylor IC at z=+{z0:.1f}")
    init_spheromak_ic(
        cl,
        R_sph=1.0,
        scale=args.scale,
        B0=1.0,
        lam_target=args.lam_target,
        u_max=args.u_max,
        amplitude=0.0,
        cx=0.0,
        cy=0.0,
        cz=+z0,
        sigma=args.sigma,
    )
    lam1 = np.array(cl.leaf_lam).copy()
    u1 = np.array(cl.leaf_u).copy()

    print(f"[sph-1] Taylor IC at z=-{z0:.1f}")
    init_spheromak_ic(
        cl,
        R_sph=1.0,
        scale=args.scale,
        B0=1.0,
        lam_target=args.lam_target,
        u_max=args.u_max,
        amplitude=0.0,
        cx=0.0,
        cy=0.0,
        cz=-z0,
        sigma=args.sigma,
    )
    lam2 = np.array(cl.leaf_lam).copy()
    u2 = np.array(cl.leaf_u).copy()

    # Additive combine — phi is zero after amplitude=0 calls
    cl.set_leaf_lam(np.minimum(lam1 + lam2, float(LAM_MAX)))
    cl.set_leaf_u(u1 + u2)

    # Opposite spin phi wakes — sequential calls add to existing phi
    print("Applying spin+1 rotation at +z0")
    apply_toroidal_rotation(
        cl,
        centre=[0.0, 0.0, +z0],
        R_core=args.scale,
        amplitude=+args.omega,
        spin_axis="z",
    )
    print("Applying spin-1 rotation at -z0")
    apply_toroidal_rotation(
        cl,
        centre=[0.0, 0.0, -z0],
        R_core=args.scale,
        amplitude=-args.omega,
        spin_axis="z",
    )

    dt = compute_dt(cl)
    phi0 = np.array(cl.leaf_phi)
    lam0 = np.array(cl.leaf_lam)
    print(
        f"\ndt={dt:.4e}  phi_rms0={np.sqrt(np.mean(phi0**2)):.4e}"
        f"  lam_mean={float(lam0.mean()):.4f}  lam_max={float(lam0.max()):.4f}"
    )

    # Diagnostic zones
    dz_mid = z0 / 4.0  # midplane slab: |z| < z0/4
    up_mask = z > z0 * 0.5  # upper spheromak body
    dn_mask = z < -z0 * 0.5  # lower spheromak body
    mid_mask = np.abs(z) < dz_mid

    R_eff = float(r_all.max())
    z_bins = np.linspace(-R_eff, R_eff, 61)

    lam_ic = np.array(cl.leaf_lam)
    print(
        f"mid_cells={mid_mask.sum()}  up_cells={up_mask.sum()}  dn_cells={dn_mask.sum()}"
    )
    print(
        f"IC lam:  up={lam_ic[up_mask].mean():.4f}  mid={lam_ic[mid_mask].mean():.4f}  dn={lam_ic[dn_mask].mean():.4f}"
    )

    n_snaps = args.n_steps // args.report + 1
    J_h = np.zeros(n_snaps)
    p_up_h = np.zeros(n_snaps)
    p_mid_h = np.zeros(n_snaps)
    p_dn_h = np.zeros(n_snaps)
    l_up_h = np.zeros(n_snaps)
    l_mid_h = np.zeros(n_snaps)
    l_dn_h = np.zeros(n_snaps)
    czc_h = np.zeros(n_snaps)
    p_rms_h = np.zeros(n_snaps)

    print(
        f"\n{'step':>7}  {'J_mid':>10}  {'phi_up':>10}  {'phi_mid':>10}  "
        f"{'phi_dn':>10}  {'lam_up':>8}  {'lam_mid':>8}  {'lam_dn':>8}  {'czc':>10}  {'phi_rms':>10}"
    )
    print("  " + "-" * 115)

    si = 0
    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            pa = np.array(cl.leaf_phi)
            la = np.array(cl.leaf_lam)

            J_m = current_sheet_j(pa, z, dz_mid)
            p_up = phi_rms_zone(pa, up_mask)
            p_md = phi_rms_zone(pa, mid_mask)
            p_dn = phi_rms_zone(pa, dn_mask)
            l_up = lam_mean_zone(la, up_mask)
            l_dn = lam_mean_zone(la, dn_mask)
            czc = circ_z_com(pa, pos, z_bins)
            p_rm = float(np.sqrt(np.mean(pa**2)))

            l_md = lam_mean_zone(la, mid_mask)

            J_h[si] = J_m
            p_up_h[si] = p_up
            p_mid_h[si] = p_md
            p_dn_h[si] = p_dn
            l_up_h[si] = l_up
            l_mid_h[si] = l_md
            l_dn_h[si] = l_dn
            czc_h[si] = czc
            p_rms_h[si] = p_rm

            print(
                f"{step:7d}  {J_m:10.4e}  {p_up:10.4e}  {p_md:10.4e}  "
                f"{p_dn:10.4e}  {l_up:8.4f}  {l_md:8.4f}  {l_dn:8.4f}  {czc:+10.5f}  {p_rm:10.4e}",
                flush=True,
            )
            si += 1

        if step < args.n_steps:
            alat.step(0, dt)
            alat.step_sponge(0)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale)
            alat.step_lam(0, dt, args.lam_scale)
            if args.lam_relax_rate > 0.0:
                step_lam_relax(cl, dt, args.lam_relax_rate, args.lam_alpha)

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(n_snaps) * args.report
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    fig.suptitle(
        f"DST counter-rotating spheromak reconnection\n"
        f"R={args.R}  scale={args.scale}  z0=±{z0:.0f}  omega={args.omega}  "
        f"cross={args.cross_scale}",
        fontsize=10,
    )

    axes[0].plot(steps[:si], J_h[:si], color="crimson", lw=1.5)
    axes[0].set_ylabel("|J| midplane\n(div-φ at z≈0)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Current sheet intensity — should peak then drop at reconnection")

    axes[1].plot(steps[:si], p_up_h[:si], color="tab:blue", label="upper (sph+1)")
    axes[1].plot(steps[:si], p_mid_h[:si], color="tab:purple", label="midplane")
    axes[1].plot(steps[:si], p_dn_h[:si], color="tab:orange", label="lower (sph-1)")
    axes[1].set_ylabel("φ rms by zone")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps[:si], l_up_h[:si], color="tab:blue", label="lam upper")
    axes[2].plot(steps[:si], l_mid_h[:si], color="tab:purple", label="lam mid")
    axes[2].plot(steps[:si], l_dn_h[:si], color="tab:orange", label="lam lower")
    axes[2].set_ylabel("mean lam by zone")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(steps[:si], czc_h[:si], color="darkgreen", lw=1.2)
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("circ_z_com\n(azimuthal circulation z-COM)")
    axes[3].set_xlabel("step")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    import os

    os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)
    plt.savefig(args.plot_out, dpi=120)
    print(f"plot → {args.plot_out}")


if __name__ == "__main__":
    main()
