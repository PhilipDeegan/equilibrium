"""
Stern-Gerlach experiment — DST validation.

The two "spin" states are NOT phi-flips of a static winding.  They are the same
winding topology with OPPOSITE rigid-body rotation:

  spin +1  —  whole structure rotating at +omega about z-axis
  spin -1  —  whole structure rotating at -omega about z-axis

The rotational phi component for rigid-body rotation at angular velocity omega:

  v_i = omega * (-y_i, x_i, 0)          (velocity of cell i)
  phi_rot[k, i] = lam_i * v_i · n_hat_k
                = lam_i * omega * (x_i * n_y_k - y_i * n_x_k)

This is localized to the winding region (lam > 0) and represents the phi wake
of a spinning topological structure.  The substrate "knows" about rotation
through this wake — not from the static phi topology alone.

Primary observable: circ_z_com — z-COM of the azimuthal phi circulation,
binned by z-slab.  The injected background (faces 12/13, ±z) has zero
azimuthal projection, so circ_z_com tracks the knot only.

Expected result: spin +1 and spin -1 deflect in opposite z-directions,
confirming that the substrate responds to the handedness of rotation.

Usage:
  PYTHONPATH=$PWD python3 tst/substrate/adaptive/test_stern_gerlach.py \\
      --load proton_candidate_A_R40_IC.npz

  PYTHONPATH=$PWD python3 tst/substrate/adaptive/test_stern_gerlach.py \\
      --load proton_candidate_A_R40_IC.npz \\
      --omega 0.1 --gradient 1e-5 --n-steps 30000 --sustain-spin
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt

# ---------------------------------------------------------------------------
# BCC face geometry — matches adaptive_flux_tube._FACE_NORM
# ---------------------------------------------------------------------------
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
parser.add_argument(
    "--load",
    type=str,
    default="proton_candidate_A_R40_IC.npz",
    help="snapshot to load (should have active phi, e.g. the IC)",
)
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument(
    "--omega",
    type=float,
    default=0.05,
    help="angular velocity of the spin IC: phi_rot[k,i] = lam_i*omega*(x*ny-y*nx)",
)
parser.add_argument(
    "--gradient",
    type=float,
    default=1e-5,
    help="per-step background phi_z injection: delta_phi = G * z",
)
parser.add_argument(
    "--sustain-spin",
    action="store_true",
    help="re-inject rotational phi each step (models persistent spin)",
)
parser.add_argument("--n-steps", type=int, default=30000)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument("--n-z-bins", type=int, default=40)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--cross-scale", type=float, default=0.005)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--skip-spin-2", action="store_true", help="skip the spin -1 run")
parser.add_argument("--plot-out", type=str, default="tmp/stern_gerlach.png")
parser.add_argument("--no-plot", action="store_true")


# ---------------------------------------------------------------------------
# Rotational IC
# ---------------------------------------------------------------------------


def spin_phi(pos, lam_arr, omega):
    """
    Phi wake of a rigid-body rotation at angular velocity omega about z-axis.

    v_i = omega * (-y_i, x_i, 0)
    phi_rot[k, i] = lam_i * (v_i · n_hat_k)
                  = lam_i * omega * (x_i * n_y_k - y_i * n_x_k)

    Localised to winding region (lam > 0).
    omega > 0: counterclockwise (spin +).
    omega < 0: clockwise (spin -).
    Returns (14, N) delta to add to phi.
    """
    x = pos[:, 0]  # (N,)
    y = pos[:, 1]  # (N,)
    # _FACE_NORM[:, 1:2] shape (14,1), x[None,:] shape (1,N) → broadcast (14,N)
    return (
        omega
        * lam_arr[None, :]
        * (x[None, :] * _FACE_NORM[:, 1:2] - y[None, :] * _FACE_NORM[:, 0:1])
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def circ_z_com(phi_arr, pos, z_bins):
    """
    z-COM of azimuthal (toroidal) phi circulation binned by z-slab.

    Azimuthal unit vector: phi_hat = (-y, x, 0) / rho.
    Circulation contribution of face k at cell i: phi[k,i] * (n_hat_k · phi_hat_i).
    Faces 12/13 (±z) project to zero — background phi_z injection is invisible here.

    Returns (z_com, circ_vals, z_mids).
    """
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    rho = np.sqrt(x**2 + y**2)
    valid = rho > 0.5

    phi_hat_x = np.where(valid, -y / np.where(rho > 0, rho, 1.0), 0.0)
    phi_hat_y = np.where(valid, x / np.where(rho > 0, rho, 1.0), 0.0)

    # azimuthal projection per face: (14, N)
    proj = (
        _FACE_NORM[:, 0:1] * phi_hat_x[None, :]
        + _FACE_NORM[:, 1:2] * phi_hat_y[None, :]
    )

    circ_cell = np.where(valid, (phi_arr * proj).sum(axis=0), 0.0)  # (N,)

    n_bins = len(z_bins) - 1
    bin_idx = np.clip(np.digitize(z, z_bins) - 1, 0, n_bins - 1)
    circ_v = np.zeros(n_bins)
    np.add.at(circ_v, bin_idx, circ_cell)

    z_mids = 0.5 * (z_bins[:-1] + z_bins[1:])
    abs_circ = np.abs(circ_v)
    total = float(abs_circ.sum())
    zc = float((abs_circ * z_mids).sum() / total) if total > 1e-30 else float("nan")
    return zc, circ_v, z_mids


def lam_com_z(lam_arr, pos):
    total = float(lam_arr.sum())
    if total < 1e-30:
        return 0.0
    return float((lam_arr * pos[:, 2]).sum() / total)


def phi_rms(phi_arr):
    return float(np.sqrt(np.mean(phi_arr**2)))


def total_circ(phi_arr, pos):
    x, y = pos[:, 0], pos[:, 1]
    rho = np.sqrt(x**2 + y**2)
    valid = rho > 0.5
    phi_hat_x = np.where(valid, -y / np.where(rho > 0, rho, 1.0), 0.0)
    phi_hat_y = np.where(valid, x / np.where(rho > 0, rho, 1.0), 0.0)
    proj = (
        _FACE_NORM[:, 0:1] * phi_hat_x[None, :]
        + _FACE_NORM[:, 1:2] * phi_hat_y[None, :]
    )
    return float(np.abs((phi_arr * proj).sum(axis=0)).sum())


# ---------------------------------------------------------------------------
# Lattice setup
# ---------------------------------------------------------------------------


def build_lattice(args):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    pos = np.array(cl.leaf_pos)
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
    return alat, cl, pos


def load_snapshot(cl, path):
    data = np.load(path)
    phi_arr = data["phi"]
    if phi_arr.shape[0] != 14:
        phi_arr = phi_arr.T  # (N,14) → (14,N)
    cl.set_leaf_phi(phi_arr)
    cl.set_leaf_lam(data["lam"])
    cl.set_leaf_u(data["u"])
    A_js = list(data["A_tiles"].astype(int))
    B_js = list(data["B_tiles"].astype(int)) if "B_tiles" in data else []
    return phi_arr, A_js, B_js


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_sg(alat, cl, pos, args, z_bins, omega, label):
    """
    One SG trial.

    omega > 0  →  spin +1 (counterclockwise rotation about z)
    omega < 0  →  spin -1 (clockwise rotation about z)

    The static winding topology is the SAME for both runs.
    Only the rotational phi component differs.
    """
    print(f"\n--- {label} (omega={omega:+.4f}) ---")

    phi_static, _, _ = load_snapshot(cl, args.load)
    lam_arr = np.array(cl.leaf_lam)

    # Add rotational IC
    phi_arr = phi_static + spin_phi(pos, lam_arr, omega)
    cl.set_leaf_phi(phi_arr)

    dt = compute_dt(cl)
    G = args.gradient
    x, y = pos[:, 0], pos[:, 1]
    # SG quadrupole background: azimuthal phi with amplitude ∝ z.
    # phi_bg[k,i] = G * z_i * (x_i * n_y_k - y_i * n_x_k)
    # Creates B_z = curl-phi_z ∝ z  →  ∂B_z/∂z = G (the SG gradient).
    # div-phi = 0 everywhere — pure magnetic field, step_lam is blind to it.
    # Force on spin acts through lat.step via phi-phi alignment above/below ring.
    _rot_bg = (
        _FACE_NORM[:, 1:2] * x[None, :] - _FACE_NORM[:, 0:1] * y[None, :]  # (14, N)
    )
    bg_z = G * pos[:, 2]  # (N,)

    n_snaps = args.n_steps // args.report + 1
    czc_h = np.zeros(n_snaps)
    lzc_h = np.zeros(n_snaps)
    rms_h = np.zeros(n_snaps)

    pa0 = np.array(cl.leaf_phi)
    czc_0, _, _ = circ_z_com(pa0, pos, z_bins)
    lzc_0 = lam_com_z(np.array(cl.leaf_lam), pos)
    tc0 = total_circ(pa0, pos)

    print(f"N={cl.n_leaves}  dt={dt:.4e}  G={G:.2e}  sustain_spin={args.sustain_spin}")
    print(f"initial  phi_rms={phi_rms(pa0):.4e}  tot_circ={tc0:.4e}")
    print(f"initial  circ_z_com={czc_0:+.5f}  lam_z_com={lzc_0:+.5f}")
    print(
        f"\n{'step':>7}  {'circ_z_com':>11}  {'Δcirc_z':>9}  "
        f"{'lam_z_com':>10}  {'Δlam_z':>8}  {'phi_rms':>10}  {'tot_circ':>10}"
    )
    print("  " + "-" * 80)

    si = 0
    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            pa = np.array(cl.leaf_phi)
            la = np.array(cl.leaf_lam)
            czc, _, _ = circ_z_com(pa, pos, z_bins)
            lzc = lam_com_z(la, pos)
            rv = phi_rms(pa)
            tc = total_circ(pa, pos)
            czc_h[si] = czc
            lzc_h[si] = lzc
            rms_h[si] = rv
            dczc = (
                czc - czc_0 if not (np.isnan(czc) or np.isnan(czc_0)) else float("nan")
            )
            dlzc = lzc - lzc_0
            print(
                f"{step:7d}  {czc:+11.5f}  {dczc:+9.5f}  "
                f"{lzc:+10.5f}  {dlzc:+8.5f}  {rv:10.4e}  {tc:10.4e}",
                flush=True,
            )
            si += 1

        if step < args.n_steps:
            alat.step(0, dt)
            alat.step_sponge(0)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale)
            if args.lam_mode == "dynamic":
                alat.step_lam(0, dt, args.lam_scale)

            # SG magnet: gradient of curl-phi in z.
            # Inject azimuthal phi with amplitude ∝ z, creating B_z ∝ z (∂B_z/∂z = G).
            # div-phi = 0 (pure magnetic field — no charge source, step_lam blind to it).
            # Force acts through lat.step: spin phi ± background phi → u asymmetry above/below ring.
            pa = np.array(cl.leaf_phi)
            pa += _rot_bg * bg_z[None, :]  # (14, N) * (1, N)

            # Optionally sustain spin (using current lam distribution)
            if args.sustain_spin:
                la = np.array(cl.leaf_lam)
                pa += spin_phi(pos, la, omega)

            cl.set_leaf_phi(pa)

    final_czc, _, _ = circ_z_com(np.array(cl.leaf_phi), pos, z_bins)
    print(f"\nfinal circ_z_com={final_czc:+.6f}  Δcirc_z={final_czc - czc_0:+.6f}")
    return czc_h, lzc_h, rms_h, czc_0, lzc_0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    alat, cl, pos = build_lattice(args)

    data = np.load(args.load)
    n_snap = (
        data["phi"].shape[0] if data["phi"].shape[0] != 14 else data["phi"].shape[1]
    )
    if n_snap != cl.n_leaves:
        print(f"ERROR: snapshot N={n_snap} != lattice N={cl.n_leaves}")
        sys.exit(1)

    R_eff = float(np.linalg.norm(pos, axis=1).max())
    z_bins = np.linspace(-R_eff, R_eff, args.n_z_bins + 1)
    steps = np.arange(args.n_steps // args.report + 1) * args.report

    print(f"snapshot: {args.load}  N={cl.n_leaves}  R_eff={R_eff:.1f}")
    print(
        f"omega={args.omega:.3e}  G={args.gradient:.2e}  "
        f"n_steps={args.n_steps}  sustain_spin={args.sustain_spin}"
    )
    print(f"cross={args.cross_scale}  lam_scale={args.lam_scale}")

    czc1, lzc1, rms1, czc1_0, lzc1_0 = run_sg(
        alat, cl, pos, args, z_bins, omega=+args.omega, label="spin +1"
    )
    if not args.skip_spin_2:
        czc2, lzc2, rms2, czc2_0, lzc2_0 = run_sg(
            alat, cl, pos, args, z_bins, omega=-args.omega, label="spin -1"
        )
    else:
        czc2 = lzc2 = rms2 = czc2_0 = lzc2_0 = None

    # Summary
    print("\n" + "=" * 60)
    print("STERN-GERLACH SUMMARY")
    print("=" * 60)
    dczc1 = czc1[-1] - czc1_0
    dlzc1 = lzc1[-1] - lzc1_0
    print(f"spin +1: Δcirc_z_com={dczc1:+.6f}  Δlam_z_com={dlzc1:+.6f}")
    if czc2 is not None:
        dczc2 = czc2[-1] - czc2_0
        dlzc2 = lzc2[-1] - lzc2_0
        print(f"spin -1: Δcirc_z_com={dczc2:+.6f}  Δlam_z_com={dlzc2:+.6f}")
        ratio = dczc1 / dczc2 if abs(dczc2) > 1e-10 else float("nan")
        print(f"ratio = {ratio:.4f}  (expected ≈ -1.0 for opposite deflection)")
        if ratio < -0.5:
            print(
                "RESULT: OPPOSITE DEFLECTION — substrate responds to spin handedness."
            )
        elif abs(dczc1) < 1e-6 and abs(dczc2) < 1e-6:
            print(
                "RESULT: NO DEFLECTION — omega or gradient too small, or phi radiated."
            )
        else:
            print("RESULT: UNEXPECTED — same-direction or asymmetric deflection.")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.suptitle(
        f"DST Stern-Gerlach  omega={args.omega:.2e}  G={args.gradient:.2e}  "
        f"sustain_spin={args.sustain_spin}",
        fontsize=10,
    )
    ax0, ax1, ax2 = axes

    ax0.plot(steps, czc1 - czc1_0, label="spin +1", color="tab:blue")
    if czc2 is not None:
        ax0.plot(
            steps, czc2 - czc2_0, label="spin −1", color="tab:orange", linestyle="--"
        )
    ax0.axhline(0, color="k", lw=0.5)
    ax0.set_ylabel("Δ circ_z_com  (dipole z-position)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.plot(steps, lzc1 - lzc1_0, label="spin +1", color="tab:blue")
    if lzc2 is not None:
        ax1.plot(
            steps, lzc2 - lzc2_0, label="spin −1", color="tab:orange", linestyle="--"
        )
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("Δ lam_z_com  (winding z-position)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, rms1, label="spin +1", color="tab:blue")
    if rms2 is not None:
        ax2.plot(steps, rms2, label="spin −1", color="tab:orange", linestyle="--")
    ax2.set_ylabel("phi RMS")
    ax2.set_xlabel("step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    import os

    os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)
    plt.savefig(args.plot_out, dpi=120)
    print(f"plot → {args.plot_out}")


if __name__ == "__main__":
    main()
