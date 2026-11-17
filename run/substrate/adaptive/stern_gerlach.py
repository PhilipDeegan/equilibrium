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

2026-09-04 FINDING (see doc/mems/project_dst_em_field_convention_crisis.md
for full derivation chain): the raw circ_z_com comparison between the two
spin runs is USELESS on its own -- the injected "SG magnet" background
(`_rot_bg * bg_z`) is large and completely spin-independent, and since the
core step()/step_sponge() dynamics are LINEAR, it dominates and can hide a
real spin-dependent signal underneath it. Isolating the signal (subtract a
background-only omega=0 control run's circ_z_com from each spin run,
computed automatically now unless --skip-bg-control) reveals a real,
near-exactly antisymmetric response to spin sign (ratio 0.99-1.00) WHILE
the winding's own circulation is still substantially intact. That signal
degrades once tot_circ has radiated away most of its initial value
(ordinary decay, unrelated to spin) -- by the time the OLD script compared
raw values at a fixed late step (e.g. step 30000 of a 30000-step run),
tot_circ had typically dropped below 1% of its initial value, so that
comparison was measuring post-decay noise, not the effect. The 2026-09-01
session's "clean null result" was this measurement artifact, not evidence
against a real chirality-dependent coupling. --circ-retain-threshold
(default 0.5) now controls which step the isolated ratio is evaluated at:
the LAST step where the background-control run's tot_circ is still >= that
fraction of its initial value, rather than a fixed late step.

Separately (also 2026-09-04, do not re-derive): `step_cross` and MHD-style
flux-freezing/advection mechanisms are NOT relevant to this experiment's
"SG magnet" or to magnetism/chirality generally at particle scale --
`step_cross` only concerns whether a multi-cell aggregate can bulk-
translate (irrelevant here, the winding isn't translating), and MHD is a
macro/continuum-fluid approximation, wrong scale category for a single
particle's own magnetic moment. `lam` itself should NOT be made
chiral/directional -- it is the throughput modifier (sigma=1/(1+lam),
gravity/time-dilation), radial and scalar by design. The directionality
that matters here already lives entirely in `phi` (spin_phi's own
rotational wake).

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/stern_gerlach.py

  PYTHONPATH=$PWD python3 run/substrate/adaptive/stern_gerlach.py \\
      --omega 0.1 --gradient 1e-5 --n-steps 30000 --sustain-spin

--load defaults to a name derived from --R/--sph-frac/--amplitude/--lam-target and is
built fresh (analytic Taylor spheromak IC, see build_and_save_ic) if that file doesn't
exist yet -- pass --load explicitly to reuse/share a specific snapshot.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from run.substrate.adaptive.probe_decay_compare import plot_snapshots
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import init_spheromak_ic

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
    default=None,
    help="snapshot to load (should have active phi, e.g. the IC); "
    "default derives a name from --R/--sph-frac/--amplitude/--lam-target and builds "
    "it fresh if it doesn't exist yet",
)
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument(
    "--amplitude", type=float, default=1.0, help="winding amplitude for a freshly built IC"
)
parser.add_argument(
    "--sph-frac",
    type=float,
    default=0.06,
    help="spheromak radius as fraction of R (scale = R * sph_frac) -- same "
    "scale-invariant convention as run/substrate/adaptive/neutron_decay.py; "
    "doubling this doubles the physical structure size, unlike the old "
    "hop-count ring geometry it replaced (find_linked_loop only found a "
    "valid Hopf-linked pair for the minimal 8-tile ring -- its search was "
    "anchored to ring A's centroid, not a point on its own path, so a "
    "scaled-up ring's small linked loop no longer threaded it at all)",
)
parser.add_argument("--B0", type=float, default=1.0, help="field amplitude for a freshly built IC")
parser.add_argument(
    "--lam-target", type=float, default=1.0, help="lam ∝ |B| peak for a freshly built IC"
)
parser.add_argument("--u-max", type=float, default=1.0, help="u ∝ |B| peak for a freshly built IC")
parser.add_argument(
    "--sigma",
    type=float,
    default=None,
    help="Gaussian taper width in spheromak units (None=hard cutoff at R_sph)",
)
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
    "--bg-mode",
    choices=["accumulate", "rate", "once"],
    default="accumulate",
    help="accumulate (original/buggy): adds G*z every raw step with no dt "
    "scaling and nothing removing it -- unbounded, confirmed to blow up "
    "regardless of cross_scale/lam_mode. rate: dt-scales the same term "
    "(dt*G*z per step) so it represents a proper rate. once: adds G*z a "
    "single time (alongside spin_phi, before stepping starts) and never "
    "again -- no per-step injection at all.",
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
parser.add_argument(
    "--skip-bg-control",
    action="store_true",
    help="skip the omega=0 background-only control run. Without it, the raw "
    "circ_z_com comparison is dominated by the injected background (see "
    "module docstring, 2026-09-04 finding) -- only skip this for a quick "
    "smoke test, not for a real result.",
)
parser.add_argument(
    "--circ-retain-threshold",
    type=float,
    default=0.5,
    help="isolated delta ratio is evaluated at the LAST step where tot_circ "
    "(background-control run) is still >= this fraction of its initial "
    "value -- the chirality signal is only expected to be clean while the "
    "spin structure itself hasn't substantially decayed away (2026-09-04 "
    "finding: signal was clean while tot_circ>=~14%%, degraded once most "
    "of it had radiated).",
)
parser.add_argument("--plot-out", type=str, default="tmp/stern_gerlach.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument(
    "--data-out",
    type=str,
    default=None,
    help="save all time-series diagnostics (steps, circ_z_com/lam_z_com/"
    "phi_rms/tot_circ for each run, isolated deltas, eval step) as an .npz "
    "so plots can be regenerated post-hoc (see --plot-from-data) without "
    "rerunning the sim",
)
parser.add_argument(
    "--plot-from-data",
    type=str,
    default=None,
    help="skip the simulation entirely -- load a previously-saved --data-out "
    "npz and (re)plot it to --plot-out",
)


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


def build_and_save_ic(cl, args):
    """Build a Taylor spheromak IC (analytic, scale-invariant -- see
    tst.substrate.adaptive.adaptive_flux_tube.init_spheromak_ic, same
    convention as run/substrate/adaptive/neutron_decay.py's sph_frac) directly
    on cl and save it to args.load, for when that snapshot doesn't exist yet.
    Replaces the old minimal-8-tile-ring + 4-tile-plaquette Hopf-link
    construction, which had no working way to scale up (see --sph-frac
    help)."""
    scale = args.R * args.sph_frac
    b_max = init_spheromak_ic(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=args.B0,
        lam_target=args.lam_target,
        u_max=args.u_max,
        amplitude=args.amplitude,
        sigma=args.sigma,
    )
    np.savez(
        args.load,
        phi=np.array(cl.leaf_phi).T,
        lam=np.array(cl.leaf_lam),
        u=np.array(cl.leaf_u),
    )
    print(
        f"no snapshot at {args.load!r} — built a Taylor spheromak IC "
        f"(scale={scale:.2f}, b_max={b_max:.3f}) and saved it there"
    )


def load_snapshot(cl, path):
    data = np.load(path)
    phi_arr = data["phi"]
    if phi_arr.shape[0] != 14:
        phi_arr = phi_arr.T  # (N,14) → (14,N)
    cl.set_leaf_phi(phi_arr)
    cl.set_leaf_lam(data["lam"])
    cl.set_leaf_u(data["u"])
    return phi_arr


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

    phi_static = load_snapshot(cl, args.load)
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

    if args.bg_mode == "once":
        phi_arr = np.array(cl.leaf_phi) + _rot_bg * bg_z[None, :]
        cl.set_leaf_phi(phi_arr)

    n_snaps = args.n_steps // args.report + 1
    czc_h = np.zeros(n_snaps)
    lzc_h = np.zeros(n_snaps)
    rms_h = np.zeros(n_snaps)
    tc_h = np.zeros(n_snaps)

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
            tc_h[si] = tc
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
            if args.bg_mode in ("accumulate", "rate") or args.sustain_spin:
                pa = np.array(cl.leaf_phi)
                if args.bg_mode == "accumulate":
                    pa += _rot_bg * bg_z[None, :]  # (14, N) * (1, N)
                elif args.bg_mode == "rate":
                    pa += dt * _rot_bg * bg_z[None, :]

                # Optionally sustain spin (using current lam distribution)
                if args.sustain_spin:
                    la = np.array(cl.leaf_lam)
                    pa += spin_phi(pos, la, omega)

                cl.set_leaf_phi(pa)

    final_czc, _, _ = circ_z_com(np.array(cl.leaf_phi), pos, z_bins)
    print(f"\nfinal circ_z_com={final_czc:+.6f}  Δcirc_z={final_czc - czc_0:+.6f}")
    final_state = dict(
        phi=np.array(cl.leaf_phi).T,  # (N, 14)
        u=np.array(cl.leaf_u),
        lam=np.array(cl.leaf_lam),
    )
    return czc_h, lzc_h, rms_h, tc_h, czc_0, lzc_0, final_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    if args.load is None:
        args.load = (
            f"stern_gerlach_ic_R{args.R:g}_sphfrac{args.sph_frac:g}"
            f"_amp{args.amplitude:g}_lam{args.lam_target:g}.npz"
        )

    alat, cl, pos = build_lattice(args)

    if not os.path.exists(args.load):
        build_and_save_ic(cl, args)

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

    czc_bg = tc_bg = final_bg = None
    if not args.skip_bg_control:
        czc_bg, lzc_bg, rms_bg, tc_bg, czc_bg_0, lzc_bg_0, final_bg = run_sg(
            alat, cl, pos, args, z_bins, omega=0.0, label="background only (omega=0)"
        )

    czc1, lzc1, rms1, tc1, czc1_0, lzc1_0, final1_state = run_sg(
        alat, cl, pos, args, z_bins, omega=+args.omega, label="spin +1"
    )
    if not args.skip_spin_2:
        czc2, lzc2, rms2, tc2, czc2_0, lzc2_0, final2_state = run_sg(
            alat, cl, pos, args, z_bins, omega=-args.omega, label="spin -1"
        )
    else:
        czc2 = lzc2 = rms2 = tc2 = lzc2_0 = final2_state = None

    # Summary
    print("\n" + "=" * 60)
    print("STERN-GERLACH SUMMARY")
    print("=" * 60)
    final1 = czc1[-1]
    dlzc1 = lzc1[-1] - lzc1_0
    print(f"spin +1: final circ_z_com={final1:+.6f}  Δlam_z_com={dlzc1:+.6f}")

    delta1 = delta2 = None
    if czc2 is not None:
        final2 = czc2[-1]
        dlzc2 = lzc2[-1] - lzc2_0
        print(f"spin -1: final circ_z_com={final2:+.6f}  Δlam_z_com={dlzc2:+.6f}")
        raw_ratio = final1 / final2 if abs(final2) > 1e-10 else float("nan")
        print(
            f"\n[reference only -- 2026-09-04 finding: this raw comparison is "
            f"dominated by the injected background and is unreliable, see "
            f"module docstring]  raw ratio = {raw_ratio:.4f}"
        )

        if czc_bg is not None:
            # isolate the spin-dependent signal from the (large, spin-independent)
            # injected background -- see 2026-09-04 finding in module docstring.
            delta1 = czc1 - czc_bg
            delta2 = czc2 - czc_bg

            tc0 = tc_bg[0] if tc_bg[0] > 0 else 1.0
            retained = tc_bg / tc0
            valid = np.nonzero(retained >= args.circ_retain_threshold)[0]
            eval_idx = int(valid[-1]) if len(valid) else 0
            eval_step = steps[eval_idx]

            d1, d2 = delta1[eval_idx], delta2[eval_idx]
            iso_ratio = d1 / d2 if abs(d2) > 1e-10 else float("nan")
            print(
                f"\nISOLATED signal (background-subtracted), evaluated at step "
                f"{eval_step} -- the last step with tot_circ >= "
                f"{args.circ_retain_threshold:.0%} of its initial value "
                f"(retained={retained[eval_idx]:.1%}):"
            )
            print(f"  Δ(spin+1) = {d1:+.6f}   Δ(spin-1) = {d2:+.6f}")
            print(f"  isolated ratio = {iso_ratio:.4f}  (expected ≈ -1.0 for opposite deflection)")
            if iso_ratio < -0.5:
                print(
                    "RESULT: OPPOSITE DEFLECTION (isolated) — substrate responds to spin handedness."
                )
            elif abs(d1) < 1e-6 and abs(d2) < 1e-6:
                print("RESULT: NO DEFLECTION (isolated) — no signal above noise floor.")
            else:
                print("RESULT: UNEXPECTED (isolated) — same-direction or asymmetric deflection.")
        else:
            print(
                "\n[--skip-bg-control was set -- no isolated verdict available; "
                "the raw ratio above is NOT a reliable result on its own]"
            )

    if args.data_out:
        payload = dict(
            steps=steps, pos=pos,
            czc1=czc1, rms1=rms1, tc1=tc1, lzc1=lzc1, lzc1_0=lzc1_0,
            phi1=final1_state["phi"], u1=final1_state["u"], lam1=final1_state["lam"],
            omega=args.omega, gradient=args.gradient, sustain_spin=args.sustain_spin,
            R=args.R, sph_frac=args.sph_frac,
        )
        if czc2 is not None:
            payload.update(
                czc2=czc2, rms2=rms2, tc2=tc2, lzc2=lzc2, lzc2_0=lzc2_0,
                phi2=final2_state["phi"], u2=final2_state["u"], lam2=final2_state["lam"],
            )
        if czc_bg is not None:
            payload.update(
                czc_bg=czc_bg, rms_bg=rms_bg, tc_bg=tc_bg, lzc_bg=lzc_bg, lzc_bg_0=lzc_bg_0,
                phi_bg=final_bg["phi"], u_bg=final_bg["u"], lam_bg=final_bg["lam"],
            )
        if delta1 is not None:
            payload.update(delta1=delta1, delta2=delta2, eval_step=eval_step, iso_ratio=iso_ratio)
        os.makedirs(os.path.dirname(args.data_out) or ".", exist_ok=True)
        np.savez(args.data_out, **payload)
        print(f"data → {args.data_out}")

    if not args.no_plot:
        make_plot(
            steps, czc1, czc2, czc_bg, lzc1, lzc2, rms1, rms2, delta1, delta2,
            eval_step if delta1 is not None else None,
            args.omega, args.gradient, args.sustain_spin, args.plot_out,
        )
        plot_field_snapshots(
            pos, final1_state, final2_state, final_bg,
            ring_r=args.R * args.sph_frac,
            plot_dir=os.path.dirname(args.plot_out) or "tmp",
        )


def make_plot(steps, czc1, czc2, czc_bg, lzc1, lzc2, rms1, rms2, delta1, delta2,
              eval_step, omega, gradient, sustain_spin, plot_out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = 4 if delta1 is not None else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3 * n_rows), sharex=True)
    fig.suptitle(
        f"DST Stern-Gerlach  omega={omega:.2e}  G={gradient:.2e}  "
        f"sustain_spin={sustain_spin}",
        fontsize=10,
    )
    ax0, ax1, ax2 = axes[:3]

    ax0.plot(steps, czc1, label="spin +1", color="tab:blue")
    if czc2 is not None:
        ax0.plot(steps, czc2, label="spin −1", color="tab:orange", linestyle="--")
    if czc_bg is not None:
        ax0.plot(steps, czc_bg, label="background only", color="tab:gray", linestyle=":")
    ax0.axhline(0, color="k", lw=0.5)
    ax0.set_ylabel("circ_z_com  (raw, background-dominated)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.plot(steps, lzc1, label="spin +1", color="tab:blue")
    if lzc2 is not None:
        ax1.plot(steps, lzc2, label="spin −1", color="tab:orange", linestyle="--")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("lam_z_com  (winding z-COM)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, rms1, label="spin +1", color="tab:blue")
    if rms2 is not None:
        ax2.plot(steps, rms2, label="spin −1", color="tab:orange", linestyle="--")
    ax2.set_ylabel("phi RMS")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if delta1 is not None:
        ax3 = axes[3]
        ax3.plot(steps, delta1, label="Δ(spin+1) = circ_z_com − background", color="tab:blue")
        ax3.plot(steps, delta2, label="Δ(spin-1) = circ_z_com − background", color="tab:orange", linestyle="--")
        ax3.axhline(0, color="k", lw=0.5)
        ax3.axvline(eval_step, color="tab:red", lw=1, linestyle=":", label=f"eval step ({eval_step})")
        ax3.set_ylabel("isolated Δcirc_z_com")
        ax3.set_xlabel("step")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax2.set_xlabel("step")

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_out) or ".", exist_ok=True)
    plt.savefig(plot_out, dpi=120)
    plt.close(fig)
    print(f"plot → {plot_out}")


def plot_field_snapshots(pos, final1_state, final2_state, final_bg, ring_r, plot_dir):
    """
    Real spatial cross-section plots (|phi|, phi_tor, phi_pol, u) of the
    final lattice state for each run, via the same plot_snapshots used by
    neutron_decay.py -- these show the actual field, not scalar diagnostics.
    """
    snaps = [dict(pos=pos, label="spin +1", ring_r=ring_r, **final1_state)]
    if final2_state is not None:
        snaps.append(dict(pos=pos, label="spin −1", ring_r=ring_r, **final2_state))
    if final_bg is not None:
        snaps.append(dict(pos=pos, label="background only", ring_r=ring_r, **final_bg))
    plot_snapshots(snaps, zoom=max(4.0 * ring_r, 10.0), plot_dir=plot_dir, tag="sg_final")


def plot_from_data(data_path, plot_out):
    """Re-render the plots from a --data-out npz, no simulation."""
    d = np.load(data_path)

    def _get(name):
        return d[name] if name in d.files else None

    make_plot(
        d["steps"], d["czc1"], _get("czc2"), _get("czc_bg"),
        d["lzc1"], _get("lzc2"), d["rms1"], _get("rms2"),
        _get("delta1"), _get("delta2"),
        int(d["eval_step"]) if "eval_step" in d.files else None,
        float(d["omega"]), float(d["gradient"]), bool(d["sustain_spin"]), plot_out,
    )

    if "phi1" in d.files:
        final1 = dict(phi=d["phi1"], u=d["u1"], lam=d["lam1"])
        final2 = (
            dict(phi=d["phi2"], u=d["u2"], lam=d["lam2"]) if "phi2" in d.files else None
        )
        final_bg = (
            dict(phi=d["phi_bg"], u=d["u_bg"], lam=d["lam_bg"])
            if "phi_bg" in d.files
            else None
        )
        plot_field_snapshots(
            d["pos"], final1, final2, final_bg,
            ring_r=float(d["R"]) * float(d["sph_frac"]),
            plot_dir=os.path.dirname(plot_out) or "tmp",
        )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.plot_from_data:
        plot_from_data(args.plot_from_data, args.plot_out)
    else:
        main(args)
