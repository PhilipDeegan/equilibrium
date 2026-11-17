"""
Electron/positron collision -- mirrored-winding double-ring dipoles under
the same step_cross v2 barrier gate calibrated for R1 (electron must not
erode) / R2 (neutron must decay) in probe_cross_v2_gates.py.

The "positron" here is just init_electron_dipole_ic with amplitude<0 --
the same poloidal-only Taylor-state construction, mirrored in phi sign,
placed at -z0 while the electron sits at +z0. Both have lam=0, u uniform
(zero-net-force), so with cross_scale=0 this is pure linear superposition
(same conclusion as probe_u_superposition.py, for phi instead of u): two
separate loops evolve independently and cannot merge/annihilate without an
irreversible topology-changing operator. cross_scale>0 (the barrier gate)
is what can actually let their windings interact/reconnect at the midplane
interface -- this script's real question.

Diagnostics:
  J_mid       : current-sheet intensity at the midplane (reconnection proxy)
  phi_rms     : by zone (upper/mid/lower)
  surv_e/surv_p : each object's own poloidal winding amplitude (face_flux_
                  geometry + phi_poloidal_scan, centred on its own r_axis),
                  normalised to its own t=0 value -- same R1-style survival
                  measure used in probe_cross_v2_gates.py.

Resume: auto-detects the highest --bk-dir/{save-prefix}NNN.hdf5 snapshot
(same pattern as exp/2026/08/10/19_24_l0_l1_boundary_error/step_l0_l1_
boundary.py) and, if found, loads it and continues for --n-steps MORE
steps instead of rebuilding the IC from t=0. History (w0_e/w0_p + the
prior step-by-step series) is pulled from the matching {out-dir}/step
{N}.npz and prepended to the new segment before saving/plotting, so the
final series always covers 0..(resume_step + n_steps). Falls back to a
fresh t=0 build if no snapshot is found. --no-resume forces a fresh
build even if a snapshot exists.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/electron_positron_collision.py
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/electron_positron_collision.py --cross-scale 0.0 --no-plot
  # continue an existing run for 1000 more steps (auto-resumes from the latest checkpoint):
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/electron_positron_collision.py --n-steps 1000 \
      --data-out exp/.../step2000.npz --save-out /bk/dst/electron_positron_step2000.hdf5
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from run.substrate.adaptive.cross_v2 import CrossV2
from run.substrate.adaptive.counter_rotating_spheromak_reconnection import (
    current_sheet_j,
    phi_rms_zone,
)
from run.substrate.adaptive.probe_decay_compare import _poloidal_phi
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_electron_dipole_ic,
    phi_poloidal_scan,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=60.0)
parser.add_argument(
    "--scale", type=float, default=8.0, help="DST units per dipole unit"
)
parser.add_argument("--z-sep", type=float, default=20.0)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--m0", type=float, default=2.49, help="barrier gate m0")
parser.add_argument(
    "--gate",
    type=str,
    default="barrier",
    help="CrossV2 gate: v1|shear|product|opposed|curl|barrier",
)
parser.add_argument(
    "--ref",
    type=float,
    default=1e-4,
    help="saturation reference for shear/product/opposed/curl gates",
)
parser.add_argument(
    "--same-sign",
    action="store_true",
    help="both objects electrons (same winding sign, repulsion test) instead of electron/positron (opposite sign, attraction test)",
)
parser.add_argument("--cross-scale", type=float, default=0.1)
parser.add_argument(
    "--lam-target",
    type=float,
    default=0.0,
    help="peak lam (lam = lam_target * |B|/|B|_max, same B that sources phi) -- default 0 preserves the pure-phi-winding baseline",
)
parser.add_argument(
    "--n-steps",
    type=int,
    default=3000,
    help="steps to run -- ADDITIONAL steps beyond the resume point when resuming",
)
parser.add_argument("--report", type=int, default=100)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument(
    "--plot-out", type=str, default="tmp/electron_positron_collision.png"
)
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--data-out", type=str, default=None)
parser.add_argument(
    "--save-out",
    type=str,
    default=None,
    help="native HDF5 final-state save (AdaptiveLattice.save)",
)
parser.add_argument(
    "--bk-dir",
    type=str,
    default="/bk/dst",
    help="directory to scan for resumable snapshots",
)
parser.add_argument(
    "--save-prefix",
    type=str,
    default="electron_positron_step",
    help="snapshot filename prefix: {prefix}{N}.hdf5",
)
parser.add_argument(
    "--no-resume",
    action="store_true",
    help="always rebuild the IC from t=0, even if a snapshot exists",
)


def _r_axis(R_sph, scale):
    """Centreline circle radius -- pure geometry, independent of the cluster
    state, matching the formula inside init_electron_dipole_ic exactly."""
    from scipy.special import spherical_jn

    k_ = 4.493 / R_sph
    rho_g = np.linspace(1e-4, R_sph, 2048)
    psi = rho_g * spherical_jn(1, k_ * rho_g)
    return float(rho_g[np.argmax(np.abs(psi))]) * scale


def _find_latest_snapshot(bk_dir, prefix):
    pat = re.compile(re.escape(prefix) + r"(\d+)\.hdf5$")
    best_n, best_path = -1, None
    for path in glob.glob(os.path.join(bk_dir, f"{prefix}*.hdf5")):
        m = pat.search(path)
        if m:
            n = int(m.group(1))
            if n > best_n:
                best_n, best_path = n, path
    return (best_n, best_path) if best_path else (None, None)


def main():
    args = parser.parse_args()

    resume_step, snap_path = (
        (None, None)
        if args.no_resume
        else _find_latest_snapshot(args.bk_dir, args.save_prefix)
    )

    z0 = args.z_sep / 2.0
    r_axis_e = r_axis_p = _r_axis(1.0, args.scale)
    dz_mid = z0 / 4.0
    s_scan = tuple(f * args.scale for f in (0.5, 1.0, 1.5))

    if resume_step is not None:
        prior_npz = os.path.join(
            os.path.dirname(args.data_out), f"step{resume_step}.npz"
        )
        if not os.path.isfile(prior_npz):
            raise FileNotFoundError(
                f"found snapshot {snap_path} for step {resume_step} but no matching history at {prior_npz}"
            )
        prior = dict(np.load(prior_npz))
        print(
            f"resuming from {snap_path}  (step={resume_step})  history <- {prior_npz}"
        )

        alat = _lat.AdaptiveLattice.load(snap_path)
        alat.set_n_threads(args.n_threads)
        cl = alat[0]

        pos = np.array(cl.leaf_pos)
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

        w0_e = float(prior["w0_e"])
        w0_p = float(prior["w0_p"])
        print(
            f"N={cl.n_leaves}  R(inferred)={r_max:.1f}  scale={args.scale}  z0=+/-{z0:.1f}"
            f"  m0={args.m0}  cross_scale={args.cross_scale}  w0_e={w0_e:.4e}  w0_p={w0_p:.4e}"
        )
    else:
        prior = None
        resume_step = 0

        alat = _lat.AdaptiveLattice()
        alat.add_cluster()
        alat.set_n_threads(args.n_threads)
        cl = alat[0]
        _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
        alat.rebuild(0)

        pos = np.array(cl.leaf_pos)
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

        print(
            f"N={cl.n_leaves}  R={args.R}  scale={args.scale}  z0=+/-{z0:.1f}"
            f"  m0={args.m0}  cross_scale={args.cross_scale}  lam_target={args.lam_target}"
        )

        print(f"\n[electron] dipole IC at z=+{z0:.1f}")
        _, r_axis_e = init_electron_dipole_ic(
            cl,
            R_sph=1.0,
            scale=args.scale,
            B0=args.B0,
            lam_target=args.lam_target,
            u_bg=0.0,
            amplitude=+1.0,
            cx=0.0,
            cy=0.0,
            cz=+z0,
        )
        phi_e = np.array(cl.leaf_phi).copy()
        lam_e = np.array(cl.leaf_lam).copy()

        species2 = "electron" if args.same_sign else "positron"
        print(
            f"[{species2}] {'same-sign' if args.same_sign else 'mirrored'} dipole IC at z=-{z0:.1f}"
        )
        # NOTE: init_electron_dipole_ic only builds/stores phi inside its own
        # `if amplitude > 0.0:` guard -- passing amplitude<0 skips that block
        # entirely and silently leaves cl.leaf_phi as whatever the previous
        # call left it (the electron's own field). Build with amplitude=+1 so
        # the object is actually constructed and divergence-cleaned, then
        # mirror it by negating the captured array in Python (skipped for
        # --same-sign, giving two electrons -- a repulsion test instead of
        # the default electron/positron attraction test). lam is set via
        # a full overwrite too (cluster.set_leaf_lam), same trap -- capture
        # and re-sum both, same as phi (lam is |B|-derived so unsigned, no
        # negation needed either way).
        _, r_axis_p = init_electron_dipole_ic(
            cl,
            R_sph=1.0,
            scale=args.scale,
            B0=args.B0,
            lam_target=args.lam_target,
            u_bg=0.0,
            amplitude=+1.0,
            cx=0.0,
            cy=0.0,
            cz=-z0,
        )
        phi_p = (
            np.array(cl.leaf_phi).copy()
            if args.same_sign
            else -np.array(cl.leaf_phi).copy()
        )
        lam_p = np.array(cl.leaf_lam).copy()

        cl.set_leaf_phi(np.ascontiguousarray(phi_e + phi_p))
        cl.set_leaf_lam(np.ascontiguousarray(lam_e + lam_p))
        cl.set_leaf_u(np.zeros(cl.n_leaves))

    dt = compute_dt(cl)
    op = CrossV2(cl, gate=args.gate, m0=args.m0, ref=args.ref)

    up_mask = z > z0 * 0.5
    dn_mask = z < -z0 * 0.5
    mid_mask = np.abs(z) < dz_mid

    def winding(centre, r_axis):
        mid, n_hat, phi_f = face_flux_geometry(cl, centre)
        pol = phi_poloidal_scan(mid, n_hat, phi_f, r_axis, s_scan)
        return float(np.abs(pol).max())

    if prior is None:
        w0_e = winding([0, 0, +z0], r_axis_e)
        w0_p = winding([0, 0, -z0], r_axis_p)
        print(f"w0_e={w0_e:.4e}  w0_p={w0_p:.4e}  dt={dt:.4e}")

    n_new_snaps = args.n_steps // args.report + (1 if prior is None else 0)
    J_h = np.zeros(n_new_snaps)
    p_up_h = np.zeros(n_new_snaps)
    p_mid_h = np.zeros(n_new_snaps)
    p_dn_h = np.zeros(n_new_snaps)
    surv_e_h = np.zeros(n_new_snaps)
    surv_p_h = np.zeros(n_new_snaps)
    H_h = np.zeros(n_new_snaps)
    H_up_h = np.zeros(n_new_snaps)
    H_dn_h = np.zeros(n_new_snaps)
    zcom_up_h = np.zeros(n_new_snaps)
    zcom_dn_h = np.zeros(n_new_snaps)

    print(
        f"\n{'step':>7}  {'J_mid':>10}  {'phi_up':>10}  {'phi_mid':>10}"
        f"  {'phi_dn':>10}  {'surv_e':>8}  {'surv_p':>8}  {'H_total':>12}"
        f"  {'dz_up':>9}  {'dz_dn':>9}"
    )
    print("  " + "-" * 112)

    def measure():
        ua = np.array(cl.leaf_u)
        pa = np.array(cl.leaf_phi)
        J_m = current_sheet_j(pa, z, dz_mid)
        p_up = phi_rms_zone(pa, up_mask)
        p_md = phi_rms_zone(pa, mid_mask)
        p_dn = phi_rms_zone(pa, dn_mask)
        se = winding([0, 0, +z0], r_axis_e) / max(w0_e, 1e-30)
        sp = winding([0, 0, -z0], r_axis_p) / max(w0_p, 1e-30)
        H_cell = ua**2 + 0.5 * np.sum(pa**2, axis=0)
        H_tot = float(H_cell.sum())
        H_up = float(H_cell[up_mask].sum())
        H_dn = float(H_cell[dn_mask].sum())
        # energy-weighted centroid z of each object's own field -- the actual
        # translation diagnostic (attraction/repulsion), independent of lam
        # (lam=0 by default here, unlike A2's weighted_com which used lam).
        zcom_up = (
            float(np.average(z[up_mask], weights=H_cell[up_mask])) if H_up > 0 else z0
        )
        zcom_dn = (
            float(np.average(z[dn_mask], weights=H_cell[dn_mask])) if H_dn > 0 else -z0
        )
        return J_m, p_up, p_md, p_dn, se, sp, H_tot, H_up, H_dn, zcom_up, zcom_dn

    si = 0
    if prior is None:
        vals = measure()
        (
            J_h[si],
            p_up_h[si],
            p_mid_h[si],
            p_dn_h[si],
            surv_e_h[si],
            surv_p_h[si],
            H_h[si],
            H_up_h[si],
            H_dn_h[si],
            zcom_up_h[si],
            zcom_dn_h[si],
        ) = vals
        print(
            f"{0:7d}  {vals[0]:10.4e}  {vals[1]:10.4e}  {vals[2]:10.4e}"
            f"  {vals[3]:10.4e}  {vals[4]:8.4f}  {vals[5]:8.4f}  {vals[6]:12.5e}"
            f"  {vals[9]-z0:+9.4f}  {vals[10]+z0:+9.4f}",
            flush=True,
        )
        si += 1

    for local_step in range(1, args.n_steps + 1):
        alat.step(0, dt)
        alat.step_sponge(0)
        if args.cross_scale > 0.0:
            op.step(dt, args.cross_scale)

        global_step = resume_step + local_step
        if global_step % args.report == 0 and si < n_new_snaps:
            vals = measure()
            (
                J_h[si],
                p_up_h[si],
                p_mid_h[si],
                p_dn_h[si],
                surv_e_h[si],
                surv_p_h[si],
                H_h[si],
                H_up_h[si],
                H_dn_h[si],
                zcom_up_h[si],
                zcom_dn_h[si],
            ) = vals
            print(
                f"{global_step:7d}  {vals[0]:10.4e}  {vals[1]:10.4e}  {vals[2]:10.4e}"
                f"  {vals[3]:10.4e}  {vals[4]:8.4f}  {vals[5]:8.4f}  {vals[6]:12.5e}"
                f"  {vals[9]-z0:+9.4f}  {vals[10]+z0:+9.4f}",
                flush=True,
            )
            si += 1

    if prior is not None:
        new_steps = resume_step + np.arange(1, si + 1) * args.report
        steps_arr = np.concatenate([prior["steps"], new_steps])
        J_all = np.concatenate([prior["J"], J_h[:si]])
        p_up_all = np.concatenate([prior["phi_up"], p_up_h[:si]])
        p_mid_all = np.concatenate([prior["phi_mid"], p_mid_h[:si]])
        p_dn_all = np.concatenate([prior["phi_dn"], p_dn_h[:si]])
        surv_e_all = np.concatenate([prior["surv_e"], surv_e_h[:si]])
        surv_p_all = np.concatenate([prior["surv_p"], surv_p_h[:si]])
        H_all = np.concatenate([prior["H"], H_h[:si]])
        H_up_all = np.concatenate([prior["H_up"], H_up_h[:si]])
        H_dn_all = np.concatenate([prior["H_dn"], H_dn_h[:si]])
        zcom_up_all = (
            np.concatenate([prior["zcom_up"], zcom_up_h[:si]])
            if "zcom_up" in prior
            else np.concatenate([np.full(len(prior["steps"]), np.nan), zcom_up_h[:si]])
        )
        zcom_dn_all = (
            np.concatenate([prior["zcom_dn"], zcom_dn_h[:si]])
            if "zcom_dn" in prior
            else np.concatenate([np.full(len(prior["steps"]), np.nan), zcom_dn_h[:si]])
        )
    else:
        steps_arr = np.arange(si) * args.report
        J_all, p_up_all, p_mid_all, p_dn_all = (
            J_h[:si],
            p_up_h[:si],
            p_mid_h[:si],
            p_dn_h[:si],
        )
        surv_e_all, surv_p_all = surv_e_h[:si], surv_p_h[:si]
        H_all, H_up_all, H_dn_all = H_h[:si], H_up_h[:si], H_dn_h[:si]
        zcom_up_all, zcom_dn_all = zcom_up_h[:si], zcom_dn_h[:si]

    if args.save_out:
        alat.save(args.save_out)
        print(f"final state saved -> {args.save_out}")

    if args.data_out:
        np.savez(
            args.data_out,
            steps=steps_arr,
            J=J_all,
            phi_up=p_up_all,
            phi_mid=p_mid_all,
            phi_dn=p_dn_all,
            surv_e=surv_e_all,
            surv_p=surv_p_all,
            H=H_all,
            H_up=H_up_all,
            H_dn=H_dn_all,
            zcom_up=zcom_up_all,
            zcom_dn=zcom_dn_all,
            z0=z0,
            w0_e=w0_e,
            w0_p=w0_p,
            dt=dt,
            m0=args.m0,
            cross_scale=args.cross_scale,
        )
        print(f"data saved -> {args.data_out}")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_snaps = len(steps_arr)
    si = n_snaps
    steps = steps_arr
    J_h, p_up_h, p_mid_h, p_dn_h = J_all, p_up_all, p_mid_all, p_dn_all
    surv_e_h, surv_p_h = surv_e_all, surv_p_all

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        f"electron/positron collision  R={args.R} scale={args.scale}"
        f" z0=+/-{z0:.0f} m0={args.m0} cross={args.cross_scale}",
        fontsize=10,
    )
    axes[0].plot(steps[:si], J_h[:si], color="crimson")
    axes[0].set_ylabel("|J| midplane")
    axes[0].set_title("Current sheet intensity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps[:si], p_up_h[:si], label="upper (e-)", color="tab:blue")
    axes[1].plot(steps[:si], p_mid_h[:si], label="midplane", color="tab:purple")
    axes[1].plot(steps[:si], p_dn_h[:si], label="lower (e+)", color="tab:orange")
    axes[1].set_ylabel("phi rms by zone")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps[:si], surv_e_h[:si], label="surv_e", color="tab:blue")
    axes[2].plot(steps[:si], surv_p_h[:si], label="surv_p", color="tab:orange")
    axes[2].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[2].set_ylabel("winding survival")
    axes[2].set_xlabel("step")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)
    plt.savefig(args.plot_out, dpi=120)
    print(f"plot -> {args.plot_out}")

    # --- Spatial cross-section panels: XZ meridional slice, composite ---
    # z>=0 half read in the electron's own frame (z-z0), z<0 half in the
    # positron's own frame (z+z0) -- each _poloidal_phi call is only
    # meaningful near its own object, split cleanly at the midplane where
    # the interface/current-sheet sits anyway.
    from scipy.interpolate import griddata

    y = pos[:, 1]
    y_u = np.sort(np.abs(np.unique(y)))
    ytol = float(y_u[y_u > 1e-6][0]) * 1.2 if (y_u > 1e-6).any() else 1.5
    clip = args.scale * 3.0
    slice_mask = (np.abs(y) < ytol) & (
        np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2) < clip + z0 + args.scale
    )

    nr = 220
    xi_g = np.linspace(-clip, clip, nr)
    zi_g = np.linspace(-(z0 + clip), z0 + clip, nr)
    Xi, Zi = np.meshgrid(xi_g, zi_g)

    final_step = int(steps_arr[-1]) if len(steps_arr) else resume_step
    fig_c, ax = plt.subplots(figsize=(7, 6.5))
    fig_c.suptitle(
        f"electron/positron collision -- XZ meridional slice, final state"
        f"  R={args.R} scale={args.scale} z0=+/-{z0:.0f}"
        f" step={final_step} m0={args.m0} cross={args.cross_scale}",
        fontsize=10,
    )
    theta = np.linspace(0, 2 * np.pi, 128)
    phi_final = np.array(cl.leaf_phi).T  # (N, 14)
    pol_e = _poloidal_phi(pos - [0, 0, z0], phi_final, r_axis_e)
    pol_p = _poloidal_phi(pos - [0, 0, -z0], phi_final, r_axis_p)
    pol = np.where(pos[:, 2] >= 0, pol_e, pol_p)
    xv, zv, pv = pos[slice_mask, 0], pos[slice_mask, 2], pol[slice_mask]
    Zp = griddata((xv, zv), pv, (Xi, Zi), method="linear")
    vmax = np.nanpercentile(np.abs(Zp), 99) or 1e-6
    im = ax.pcolormesh(Xi, Zi, Zp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig_c.colorbar(im, ax=ax, shrink=0.8, label="phi_poloidal (per-object frame)")
    ax.plot(
        args.scale * np.cos(theta),
        z0 + args.scale * np.sin(theta),
        "k--",
        lw=0.6,
        alpha=0.5,
    )
    ax.plot(
        args.scale * np.cos(theta),
        -z0 + args.scale * np.sin(theta),
        "k--",
        lw=0.6,
        alpha=0.5,
    )
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.plot([r_axis_e, -r_axis_e], [z0, z0], "kx", ms=6, mew=1.3)
    ax.plot([r_axis_p, -r_axis_p], [-z0, -z0], "kx", ms=6, mew=1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    plt.tight_layout()
    cross_out = args.plot_out.replace(".png", "_cross.png")
    plt.savefig(cross_out, dpi=130, bbox_inches="tight")
    print(f"cross-section plot -> {cross_out}")


if __name__ == "__main__":
    main()
