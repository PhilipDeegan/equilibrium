"""
Shared core for the isolated electron/positron control runs -- same
lattice, geometry, sponge BC, and step_cross v2 barrier gate as
electron_positron_collision.py, but with only ONE dipole built (the
electron sits at z=+z0 same as in the pair run, the positron at z=-z0
same as in the pair run), so cross_scale can only act on that object
alone -- no midplane reconnection is possible, no other object's field
is present. This isolates each object's own decay/erosion under
step_cross v2 from anything driven by interaction with the other one,
giving a no-interaction control to compare against the pair run's
surv_e/surv_p history.

Entry points: electron_isolated.py / positron_isolated.py.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from run.substrate.adaptive.cross_v2 import CrossV2
from run.substrate.adaptive.counter_rotating_spheromak_reconnection import (
    current_sheet_j,
    phi_rms_zone,
)
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_electron_dipole_ic,
    phi_poloidal_scan,
)


def build_parser(species):
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=60.0)
    parser.add_argument(
        "--scale", type=float, default=8.0, help="DST units per dipole unit"
    )
    parser.add_argument(
        "--z-sep", type=float, default=20.0, help="same z0 offset used in the pair run"
    )
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
    parser.add_argument("--cross-scale", type=float, default=0.1)
    parser.add_argument(
        "--lam-target",
        type=float,
        default=0.0,
        help="peak lam (lam = lam_target * |B|/|B|_max, same B that sources phi) -- default 0 preserves the pure-phi-winding baseline",
    )
    parser.add_argument(
        "--lam-relax-rate",
        type=float,
        default=0.0,
        help="rate for step_lam_relax: dlam/dt = rate*(alpha*|phi| - lam), applied every step (0=off)",
    )
    parser.add_argument(
        "--lam-alpha",
        type=float,
        default=1.0,
        help="phi->lam coupling: lam_target_local = alpha * |phi|",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1000,
        help="steps to run -- ADDITIONAL steps beyond the resume point when resuming",
    )
    parser.add_argument("--report", type=int, default=100)
    parser.add_argument("--sponge", type=float, default=0.15)
    parser.add_argument("--damp-max", type=float, default=0.003)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--plot-out", type=str, default=f"tmp/{species}_isolated.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--data-out", type=str, default=None)
    parser.add_argument("--save-out", type=str, default=None)
    parser.add_argument(
        "--bk-dir",
        type=str,
        default="/bk/dst",
        help="directory to scan for resumable snapshots",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="snapshot filename prefix: {prefix}{N}.hdf5 (default: {species}_isolated_step)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="always rebuild the IC from t=0, even if a snapshot exists",
    )
    return parser


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


def run(args, species):
    assert species in ("electron", "positron")
    amplitude = +1.0 if species == "electron" else -1.0
    z_sign = +1.0 if species == "electron" else -1.0

    save_prefix = args.save_prefix or f"{species}_isolated_step"
    resume_step, snap_path = (
        (None, None)
        if args.no_resume
        else _find_latest_snapshot(args.bk_dir, save_prefix)
    )

    z0 = args.z_sep / 2.0
    cz = z_sign * z0
    dz_mid = z0 / 4.0
    s_scan = tuple(f * args.scale for f in (0.5, 1.0, 1.5))

    if resume_step is not None:
        prior_npz = os.path.join(
            os.path.dirname(args.data_out), f"{species}_isolated_step{resume_step}.npz"
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

        r_axis = _r_axis(1.0, args.scale)
        w0 = float(prior["w0"])
        print(
            f"N={cl.n_leaves}  R(inferred)={r_max:.1f}  scale={args.scale}  species={species}"
            f"  z={cz:+.1f}  m0={args.m0}  cross_scale={args.cross_scale}  w0={w0:.4e}"
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
            f"N={cl.n_leaves}  R={args.R}  scale={args.scale}  species={species}"
            f"  z={cz:+.1f}  m0={args.m0}  cross_scale={args.cross_scale}  lam_target={args.lam_target}"
        )

        print(f"\n[{species}] dipole IC at z={cz:+.1f} (isolated -- no partner object)")
        # NOTE (same as electron_positron_collision.py): init_electron_dipole_ic
        # only builds/stores phi inside its `if amplitude > 0.0:` guard --
        # passing amplitude<0 skips that block entirely. Build with +1.0 then
        # negate the captured array in Python for the positron.
        _, r_axis = init_electron_dipole_ic(
            cl,
            R_sph=1.0,
            scale=args.scale,
            B0=args.B0,
            lam_target=args.lam_target,
            u_bg=0.0,
            amplitude=+1.0,
            cx=0.0,
            cy=0.0,
            cz=cz,
        )
        if amplitude < 0.0:
            cl.set_leaf_phi(-np.array(cl.leaf_phi))
        cl.set_leaf_u(np.zeros(cl.n_leaves))

    dt = compute_dt(cl)
    op = CrossV2(cl, gate=args.gate, m0=args.m0, ref=args.ref)

    own_mask = (z > z0 * 0.5) if species == "electron" else (z < -z0 * 0.5)
    far_mask = (z < -z0 * 0.5) if species == "electron" else (z > z0 * 0.5)
    mid_mask = np.abs(z) < dz_mid

    def winding():
        mid, n_hat, phi_f = face_flux_geometry(cl, [0, 0, cz])
        pol = phi_poloidal_scan(mid, n_hat, phi_f, r_axis, s_scan)
        return float(np.abs(pol).max())

    if prior is None:
        w0 = winding()
        print(f"w0={w0:.4e}  dt={dt:.4e}")

    n_new_snaps = args.n_steps // args.report + (1 if prior is None else 0)
    J_h = np.zeros(n_new_snaps)
    p_own_h = np.zeros(n_new_snaps)
    p_mid_h = np.zeros(n_new_snaps)
    p_far_h = np.zeros(n_new_snaps)
    surv_h = np.zeros(n_new_snaps)
    H_h = np.zeros(n_new_snaps)
    H_own_h = np.zeros(n_new_snaps)

    print(
        f"\n{'step':>7}  {'J_mid':>10}  {'phi_own':>10}  {'phi_mid':>10}"
        f"  {'phi_far':>10}  {'surv':>8}  {'H_total':>12}"
    )
    print("  " + "-" * 78)

    def measure():
        ua = np.array(cl.leaf_u)
        pa = np.array(cl.leaf_phi)
        J_m = current_sheet_j(pa, z, dz_mid)
        p_own = phi_rms_zone(pa, own_mask)
        p_md = phi_rms_zone(pa, mid_mask)
        p_far = phi_rms_zone(pa, far_mask)
        sv = winding() / max(w0, 1e-30)
        H_cell = ua**2 + 0.5 * np.sum(pa**2, axis=0)
        H_tot = float(H_cell.sum())
        H_own = float(H_cell[own_mask].sum())
        return J_m, p_own, p_md, p_far, sv, H_tot, H_own

    si = 0
    if prior is None:
        vals = measure()
        (
            J_h[si],
            p_own_h[si],
            p_mid_h[si],
            p_far_h[si],
            surv_h[si],
            H_h[si],
            H_own_h[si],
        ) = vals
        print(
            f"{0:7d}  {vals[0]:10.4e}  {vals[1]:10.4e}  {vals[2]:10.4e}"
            f"  {vals[3]:10.4e}  {vals[4]:8.4f}  {vals[5]:12.5e}",
            flush=True,
        )
        si += 1

    for local_step in range(1, args.n_steps + 1):
        alat.step(0, dt)
        alat.step_sponge(0)
        if args.cross_scale > 0.0:
            op.step(dt, args.cross_scale)
        if args.lam_relax_rate > 0.0:
            step_lam_relax(cl, dt, args.lam_relax_rate, args.lam_alpha)

        global_step = resume_step + local_step
        if global_step % args.report == 0 and si < n_new_snaps:
            vals = measure()
            (
                J_h[si],
                p_own_h[si],
                p_mid_h[si],
                p_far_h[si],
                surv_h[si],
                H_h[si],
                H_own_h[si],
            ) = vals
            print(
                f"{global_step:7d}  {vals[0]:10.4e}  {vals[1]:10.4e}  {vals[2]:10.4e}"
                f"  {vals[3]:10.4e}  {vals[4]:8.4f}  {vals[5]:12.5e}",
                flush=True,
            )
            si += 1

    if prior is not None:
        new_steps = resume_step + np.arange(1, si + 1) * args.report
        steps_arr = np.concatenate([prior["steps"], new_steps])
        J_all = np.concatenate([prior["J"], J_h[:si]])
        p_own_all = np.concatenate([prior["phi_own"], p_own_h[:si]])
        p_mid_all = np.concatenate([prior["phi_mid"], p_mid_h[:si]])
        p_far_all = np.concatenate([prior["phi_far"], p_far_h[:si]])
        surv_all = np.concatenate([prior["surv"], surv_h[:si]])
        H_all = np.concatenate([prior["H"], H_h[:si]])
        H_own_all = np.concatenate([prior["H_own"], H_own_h[:si]])
    else:
        steps_arr = np.arange(si) * args.report
        J_all, p_own_all, p_mid_all, p_far_all = (
            J_h[:si],
            p_own_h[:si],
            p_mid_h[:si],
            p_far_h[:si],
        )
        surv_all, H_all, H_own_all = surv_h[:si], H_h[:si], H_own_h[:si]

    si = len(steps_arr)
    steps_arr_final = steps_arr

    if args.save_out:
        alat.save(args.save_out)
        print(f"final state saved -> {args.save_out}")

    if args.data_out:
        np.savez(
            args.data_out,
            steps=steps_arr_final,
            J=J_all,
            phi_own=p_own_all,
            phi_mid=p_mid_all,
            phi_far=p_far_all,
            surv=surv_all,
            H=H_all,
            H_own=H_own_all,
            w0=w0,
            dt=dt,
            m0=args.m0,
            cross_scale=args.cross_scale,
            species=species,
        )
        print(f"data saved -> {args.data_out}")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = steps_arr_final
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        f"isolated {species}  R={args.R} scale={args.scale} z={cz:+.0f}"
        f" m0={args.m0} cross={args.cross_scale}",
        fontsize=10,
    )
    axes[0].plot(steps[:si], J_all[:si], color="crimson")
    axes[0].set_ylabel("|J| midplane")
    axes[0].set_title(
        "Current sheet intensity (should stay ~background, no partner to reconnect with)"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps[:si], p_own_all[:si], label=f"own ({species})", color="tab:blue")
    axes[1].plot(steps[:si], p_mid_all[:si], label="midplane", color="tab:purple")
    axes[1].plot(steps[:si], p_far_all[:si], label="far (empty)", color="tab:orange")
    axes[1].set_ylabel("phi rms by zone")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps[:si], surv_all[:si], label="surv", color="tab:blue")
    axes[2].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[2].set_ylabel("winding survival")
    axes[2].set_xlabel("step")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)
    plt.savefig(args.plot_out, dpi=120)
    print(f"plot -> {args.plot_out}")

    # --- Spatial cross-section: XZ meridional slice, final state, single
    # object's own frame (centred at cz) -- same construction as the
    # composite plot in electron_positron_collision.py.
    from scipy.interpolate import griddata

    from run.substrate.adaptive.probe_decay_compare import _poloidal_phi

    y = pos[:, 1]
    y_u = np.sort(np.abs(np.unique(y)))
    ytol = float(y_u[y_u > 1e-6][0]) * 1.2 if (y_u > 1e-6).any() else 1.5
    clip = args.scale * 3.0
    slice_mask = (np.abs(y) < ytol) & (
        np.sqrt(pos[:, 0] ** 2 + (pos[:, 2] - cz) ** 2) < clip
    )

    nr = 220
    xi_g = np.linspace(-clip, clip, nr)
    zi_g = np.linspace(cz - clip, cz + clip, nr)
    Xi, Zi = np.meshgrid(xi_g, zi_g)

    final_step = int(steps_arr_final[-1]) if len(steps_arr_final) else resume_step
    fig_c, ax = plt.subplots(figsize=(7, 6.5))
    fig_c.suptitle(
        f"isolated {species} -- XZ meridional slice, final state"
        f"  R={args.R} scale={args.scale} z={cz:+.0f}"
        f" step={final_step} m0={args.m0} cross={args.cross_scale}",
        fontsize=10,
    )
    theta = np.linspace(0, 2 * np.pi, 128)
    phi_final = np.array(cl.leaf_phi).T  # (N, 14)
    pol = _poloidal_phi(pos - [0, 0, cz], phi_final, r_axis)
    xv, zv, pv = pos[slice_mask, 0], pos[slice_mask, 2], pol[slice_mask]
    Zp = griddata((xv, zv), pv, (Xi, Zi), method="linear")
    vmax = np.nanpercentile(np.abs(Zp), 99) or 1e-6
    im = ax.pcolormesh(Xi, Zi, Zp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig_c.colorbar(im, ax=ax, shrink=0.8, label="phi_poloidal (own frame)")
    ax.plot(
        args.scale * np.cos(theta),
        cz + args.scale * np.sin(theta),
        "k--",
        lw=0.6,
        alpha=0.5,
    )
    ax.axhline(cz, color="k", lw=0.5, alpha=0.6)
    ax.plot([r_axis, -r_axis], [cz, cz], "kx", ms=6, mew=1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    plt.tight_layout()
    cross_out = args.plot_out.replace(".png", "_cross.png")
    plt.savefig(cross_out, dpi=130, bbox_inches="tight")
    print(f"cross-section plot -> {cross_out}")
