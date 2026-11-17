"""
DST neutron decay — Taylor spheromak IC on adaptive lattice.

Uses the analytical Taylor spheromak (force-free, curl(B)=kB) as the initial
condition instead of the minimal 8-tile Hopf ring.  The spheromak fills ~20%
of the domain radius by default, giving proper spatial extent and a
cross-sectional two-lobe profile matching the target neutron structure.

The Gaussian envelope (sigma < R_sph) replaces the hard radial cutoff so that
boundary tiles have smoothly decaying phi, keeping discrete div(phi) small.

step_cross opens topology; step_lam relaxes the winding field.

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/neutron_decay.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/neutron_decay.py --R 100 --n-steps 50000
  PYTHONPATH=$PWD python3 run/substrate/adaptive/neutron_decay.py --cross-scale 0.0 --n-steps 2000 --no-plot
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from run.substrate.adaptive.probe_decay_compare import plot_snapshots
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_NORM,
    init_spheromak_ic,
    phi_toroidal_circulation,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=80.0, help="domain sphere radius")
parser.add_argument(
    "--sph-frac",
    type=float,
    default=0.06,
    help="spheromak radius as fraction of R (scale = R * sph_frac)",
)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument(
    "--amplitude", type=float, default=1.0, help="phi pre-load amplitude"
)
parser.add_argument(
    "--sigma",
    type=float,
    default=None,
    help="Gaussian taper width in spheromak units (None=hard cutoff at R_sph)",
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument(
    "--lam-relax-rate",
    type=float,
    default=0.05,
    help="rate for step_lam_relax: dlam/dt = rate*(alpha*|phi| - lam)  (0=off)",
)
parser.add_argument(
    "--lam-alpha",
    type=float,
    default=1.0,
    help="phi→lam coupling: lam_target = alpha * |phi|",
)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument(
    "--gate",
    choices=("v1", "barrier"),
    default="barrier",
    help="v1: continuous always-fires step_cross. barrier: quantized "
    "all-or-nothing gate (doc/dst_next.md sec 8), ported to "
    "AdaptiveLattice::step_cross (inc/eq/core/data/lattice/adaptive_lattice"
    ".hpp, _step_cross_local/_step_cross_mirror). Default m0 (1e-4) is the "
    "value calibrated on the adaptive Poynting-ring probe (see "
    "doc/mems/project_poynting_dipole_calibration.md) to actually let the "
    "gate fire beyond the immediate source -- NOT the earlier m0=0.3 probe "
    "default, which never fires beyond the source at all.",
)
parser.add_argument(
    "--m0",
    type=float,
    default=1e-4,
    help="barrier gate threshold (pooled per-cell drive Sum W*max(-phi,0)); "
    "ignored for gate=v1",
)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--plot-out", type=str, default="neutron_decay.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--save-snapshot", type=str, default=None)
parser.add_argument(
    "--data-out", type=str, default=None, help="save time-series diagnostics as .npz"
)
parser.add_argument(
    "--plot-cadence",
    type=int,
    default=0,
    help="call probe_decay_compare every N steps (0=off)",
)
parser.add_argument("--plot-dir", type=str, default="tmp")
parser.add_argument(
    "--max-compare",
    type=int,
    default=1,
    help="max snapshots shown side-by-side in cadence plots",
)
parser.add_argument(
    "--zoom",
    type=float,
    default=None,
    help="clip radius for cadence cross-section plots",
)
parser.add_argument(
    "--lam-from",
    type=str,
    default=None,
    help="load converged lam from an NPZ (pos, lam keys) instead of spheromak IC",
)
parser.add_argument(
    "--load-snapshot",
    type=str,
    default=None,
    help="restart from a saved snapshot NPZ (phi, lam, u, pos); skips spheromak IC",
)

R_SCAN = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def toroidal_phi_per_cell(pos, phi_arr):
    """Net azimuthal (z-axis toroidal) phi component per cell. phi_arr: (14, N)."""
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    hat_x = -pos[:, 1] / rho_s  # φ̂_x = -y/ρ
    hat_y = pos[:, 0] / rho_s  # φ̂_y =  x/ρ
    # project each face normal onto the azimuthal direction at the cell centre
    proj = (
        _FACE_NORM[:, 0:1] * hat_x[None, :] + _FACE_NORM[:, 1:2] * hat_y[None, :]
    )  # (14, N)
    return np.sum(phi_arr * proj, axis=0)  # (N,)


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

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    centre = np.zeros(3)
    scale = args.R * args.sph_frac  # DST units per spheromak unit

    if args.load_snapshot:
        src = np.load(args.load_snapshot)
        cl.set_leaf_phi(src["phi"].T.astype(np.float64))  # saved (N,14) → (14,N)
        cl.set_leaf_lam(src["lam"].astype(np.float64))
        cl.set_leaf_u(src["u"].astype(np.float64))
        b_max = float(np.abs(src["phi"]).max())
        print(f"snapshot loaded from {args.load_snapshot}  (phi_max={b_max:.4f})")
    else:
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

    if args.lam_from:
        src = np.load(args.lam_from)
        cl.set_leaf_lam(src["lam"].astype(np.float64))
        print(f"lam loaded from {args.lam_from}  (max={src['lam'].max():.4f})")

    phi0 = np.array(cl.leaf_phi)  # (14, N)
    phi_rms = float(np.sqrt(np.mean(phi0**2)))
    dt = compute_dt(cl)
    print(
        f"N={cl.n_leaves}  R={args.R}  scale={scale:.1f}  B0_max={b_max:.3f}"
        f"  phi_rms={phi_rms:.4e}  dt={dt:.4e}"
    )
    print(
        f"sigma={args.sigma}  amplitude={args.amplitude}"
        f"  cross_scale={args.cross_scale}  lam_scale={args.lam_scale}"
        f"  gate={args.gate}"
        + (f"  m0={args.m0:.2e}" if args.gate == "barrier" else "")
    )
    cross_kwargs = {"gate": "barrier", "m0": args.m0} if args.gate == "barrier" else {}

    n_snaps = args.n_steps // args.report + 1
    times = np.zeros(n_snaps)
    H_hist = np.zeros(n_snaps)
    rms_hist = np.zeros(n_snaps)
    circ_hist = np.zeros((n_snaps, len(R_SCAN)))
    lam_mean_h = np.zeros(n_snaps)
    lam_max_h = np.zeros(n_snaps)
    lam_std_h = np.zeros(n_snaps)

    snap_steps = sorted({0, args.n_steps // 4, args.n_steps // 2, args.n_steps})
    snap_2d = {}

    def _live_snap(label):
        return {
            "phi": np.array(cl.leaf_phi).T,  # (N, 14)
            "u": np.array(cl.leaf_u),
            "lam": np.array(cl.leaf_lam),
            "pos": np.array(cl.leaf_pos),
            "label": label,
            "ring_r": 0.0,
        }

    compare_snaps = [_live_snap("IC")]

    def record(si, step):
        phi_a = np.array(cl.leaf_phi)
        u_a = np.array(cl.leaf_u)
        lam_a = np.array(cl.leaf_lam)
        times[si] = step * dt
        H_hist[si] = float(np.sum(u_a**2) + 0.5 * np.sum(phi_a**2))
        rms_hist[si] = float(np.sqrt(np.mean(phi_a**2)))
        lam_mean_h[si] = float(lam_a.mean())
        lam_max_h[si] = float(lam_a.max())
        lam_std_h[si] = float(lam_a.std())
        for ki, r in enumerate(R_SCAN):
            circ_hist[si, ki] = phi_toroidal_circulation(cl, centre, r)

    r_hdr = "  ".join(f"r={r:2d}" for r in R_SCAN)
    print(f"\n{'step':>6}  {'H':>10}  {'phi_rms':>10}  {r_hdr}")
    print("  " + "-" * (32 + 8 * len(R_SCAN)))

    si = 0
    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            record(si, step)
            cs = "  ".join(f"{circ_hist[si, ki]:+.2e}" for ki in range(len(R_SCAN)))
            print(
                f"{step:6d}  {H_hist[si]:10.4e}  {rms_hist[si]:10.4e}  {cs}",
                flush=True,
            )
            si += 1

        if step in snap_steps:
            snap_2d[step] = {
                "phi": np.array(cl.leaf_phi).copy(),
                "u": np.array(cl.leaf_u).copy(),
                "lam": np.array(cl.leaf_lam).copy(),
                "pos": np.array(cl.leaf_pos).copy(),
            }

        if args.plot_cadence > 0 and step > 0 and step % args.plot_cadence == 0:
            compare_snaps.append(_live_snap(f"s{step}"))
            if args.max_compare > 0:
                compare_snaps = compare_snaps[-args.max_compare :]
            plot_snapshots(
                compare_snaps,
                zoom=args.zoom,
                plot_dir=args.plot_dir,
                tag=f"s{step:07d}",
            )

        if step < args.n_steps:
            alat.step(0, dt)
            alat.step_sponge(0)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale, **cross_kwargs)
            if args.lam_mode == "dynamic":
                alat.step_lam(0, dt, args.lam_scale)
            if args.lam_relax_rate > 0.0:
                step_lam_relax(cl, dt, args.lam_relax_rate, args.lam_alpha)

    if args.save_snapshot:
        np.savez(
            args.save_snapshot,
            phi=np.array(cl.leaf_phi).T,
            lam=np.array(cl.leaf_lam),
            u=np.array(cl.leaf_u),
            pos=np.array(cl.leaf_pos),
        )
        print(f"snapshot saved → {args.save_snapshot}")

    if args.data_out:
        np.savez(
            args.data_out,
            times=times[:si],
            H=H_hist[:si],
            phi_rms=rms_hist[:si],
            circ=circ_hist[:si],
            lam_mean=lam_mean_h[:si],
            lam_max=lam_max_h[:si],
            lam_std=lam_std_h[:si],
            R_SCAN=np.array(R_SCAN),
        )
        print(f"data saved → {args.data_out}")

    if args.no_plot:
        return

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    clip = args.R * 0.75
    nr = 200
    xi_g = np.linspace(-clip, clip, nr)
    zi_g = np.linspace(-clip, clip, nr)
    Xi, Zi = np.meshgrid(xi_g, zi_g)

    def xz_slice(sd, clip_r=None):
        """Tiles in the XZ midplane (|y| minimal), returning x, z, phi_tor, u, lam."""
        p = sd["pos"]
        y = p[:, 1]
        y_u = np.sort(np.abs(np.unique(y)))
        ztol = float(y_u[y_u > 1e-6][0]) * 1.2 if (y_u > 1e-6).any() else 1.5
        mask = np.abs(y) < ztol
        if clip_r is not None:
            mask &= np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) < clip_r
        phi_tor = toroidal_phi_per_cell(p, sd["phi"])
        return p[mask, 0], p[mask, 2], phi_tor[mask], sd["u"][mask], sd["lam"][mask]

    theta = np.linspace(0, 2 * np.pi, 128)

    # --- Main diagnostics figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"DST neutron (Taylor IC)  R={args.R}  scale={scale:.0f}"
        f"  cross={args.cross_scale}  sigma={args.sigma}",
        fontsize=10,
    )

    ax_H, ax_rms, ax_circ, ax_ic = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_H.semilogy(times[:si], H_hist[:si], color="steelblue")
    ax_H.set_ylabel("H (log)")
    ax_H.set_xlabel("time")
    ax_H.set_title("Total energy")

    ax_rms.plot(times[:si], rms_hist[:si], color="steelblue")
    ax_rms.set_ylabel("phi_rms")
    ax_rms.set_xlabel("time")
    ax_rms.set_title("phi RMS")

    cmap_c = plt.cm.Blues(np.linspace(0.35, 0.9, len(R_SCAN)))
    for ki, r in enumerate(R_SCAN):
        ax_circ.plot(times[:si], circ_hist[:si, ki], color=cmap_c[ki], label=f"r={r}")
    ax_circ.axhline(0, color="k", lw=0.5)
    ax_circ.set_ylabel("φ toroidal circ.")
    ax_circ.set_xlabel("time")
    ax_circ.set_title("Shell toroidal circulation")
    ax_circ.legend(fontsize=6, ncol=2)

    # IC cross-section
    if 0 in snap_2d:
        xv, zv, pt, uv, lv = xz_slice(snap_2d[0], clip_r=clip)
        if len(xv) > 3:
            Zp = griddata((xv, zv), pt, (Xi, Zi), method="linear")
            vmax = np.nanpercentile(np.abs(Zp), 99) or 1e-6
            im = ax_ic.pcolormesh(
                Xi, Zi, Zp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
            )
            fig.colorbar(im, ax=ax_ic, shrink=0.8)
            ax_ic.plot(
                scale * np.cos(theta), scale * np.sin(theta), "w--", lw=0.7, alpha=0.6
            )
    ax_ic.set_aspect("equal")
    ax_ic.set_title("φ_toroidal — XZ slice  (IC, step 0)")
    ax_ic.set_xlabel("x")
    ax_ic.set_ylabel("z")

    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved → {args.plot_out}")

    # --- Before/after cross-section panels ---
    if len(snap_2d) < 2:
        return

    steps_s = sorted(snap_2d)
    fig2, axs2 = plt.subplots(1, len(steps_s), figsize=(6 * len(steps_s), 6))
    if len(steps_s) == 1:
        axs2 = [axs2]
    fig2.suptitle(
        f"XZ cross-section φ_toroidal — neutron IC  R={args.R}  scale={scale:.0f}",
        fontsize=10,
    )
    for ax, step in zip(axs2, steps_s):
        xv, zv, pt, uv, lv = xz_slice(snap_2d[step], clip_r=clip)
        if len(xv) > 3:
            Zp = griddata((xv, zv), pt, (Xi, Zi), method="linear")
            vmax = np.nanpercentile(np.abs(Zp), 99) or 1e-6
            im = ax.pcolormesh(
                Xi, Zi, Zp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
            )
            fig2.colorbar(im, ax=ax, shrink=0.8)
            ax.plot(
                scale * np.cos(theta), scale * np.sin(theta), "w--", lw=0.7, alpha=0.6
            )
        ax.set_aspect("equal")
        ax.set_title(f"step {step}  t={step * dt:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    cross_out = args.plot_out.replace(".png", "_cross.png")
    plt.tight_layout()
    plt.savefig(cross_out, dpi=130, bbox_inches="tight")
    print(f"cross-section panels saved → {cross_out}")


if __name__ == "__main__":
    main()
