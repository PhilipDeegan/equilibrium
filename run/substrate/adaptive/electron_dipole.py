"""
DST electron — double-ring dipole IC on adaptive lattice.

The electron is modelled as the poloidal component of the Taylor state:
nested tori of phi circulation around a centreline circle, showing two
counter-circulating lobes ("double ring") in a meridional cross-section.
This is the dipole structure the neutron spheromak decays into once its
toroidal winding unwinds — the electron IC is that surviving component,
seeded directly at smaller scale.

Contrast with run/substrate/adaptive/neutron_decay.py (full Taylor
spheromak, both windings).  Under the identical operator set the neutron's
toroidal winding decays; the question here is whether the double-ring
dipole persists (or decays slower) — the DST beta-decay consistency check.

Defaults: lam_target=0 (pure phi winding — whether the electron carries
lam is an open question), sph_frac=0.05 (scale 4.0 at R=80 — smaller than
the neutron's 4.8, but kept above the ~3-cell regime where the BCC k·L
dispersion anisotropy / diamond artifact dominates).

Diagnostics:
  circ_pol : poloidal circulation on tube shells around the centreline —
             the electron's winding amplitude (decay observable)
  circ_tor : toroidal circulation on spherical shells — should stay ~0;
             growth means the dipole is regenerating a toroidal winding

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/electron_dipole.py
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/electron_dipole.py --R 100 --n-steps 50000
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/electron_dipole.py --cross-scale 0.0 --n-steps 2000 --no-plot
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from run.substrate.adaptive.probe_decay_compare import _poloidal_phi
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_electron_dipole_ic,
    phi_poloidal_scan,
    phi_toroidal_scan,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=80.0, help="domain sphere radius")
parser.add_argument(
    "--sph-frac",
    type=float,
    default=0.05,
    help="dipole radius as fraction of R (scale = R * sph_frac)",
)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument(
    "--lam-target",
    type=float,
    default=0.0,
    help="peak lam (0 = pure phi winding — electron null hypothesis)",
)
parser.add_argument(
    "--u-bg",
    type=float,
    default=0.0,
    help="uniform u background — any constant is force-free; >0 for Saha-vacuum runs",
)
parser.add_argument(
    "--no-div-clean",
    action="store_true",
    help="skip the Hodge projection (falls back to dual-zero cutoff handling)",
)
parser.add_argument(
    "--amplitude", type=float, default=1.0, help="phi pre-load amplitude"
)
parser.add_argument(
    "--sigma",
    type=float,
    default=None,
    help="Gaussian taper width in dipole units (None=hard cutoff at R_sph)",
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument(
    "--lam-relax-rate",
    type=float,
    default=0.0,
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
    "--relax-steps",
    type=int,
    default=0,
    help="pre-run steps before diagnostics baseline: the analytic IC is not "
    "a stationary state of the update rule, so it sheds a phi transient; "
    "the sponge absorbs it and what survives is the dressed electron state",
)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--plot-out", type=str, default="electron_dipole.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--save-snapshot", type=str, default=None)
parser.add_argument(
    "--data-out", type=str, default=None, help="save time-series diagnostics as .npz"
)
parser.add_argument(
    "--load-snapshot",
    type=str,
    default=None,
    help="restart from a saved snapshot NPZ (phi, lam, u, pos); skips dipole IC",
)

R_SCAN = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30)
S_SCAN_FRAC = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)  # tube shells, in units of scale


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
    scale = args.R * args.sph_frac  # DST units per dipole unit

    if args.load_snapshot:
        src = np.load(args.load_snapshot)
        cl.set_leaf_phi(src["phi"].T.astype(np.float64))  # saved (N,14) → (14,N)
        cl.set_leaf_lam(src["lam"].astype(np.float64))
        cl.set_leaf_u(src["u"].astype(np.float64))
        b_max = float(np.abs(src["phi"]).max())
        r_axis = float(src["r_axis"]) if "r_axis" in src else 0.63 * scale
        print(f"snapshot loaded from {args.load_snapshot}  (phi_max={b_max:.4f})")
    else:
        b_max, r_axis = init_electron_dipole_ic(
            cl,
            R_sph=1.0,
            scale=scale,
            B0=args.B0,
            lam_target=args.lam_target,
            u_bg=args.u_bg,
            amplitude=args.amplitude,
            sigma=args.sigma,
            div_clean=not args.no_div_clean,
        )

    s_scan = tuple(f * scale for f in S_SCAN_FRAC)
    dt_relax = compute_dt(cl)

    if args.relax_steps > 0:
        print(f"\n=== relax phase: {args.relax_steps} steps ===")
        for step in range(1, args.relax_steps + 1):
            alat.step(0, dt_relax)
            alat.step_sponge(0)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt_relax, args.cross_scale)
            if args.lam_mode == "dynamic":
                alat.step_lam(0, dt_relax, args.lam_scale)
            if args.lam_relax_rate > 0.0:
                step_lam_relax(cl, dt_relax, args.lam_relax_rate, args.lam_alpha)
            if step % args.report == 0:
                mid, n_hat, phi_f = face_flux_geometry(cl, centre)
                pol = phi_poloidal_scan(mid, n_hat, phi_f, r_axis, s_scan)
                prms = float(np.sqrt(np.mean(np.array(cl.leaf_phi) ** 2)))
                print(
                    f"  relax {step:6d}  phi_rms={prms:.4e}"
                    f"  circ_pol_max={float(np.abs(pol).max()):+.3e}",
                    flush=True,
                )
        print("=== relax done — diagnostics baseline starts here ===\n")

    phi0 = np.array(cl.leaf_phi)  # (14, N)
    phi_rms = float(np.sqrt(np.mean(phi0**2)))
    dt = compute_dt(cl)
    print(
        f"N={cl.n_leaves}  R={args.R}  scale={scale:.1f}  r_axis={r_axis:.2f}"
        f"  B0_max={b_max:.3f}  phi_rms={phi_rms:.4e}  dt={dt:.4e}"
    )
    print(
        f"sigma={args.sigma}  amplitude={args.amplitude}"
        f"  cross_scale={args.cross_scale}  lam_scale={args.lam_scale}"
        f"  lam_target={args.lam_target}"
    )

    n_snaps = args.n_steps // args.report + 1
    times = np.zeros(n_snaps)
    H_hist = np.zeros(n_snaps)
    rms_hist = np.zeros(n_snaps)
    pol_hist = np.zeros((n_snaps, len(s_scan)))
    tor_hist = np.zeros((n_snaps, len(R_SCAN)))
    lam_mean_h = np.zeros(n_snaps)
    lam_max_h = np.zeros(n_snaps)

    snap_steps = sorted({0, args.n_steps // 4, args.n_steps // 2, args.n_steps})
    snap_2d = {}

    def record(si, step):
        phi_a = np.array(cl.leaf_phi)
        u_a = np.array(cl.leaf_u)
        lam_a = np.array(cl.leaf_lam)
        times[si] = step * dt
        H_hist[si] = float(np.sum(u_a**2) + 0.5 * np.sum(phi_a**2))
        rms_hist[si] = float(np.sqrt(np.mean(phi_a**2)))
        lam_mean_h[si] = float(lam_a.mean())
        lam_max_h[si] = float(lam_a.max())
        mid, n_hat, phi_f = face_flux_geometry(cl, centre)
        pol_hist[si] = phi_poloidal_scan(mid, n_hat, phi_f, r_axis, s_scan)
        tor_hist[si] = phi_toroidal_scan(mid, n_hat, phi_f, R_SCAN)

    s_hdr = "  ".join(f"s={s:4.1f}" for s in s_scan)
    print(f"\n{'step':>6}  {'H':>10}  {'phi_rms':>10}  {'|tor|max':>10}  {s_hdr}")
    print("  " + "-" * (44 + 9 * len(s_scan)))

    si = 0
    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            record(si, step)
            ps = "  ".join(f"{pol_hist[si, ki]:+.2e}" for ki in range(len(s_scan)))
            tor_max = float(np.abs(tor_hist[si]).max())
            print(
                f"{step:6d}  {H_hist[si]:10.4e}  {rms_hist[si]:10.4e}"
                f"  {tor_max:10.4e}  {ps}",
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

        if step < args.n_steps:
            alat.step(0, dt)
            alat.step_sponge(0)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale)
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
            r_axis=r_axis,
        )
        print(f"snapshot saved → {args.save_snapshot}")

    if args.data_out:
        np.savez(
            args.data_out,
            times=times[:si],
            H=H_hist[:si],
            phi_rms=rms_hist[:si],
            circ_pol=pol_hist[:si],
            circ_tor=tor_hist[:si],
            lam_mean=lam_mean_h[:si],
            lam_max=lam_max_h[:si],
            S_SCAN=np.array(s_scan),
            R_SCAN=np.array(R_SCAN),
            r_axis=r_axis,
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

    clip = max(scale * 3.0, 12.0)
    nr = 200
    xi_g = np.linspace(-clip, clip, nr)
    zi_g = np.linspace(-clip, clip, nr)
    Xi, Zi = np.meshgrid(xi_g, zi_g)

    def xz_slice(sd, clip_r=None):
        """Tiles in the XZ midplane, returning x, z, phi_pol."""
        p = sd["pos"]
        y = p[:, 1]
        y_u = np.sort(np.abs(np.unique(y)))
        ztol = float(y_u[y_u > 1e-6][0]) * 1.2 if (y_u > 1e-6).any() else 1.5
        mask = np.abs(y) < ztol
        if clip_r is not None:
            mask &= np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) < clip_r
        phi_pol = _poloidal_phi(p, sd["phi"].T, r_axis)
        return p[mask, 0], p[mask, 2], phi_pol[mask]

    theta = np.linspace(0, 2 * np.pi, 128)

    # --- Main diagnostics figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"DST electron (double-ring dipole IC)  R={args.R}  scale={scale:.1f}"
        f"  cross={args.cross_scale}  lam_target={args.lam_target}",
        fontsize=10,
    )

    ax_H, ax_rms, ax_pol, ax_tor = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_H.semilogy(times[:si], H_hist[:si], color="steelblue")
    ax_H.set_ylabel("H (log)")
    ax_H.set_xlabel("time")
    ax_H.set_title("Total energy")

    ax_rms.plot(times[:si], rms_hist[:si], color="steelblue")
    ax_rms.set_ylabel("phi_rms")
    ax_rms.set_xlabel("time")
    ax_rms.set_title("phi RMS")

    cmap_p = plt.cm.Greens(np.linspace(0.4, 0.9, len(s_scan)))
    for ki, s in enumerate(s_scan):
        ax_pol.plot(times[:si], pol_hist[:si, ki], color=cmap_p[ki], label=f"s={s:.1f}")
    ax_pol.axhline(0, color="k", lw=0.5)
    ax_pol.set_ylabel("φ poloidal circ.")
    ax_pol.set_xlabel("time")
    ax_pol.set_title("Tube-shell poloidal circulation (electron winding)")
    ax_pol.legend(fontsize=6, ncol=2)

    cmap_t = plt.cm.Blues(np.linspace(0.35, 0.9, len(R_SCAN)))
    for ki, r in enumerate(R_SCAN):
        ax_tor.plot(times[:si], tor_hist[:si, ki], color=cmap_t[ki], label=f"r={r}")
    ax_tor.axhline(0, color="k", lw=0.5)
    ax_tor.set_ylabel("φ toroidal circ.")
    ax_tor.set_xlabel("time")
    ax_tor.set_title("Shell toroidal circulation (should stay ~0)")
    ax_tor.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved → {args.plot_out}")

    # --- Cross-section panels: φ_poloidal — the double ring ---
    steps_s = sorted(snap_2d)
    fig2, axs2 = plt.subplots(1, len(steps_s), figsize=(6 * len(steps_s), 6))
    if len(steps_s) == 1:
        axs2 = [axs2]
    fig2.suptitle(
        f"XZ cross-section φ_poloidal — double-ring dipole  R={args.R}"
        f"  scale={scale:.1f}",
        fontsize=10,
    )
    for ax, step in zip(axs2, steps_s):
        xv, zv, pp = xz_slice(snap_2d[step], clip_r=clip)
        if len(xv) > 3:
            Zp = griddata((xv, zv), pp, (Xi, Zi), method="linear")
            vmax = np.nanpercentile(np.abs(Zp), 99) or 1e-6
            im = ax.pcolormesh(
                Xi, Zi, Zp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
            )
            fig2.colorbar(im, ax=ax, shrink=0.8)
            ax.plot(
                scale * np.cos(theta), scale * np.sin(theta), "w--", lw=0.7, alpha=0.6
            )
            ax.plot([r_axis, -r_axis], [0, 0], "wx", ms=6, mew=1.5)
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
