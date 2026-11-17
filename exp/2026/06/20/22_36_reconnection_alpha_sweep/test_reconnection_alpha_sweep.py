"""
Reconnection lam-alpha sweep — calibrate phi→lam coupling before harris_ic_sweep.

Runs the counter-rotating spheromak reconnection geometry at a set of lam_alpha
values.  The key diagnostic is whether lam at the midplane rises high enough to
measurably suppress sigma (throughput = 1/(1+lam)) during the current-sheet peak.

Relevant range:
  sigma(lam=0.04) = 0.96  →  4%  reduction  (passenger — previous result)
  sigma(lam=0.5)  = 0.67  →  33% reduction  (starting to matter)
  sigma(lam=1.0)  = 0.50  →  50% reduction  (significant driver)
  sigma(lam=9.0)  = 0.10  →  90% reduction  (strong suppression)

lam_target_i = clip(alpha * |phi|_i, 0, LAM_MAX)
dlam_i/dt    = rate * (lam_target_i - lam_i)

For phi_rms ~ 0.02 (reconnection IC), lam reaches O(1) when alpha ~ 50.
This sweep spans alpha ∈ {1, 10, 30, 50} to locate the causal threshold.

Usage:
  PYTHONPATH=$PWD python3 tst/substrate/adaptive/test_reconnection_alpha_sweep.py
  PYTHONPATH=$PWD python3 tst/substrate/adaptive/test_reconnection_alpha_sweep.py \\
      --alpha-values 1,10,30,50,100 --n-steps 10000 --R 60
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from tst.substrate.adaptive.test_lam_phi_drive import step_lam_relax

from equilibrium import lattice as _lat
from equilibrium.substrate.discrete import LAM_MAX
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    apply_toroidal_rotation,
    init_spheromak_ic,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=60.0)
parser.add_argument("--scale", type=float, default=12.0)
parser.add_argument("--z-sep", type=float, default=16.0)
parser.add_argument("--sigma", type=float, default=0.7)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument("--omega", type=float, default=0.1)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-relax-rate", type=float, default=0.1)
parser.add_argument(
    "--alpha-values",
    type=str,
    default="1,10,30,50",
    help="comma-separated lam_alpha values to sweep",
)
parser.add_argument("--cross-scale", type=float, default=0.005)
parser.add_argument("--n-steps", type=int, default=10000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--plot-out", type=str, default="tmp/reconnection_alpha_sweep.png")
parser.add_argument("--no-plot", action="store_true")


def _build_lattice(args):
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

    z0 = args.z_sep / 2.0
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
    lam1, u1 = np.array(cl.leaf_lam).copy(), np.array(cl.leaf_u).copy()

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
    lam2, u2 = np.array(cl.leaf_lam).copy(), np.array(cl.leaf_u).copy()

    cl.set_leaf_lam(np.minimum(lam1 + lam2, float(LAM_MAX)))
    cl.set_leaf_u(u1 + u2)

    apply_toroidal_rotation(
        cl,
        centre=[0.0, 0.0, +z0],
        R_core=args.scale,
        amplitude=+args.omega,
        spin_axis="z",
    )
    apply_toroidal_rotation(
        cl,
        centre=[0.0, 0.0, -z0],
        R_core=args.scale,
        amplitude=-args.omega,
        spin_axis="z",
    )
    return alat, cl, pos, r_all


def run_alpha(args, alpha):
    print(f"\n{'='*60}")
    print(f"  alpha = {alpha}")
    print(f"{'='*60}")

    alat, cl, pos, r_all = _build_lattice(args)
    z = pos[:, 2]
    z0 = args.z_sep / 2.0
    dz_mid = z0 / 4.0

    up_mask = z > z0 * 0.5
    dn_mask = z < -z0 * 0.5
    mid_mask = np.abs(z) < dz_mid

    dt = compute_dt(cl)
    phi0 = np.array(cl.leaf_phi)
    print(f"  N={cl.n_leaves}  dt={dt:.4e}  phi_rms0={np.sqrt(np.mean(phi0**2)):.4e}")
    print(f"  Expected lam_target at phi_rms: {alpha * np.sqrt(np.mean(phi0**2)):.4f}")

    n_snaps = args.n_steps // args.report + 1
    J_h = np.zeros(n_snaps)
    p_rms_h = np.zeros(n_snaps)
    p_mid_h = np.zeros(n_snaps)
    l_mid_h = np.zeros(n_snaps)
    sig_mid_h = np.zeros(n_snaps)  # mean sigma at midplane — key driver metric

    print(
        f"\n  {'step':>7}  {'J_mid':>10}  {'phi_rms':>10}  {'phi_mid':>10}"
        f"  {'lam_mid':>9}  {'sig_mid':>9}"
    )
    print("  " + "-" * 72)

    si = 0
    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            pa = np.array(cl.leaf_phi)
            la = np.array(cl.leaf_lam)

            J_m = (
                float(np.abs(pa.sum(axis=0)[mid_mask]).mean())
                if mid_mask.any()
                else 0.0
            )
            p_rms = float(np.sqrt(np.mean(pa**2)))
            p_mid = (
                float(np.sqrt(np.mean(pa[:, mid_mask] ** 2))) if mid_mask.any() else 0.0
            )
            l_mid = float(la[mid_mask].mean()) if mid_mask.any() else 0.0
            sig_mid = (
                float((1.0 / (1.0 + la[mid_mask])).mean()) if mid_mask.any() else 1.0
            )

            J_h[si] = J_m
            p_rms_h[si] = p_rms
            p_mid_h[si] = p_mid
            l_mid_h[si] = l_mid
            sig_mid_h[si] = sig_mid

            print(
                f"  {step:7d}  {J_m:10.4e}  {p_rms:10.4e}  {p_mid:10.4e}"
                f"  {l_mid:9.4f}  {sig_mid:9.4f}",
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
                step_lam_relax(cl, dt, args.lam_relax_rate, alpha)

    return {
        "J": J_h[:si],
        "phi_rms": p_rms_h[:si],
        "phi_mid": p_mid_h[:si],
        "lam_mid": l_mid_h[:si],
        "sig_mid": sig_mid_h[:si],
        "steps": np.arange(si) * args.report,
        "dt": dt,
    }


def main():
    args = parser.parse_args()
    alphas = [float(a) for a in args.alpha_values.split(",")]

    print(f"Reconnection alpha sweep: {alphas}")
    print(f"R={args.R}  scale={args.scale}  z_sep={args.z_sep}  omega={args.omega}")
    print(
        f"lam_relax_rate={args.lam_relax_rate}  lam_scale={args.lam_scale}  n_steps={args.n_steps}"
    )

    results = {}
    for alpha in alphas:
        results[alpha] = run_alpha(args, alpha)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: peak J_mid and lam_mid per alpha")
    print(
        f"  {'alpha':>8}  {'J_peak':>10}  {'J_step':>8}  {'lam_mid_pk':>12}  {'sig_mid_min':>12}"
    )
    print("  " + "-" * 55)
    for alpha in alphas:
        r = results[alpha]
        j_pk_idx = int(np.argmax(r["J"]))
        j_pk = float(r["J"][j_pk_idx])
        j_step = int(r["steps"][j_pk_idx])
        l_pk = float(r["lam_mid"].max())
        s_min = float(r["sig_mid"].min())
        print(
            f"  {alpha:>8.1f}  {j_pk:>10.4e}  {j_step:>8d}  {l_pk:>12.4f}  {s_min:>12.4f}"
        )

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / max(len(alphas) - 1, 1)) for i in range(len(alphas))]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(
        f"Reconnection alpha sweep  (R={args.R}, scale={args.scale}, "
        f"omega={args.omega}, rate={args.lam_relax_rate})",
        fontsize=11,
    )

    panels = [
        ("J", axes[0], "|J| midplane (div-φ at z≈0)", "current sheet intensity"),
        ("phi_rms", axes[1], "φ rms (global)", "phi energy decay"),
        ("lam_mid", axes[2], "mean lam midplane", "lam build-up at sheet"),
        ("sig_mid", axes[3], "mean σ = 1/(1+lam) midplane", "throughput suppression"),
    ]

    for key, ax, ylabel, title in panels:
        for alpha, col in zip(alphas, colors):
            r = results[alpha]
            ax.plot(r["steps"], r[key], color=col, lw=1.6, label=f"α={alpha:.0f}")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)

    # sigma reference lines
    for thresh, label in [
        (0.9, "σ=0.9 (10% slow)"),
        (0.67, "σ=0.67 (33% slow)"),
        (0.5, "σ=0.5 (50% slow)"),
    ]:
        axes[3].axhline(thresh, color="gray", lw=0.8, ls="--")
        axes[3].text(10, thresh + 0.01, label, fontsize=7, color="gray")

    axes[0].legend(fontsize=9)
    axes[3].set_xlabel("step")

    Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=120)
    print(f"\nplot → {args.plot_out}")


if __name__ == "__main__":
    main()
