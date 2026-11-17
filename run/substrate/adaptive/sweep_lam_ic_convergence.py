"""
Lam IC convergence sweep — fix Taylor spheromak phi, sweep alpha.

Freezes phi (no alat.step()) and runs step_lam_relax to find the lam
profile each alpha value naturally drives toward.  The converged profile
is the physically-motivated lam IC for subsequent neutron decay runs,
replacing the current "lam mirrors phi" assumption.

Rate controls convergence speed only — the equilibrium shape depends
solely on alpha.  A single high rate is used for all runs.

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_lam_ic_convergence.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_lam_ic_convergence.py \\
      --alphas 0.1 0.5 1.0 2.0 5.0 --rate 2.0 --tol 1e-5

Output (--out-dir):
  alpha_<a>.npz         — converged lam, phi_mag, pos, convergence curve
  convergence.png       — max|Δlam| vs time for all alphas
  scatter_lam_vs_target.png — lam_converged vs alpha·|φ| scatter
  xz_slices.png         — XZ cross-section of |φ| and lam for each alpha
  radial_profiles.png   — radial mean lam and lam/alpha (shape invariance check)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import init_spheromak_ic

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--sph-frac", type=float, default=0.2)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--sigma", type=float, default=None)
parser.add_argument(
    "--alphas",
    type=float,
    nargs="+",
    default=[0.1, 0.5, 1.0, 2.0, 5.0],
    help="phi→lam coupling values to sweep",
)
parser.add_argument(
    "--rate",
    type=float,
    default=2.0,
    help="lam_relax rate — controls convergence speed, not equilibrium shape",
)
parser.add_argument(
    "--tol",
    type=float,
    default=1e-5,
    help="convergence threshold: max|Δlam| < tol * phi_mag_max",
)
parser.add_argument("--max-steps", type=int, default=20000)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--out-dir", type=str, default="tmp/lam_ic_sweep")
parser.add_argument("--no-plot", action="store_true")


def main():
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]

    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    scale = args.R * args.sph_frac
    init_spheromak_ic(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=args.B0,
        lam_target=args.lam_target,
        u_max=args.u_max,
        amplitude=args.amplitude,
        sigma=args.sigma,
    )

    dt = compute_dt(cl)
    phi_frozen = np.array(cl.leaf_phi)  # (14, N) — never changes
    pos = np.array(cl.leaf_pos)  # (N, 3)
    phi_mag = np.sqrt((phi_frozen**2).sum(axis=0))  # (N,)
    phi_mag_max = float(phi_mag.max()) or 1.0

    print(f"N={cl.n_leaves}  R={args.R}  scale={scale:.1f}  dt={dt:.4e}")
    print(f"phi_mag_max={phi_mag_max:.4e}  tol={args.tol:.1e}  rate={args.rate}")
    print(f"alphas: {args.alphas}\n")

    results = {}

    for alpha in args.alphas:
        cl.set_leaf_lam(np.zeros(cl.n_leaves))  # start from lam=0 each run

        conv = []
        threshold = args.tol * phi_mag_max
        converged_step = args.max_steps

        for step in range(args.max_steps):
            lam_old = np.array(cl.leaf_lam)
            step_lam_relax(cl, dt, args.rate, alpha)
            lam_new = np.array(cl.leaf_lam)
            delta = float(np.max(np.abs(lam_new - lam_old)))
            conv.append(delta)

            if delta < threshold:
                converged_step = step + 1
                print(
                    f"  alpha={alpha:.2f}  converged at step {converged_step}"
                    f"  max|Δlam|={delta:.2e}"
                )
                break

            if step % 2000 == 0 and step > 0:
                print(
                    f"  alpha={alpha:.2f}  step {step}  max|Δlam|={delta:.2e}",
                    flush=True,
                )
        else:
            print(
                f"  alpha={alpha:.2f}  max steps reached" f"  max|Δlam|={conv[-1]:.2e}"
            )

        results[alpha] = {
            "lam": np.array(cl.leaf_lam).copy(),
            "steps": converged_step,
            "conv": np.array(conv),
        }
        np.savez(
            out_dir / f"alpha_{alpha:.4f}.npz",
            lam=results[alpha]["lam"],
            phi_mag=phi_mag,
            pos=pos,
            conv=results[alpha]["conv"],
            alpha=np.float64(alpha),
            steps=np.int64(converged_step),
        )
        print(f"  saved  alpha_{alpha:.4f}.npz\n")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    clip = args.R * 0.75
    nr = 200
    xi_g = np.linspace(-clip, clip, nr)
    zi_g = np.linspace(-clip, clip, nr)
    Xi, Zi = np.meshgrid(xi_g, zi_g)
    theta = np.linspace(0, 2 * np.pi, 128)

    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(args.alphas)))

    # --- Convergence curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, alpha in zip(cmap, args.alphas):
        conv = results[alpha]["conv"]
        ax.semilogy(
            np.arange(len(conv)) * dt, conv, color=c, lw=1.5, label=f"α={alpha}"
        )
    ax.axhline(args.tol * phi_mag_max, color="k", ls="--", lw=0.8, label="tol")
    ax.set_xlabel("time")
    ax.set_ylabel("max |Δlam|")
    ax.set_title(f"lam_relax convergence  R={args.R}  rate={args.rate}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "convergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved convergence.png")

    # --- Scatter: lam_converged vs alpha * phi_mag ---
    n_a = len(args.alphas)
    fig, axes = plt.subplots(1, n_a, figsize=(4 * n_a, 4), squeeze=False)
    for ax, (c, alpha) in zip(axes[0], zip(cmap, args.alphas)):
        lam_c = results[alpha]["lam"]
        target = alpha * phi_mag
        lim = max(float(target.max()), float(lam_c.max())) * 1.05 or 1.0
        ax.scatter(target, lam_c, s=1, alpha=0.3, color=c)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlabel("α·|φ|")
        ax.set_ylabel("lam")
        ax.set_title(f"α={alpha}")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
    fig.suptitle("Converged lam vs target α·|φ|", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_lam_vs_target.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved scatter_lam_vs_target.png")

    # --- XZ slices: |phi| and lam for each alpha ---
    y = pos[:, 1]
    y_u = np.sort(np.abs(np.unique(y)))
    ztol = float(y_u[y_u > 1e-6][0]) * 1.2 if (y_u > 1e-6).any() else 1.5
    xz_mask = np.abs(y) < ztol
    xv, zv = pos[xz_mask, 0], pos[xz_mask, 2]

    pm_slice = phi_mag[xz_mask]
    Pm = (
        griddata((xv, zv), pm_slice, (Xi, Zi), method="linear") if len(xv) > 3 else None
    )

    fig, axes = plt.subplots(2, n_a, figsize=(4 * n_a, 8))
    if n_a == 1:
        axes = axes.reshape(2, 1)

    for ai, (c, alpha) in enumerate(zip(cmap, args.alphas)):
        ax_phi = axes[0, ai]
        ax_lam = axes[1, ai]
        lam_slice = results[alpha]["lam"][xz_mask]
        Lm = (
            griddata((xv, zv), lam_slice, (Xi, Zi), method="linear")
            if len(xv) > 3
            else None
        )

        for ax, grid, label in [
            (ax_phi, Pm, f"|φ|  (α={alpha})"),
            (ax_lam, Lm, f"lam  (α={alpha})"),
        ]:
            if grid is not None:
                vmax = np.nanpercentile(grid, 99) or 1e-6
                ax.pcolormesh(
                    Xi, Zi, grid, cmap="viridis", vmin=0, vmax=vmax, shading="auto"
                )
            ax.plot(
                scale * np.cos(theta), scale * np.sin(theta), "w--", lw=0.7, alpha=0.6
            )
            ax.set_aspect("equal")
            ax.set_xlim(-clip, clip)
            ax.set_ylim(-clip, clip)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("x")
        if ai == 0:
            axes[0, 0].set_ylabel("z")
            axes[1, 0].set_ylabel("z")

    fig.suptitle(
        f"XZ: |φ| and converged lam  R={args.R}  rate={args.rate}", fontsize=10
    )
    plt.tight_layout()
    plt.savefig(out_dir / "xz_slices.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved xz_slices.png")

    # --- Radial profiles ---
    r_all = np.linalg.norm(pos, axis=1)
    r_bins = np.linspace(0, args.R * 0.9, 50)
    r_mids = 0.5 * (r_bins[:-1] + r_bins[1:])
    bin_idx = np.digitize(r_all, r_bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(r_mids))

    phi_radial = np.array(
        [
            (
                phi_mag[valid & (bin_idx == b)].mean()
                if (valid & (bin_idx == b)).any()
                else 0.0
            )
            for b in range(len(r_mids))
        ]
    )

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(10, 4))
    for c, alpha in zip(cmap, args.alphas):
        lam_c = results[alpha]["lam"]
        lam_radial = np.array(
            [
                (
                    lam_c[valid & (bin_idx == b)].mean()
                    if (valid & (bin_idx == b)).any()
                    else 0.0
                )
                for b in range(len(r_mids))
            ]
        )
        ax_raw.plot(r_mids, lam_radial, color=c, lw=1.5, label=f"α={alpha}")
        ax_norm.plot(
            r_mids, lam_radial / max(alpha, 1e-8), color=c, lw=1.5, label=f"α={alpha}"
        )

    ax_raw.plot(r_mids, phi_radial, "k--", lw=1.5, label="|φ| mean")
    ax_norm.plot(r_mids, phi_radial, "k--", lw=1.5, label="|φ| mean")
    for ax, title in [
        (ax_raw, "Radial mean lam"),
        (ax_norm, "lam/α  (shape invariance)"),
    ]:
        ax.set_xlabel("r")
        ax.set_ylabel("value")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Converged lam radial profiles  R={args.R}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "radial_profiles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved radial_profiles.png")


if __name__ == "__main__":
    main()
