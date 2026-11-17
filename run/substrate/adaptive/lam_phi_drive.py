"""
Test: lam driving term comparison — all three modes in one run.

The Taylor spheromak IC has ∇·B = 0, so step_lam (driven by div(phi)) gives
Δlam = 0 everywhere — lam is frozen indefinitely.

Runs all three driving modes on independent lattice copies:
  div   — step_lam only (div(phi) driven, current default)
  relax — step_lam_relax: local relaxation lam → alpha*|phi|  (no transport)
  both  — both terms combined

Physical basis for step_lam_relax:
  lam is a proxy for substrate displacement by the local phi field.
  Where field is strong, more substrate is displaced (lam high).
  Where field collapses, lam drains. No transport between cells.

Hypothesis: as phi relaxes toward a dipole, lam should co-evolve toward a knot
(geometric minimum). Tracked via:
  - phi_dipole_frac: fraction of phi energy in the dipole mode (should rise)
  - lam_kurtosis:    excess kurtosis of lam distribution (should rise = more peaked)
  - lam_top5_frac:   fraction of total lam in top-5% cells (direct concentration)

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/lam_phi_drive.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/lam_phi_drive.py \\
      --lam-relax-rate 0.1 --lam-alpha 4.0 --n-steps 5000
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from tst.substrate.adaptive.adaptive_flux_tube import init_spheromak_ic

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=int, default=40)
parser.add_argument("--sph-frac", type=float, default=0.5)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--sigma", type=float, default=None)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument(
    "--lam-relax-rate",
    type=float,
    default=0.1,
    help="rate for step_lam_relax: dlam/dt = rate*(alpha*|phi| - lam)",
)
parser.add_argument(
    "--lam-alpha",
    type=float,
    default=1.0,
    help="phi→lam coupling: lam_target = alpha * |phi|",
)
parser.add_argument("--cross-scale", type=float, default=0.005)
parser.add_argument("--n-steps", type=int, default=10000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--threads", type=int, default=10)
parser.add_argument("--out-dir", type=str, default="tmp/lam_phi_drive")

MODES = ["div", "relax", "both"]

# BCC face directions (14 faces: 8 hex + 6 sq) — mirror bcc_geometry.hpp
_HEX = np.array(
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
_SQ = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=np.float64,
)
FACE_DIRS = np.vstack([_HEX, _SQ])  # (14, 3)


def _precompute_nb_local(cl):
    gid = np.array(cl.leaf_global_indices, dtype=np.int64)
    nb_g = np.array(cl.leaf_nb, dtype=np.int64)
    g_to_j = np.full(int(gid.max()) + 1, -1, dtype=np.int64)
    g_to_j[gid] = np.arange(len(gid), dtype=np.int64)
    safe = np.clip(nb_g, 0, len(g_to_j) - 1)
    return np.where(nb_g >= 0, g_to_j[safe], -1)


def _compute_diags(cl):
    """Return dict of scalar diagnostics for the current lattice state."""
    phi = cl.leaf_phi  # (14, N)
    lam = cl.leaf_lam  # (N,)

    # --- lam metrics ---
    lam_mean = lam.mean()
    lam_std = lam.std()
    lam_max = lam.max()
    if lam_std > 1e-12:
        lam_kurt = float(np.mean(((lam - lam_mean) / lam_std) ** 4)) - 3.0
    else:
        lam_kurt = 0.0
    total_lam = lam.sum()
    thresh = np.percentile(lam, 95)
    lam_top5 = float(lam[lam >= thresh].sum() / (total_lam + 1e-30))

    # --- phi → B reconstruction ---
    # B_i ≈ Σ_k phi[k,i] * face_dir[k]  (flux-weighted direction sum)
    B = np.einsum("kn,kd->nd", phi, FACE_DIRS)  # (N, 3)

    phi_E = float(np.sum(phi**2))

    # Dipole moment: m = Σ_i B_i  (should be zero for symmetric spheromak)
    m = B.sum(axis=0)  # (3,)
    phi_dipole_frac = float(np.dot(m, m) / (phi_E * len(lam) + 1e-30))

    # phi concentration: max |B| per cell / mean |B|
    B_mag = np.linalg.norm(B, axis=1)  # (N,)
    phi_B_max = float(B_mag.max())
    phi_B_mean = float(B_mag.mean() + 1e-30)

    return dict(
        lam_std=float(lam_std),
        lam_max=float(lam_max),
        lam_kurt=float(lam_kurt),
        lam_top5=float(lam_top5),
        lam_total=float(total_lam),
        phi_E=float(phi_E),
        phi_dipole_frac=float(phi_dipole_frac),
        phi_B_max=float(phi_B_max),
        phi_B_ratio=float(phi_B_max / phi_B_mean),
    )


def build_lattice(args):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)
    scale = args.R * args.sph_frac
    init_spheromak_ic(
        cl,
        R_sph=1.0,
        scale=scale,
        lam_target=args.lam_target,
        u_max=args.u_max,
        amplitude=args.amplitude,
        sigma=args.sigma,
    )
    return alat, cl


def run_mode(args, mode):
    print(f"\n--- mode={mode} ---")
    alat, cl = build_lattice(args)
    dt = float(1.0 / (cl.n_leaves ** (1.0 / 3.0))) * 0.5

    lam0 = cl.leaf_lam.copy()
    steps = []
    hist = {
        k: []
        for k in [
            "lam_std",
            "lam_max",
            "lam_kurt",
            "lam_top5",
            "phi_dipole_frac",
            "phi_B_ratio",
        ]
    }
    snap = {0: lam0.copy()}

    print(f"N={cl.n_leaves}  lam0: max={lam0.max():.4f}  std={lam0.std():.4f}")
    print(f"  alpha={args.lam_alpha}  relax_rate={args.lam_relax_rate}")
    print(
        f"{'step':>7}  {'lam_max':>9}  {'lam_kurt':>9}  {'lam_top5':>9}  {'phi_dip':>9}  {'phi_Br':>9}"
    )

    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            d = _compute_diags(cl)
            steps.append(step)
            for k in hist:
                hist[k].append(d[k])
            print(
                f"{step:7d}  {d['lam_max']:9.4f}  {d['lam_kurt']:9.4f}"
                f"  {d['lam_top5']:9.4f}  {d['phi_dipole_frac']:9.4e}  {d['phi_B_ratio']:9.4f}"
            )

        if step == args.n_steps:
            snap[step] = cl.leaf_lam.copy()
            break

        alat.step(0, dt)
        alat.step_sponge(0)
        if args.cross_scale > 0:
            alat.step_cross(0, dt, args.cross_scale)
        if mode in ("div", "both"):
            alat.step_lam(0, dt, args.lam_scale)
        if mode in ("relax", "both"):
            step_lam_relax(cl, dt, args.lam_relax_rate, args.lam_alpha)

    pos = cl.leaf_pos
    return {"steps": steps, "hist": hist, "snap": snap, "pos": pos}


def comparison_plot(results, args, out_dir):
    from scipy.interpolate import griddata

    colors = {"div": "steelblue", "mag": "tomato", "both": "seagreen"}

    # --- co-evolution timeseries ---
    diag_panels = [
        ("lam_max", "lam max"),
        ("lam_kurt", "lam kurtosis (excess)"),
        ("lam_top5", "lam top-5% fraction"),
        ("phi_dipole_frac", "phi dipole fraction"),
        ("phi_B_ratio", "|B|max / |B|mean"),
    ]
    fig, axes = plt.subplots(
        len(diag_panels), 1, figsize=(10, 3 * len(diag_panels)), constrained_layout=True
    )
    fig.suptitle(
        f"Lam + phi co-evolution  (lam_scale={args.lam_scale}, "
        f"relax_rate={args.lam_relax_rate}, alpha={args.lam_alpha})",
        fontsize=11,
    )
    for ax, (key, label) in zip(axes, diag_panels):
        for mode in MODES:
            r = results[mode]
            ax.plot(r["steps"], r["hist"][key], color=colors[mode], lw=1.8, label=mode)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.savefig(out_dir / "coevolution.png", dpi=100)
    plt.close(fig)

    # --- midplane lam slices: IC / final ---
    pos0 = results[MODES[0]]["pos"]
    z0 = pos0[:, 2]
    m0 = np.abs(z0 - z0.mean()) < (z0.max() - z0.min()) / 6.0
    x0, y0 = pos0[m0, 0], pos0[m0, 1]
    xi = np.linspace(x0.min(), x0.max(), 150)
    yi = np.linspace(y0.min(), y0.max(), 150)
    Xg, Yg = np.meshgrid(xi, yi)
    lam_ic = results[MODES[0]]["snap"][0]
    vmax = max(lam_ic[m0].max(), 1e-6)

    fig2, axes2 = plt.subplots(
        2, len(MODES), figsize=(5 * len(MODES), 9), constrained_layout=True
    )
    fig2.suptitle("Lam midplane: IC (top) / final (bottom)", fontsize=11)
    for col, mode in enumerate(MODES):
        r = results[mode]
        pos = r["pos"]
        z = pos[:, 2]
        m = np.abs(z - z.mean()) < (z.max() - z.min()) / 6.0
        xm, ym = pos[m, 0], pos[m, 1]
        for row, sk in enumerate([0, args.n_steps]):
            Lg = griddata((xm, ym), r["snap"][sk][m], (Xg, Yg), method="linear")
            im = axes2[row, col].imshow(
                Lg,
                origin="lower",
                aspect="auto",
                extent=[xi[0], xi[-1], yi[0], yi[-1]],
                cmap="viridis",
                vmin=0,
                vmax=vmax,
            )
            axes2[row, col].set_title(
                f"{mode}  {'IC' if sk==0 else f'step {sk}'}", fontsize=9
            )
            axes2[row, col].set_xlabel("x")
            axes2[row, col].set_ylabel("y")
            fig2.colorbar(im, ax=axes2[row, col], shrink=0.8)

    fig2.savefig(out_dir / "slices.png", dpi=100)
    plt.close(fig2)
    print(f"\nplots → {out_dir}/coevolution.png  /  {out_dir}/slices.png")


def main():
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode in MODES:
        results[mode] = run_mode(args, mode)

    comparison_plot(results, args, out_dir)


if __name__ == "__main__":
    main()
