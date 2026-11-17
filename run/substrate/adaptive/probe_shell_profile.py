"""
Shell circulation profile diagnostic — C++ adaptive lattice, honest IC.

IC: single u-bump at origin (not pinned), lam=0 everywhere, optional tiny phi
noise for symmetry breaking.  step + step_lam + step_cross run together.

step_lam and step_cross are both justified (translation cost is real even
though lam has no existence at Planck scale), but nothing is forced: the
winding must self-organise from the u-gradient or not emerge at all.

Key diagnostics: u_sum (is substrate conserved?), u_orig (does the knot
deplete?), lam_max / lam_orig (does winding accumulate at the right place?).

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_shell_profile.py --R 40
  PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_shell_profile.py \
      --R 40 --u-central 5.0 --phi-noise 1e-4 --n-settle 50000
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from equilibrium.substrate.amr import auto_split_criterion
from tst.substrate.adaptive import compute_dt, find_origin, get_r_physical
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    init_phare_ic,
)


def _plot_snapshot(cl, step, outpath, thickness=2.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.array(cl.leaf_pos)
    phi = np.array(cl.leaf_phi).T
    u = np.array(cl.leaf_u)
    lam = np.array(cl.leaf_lam)
    phi_mag = np.sqrt((phi**2).sum(axis=1))

    r_max = np.linalg.norm(pos, axis=1).max()
    gridsize = max(20, int(r_max * 1.3))

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f"Step {step}", fontsize=12)

    slices = [
        ("XY (z≈0)", np.abs(pos[:, 2]) < thickness, 0, 1, "x", "y"),
        ("XZ (y≈0)", np.abs(pos[:, 1]) < thickness, 0, 2, "x", "z"),
        ("YZ (x≈0)", np.abs(pos[:, 0]) < thickness, 1, 2, "y", "z"),
    ]
    for col, (title, mask, ix, iy, xl, yl) in enumerate(slices):
        px, py = pos[mask, ix], pos[mask, iy]

        ax = axes[0, col]
        hb = ax.hexbin(
            px,
            py,
            C=phi_mag[mask],
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap="inferno",
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — |phi|")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

        ax = axes[1, col]
        um = u[mask]
        vm = max(abs(um.min()), abs(um.max())) or 1.0
        hb = ax.hexbin(
            px,
            py,
            C=um,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap="seismic",
            vmin=-vm,
            vmax=vm,
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — u")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

        ax = axes[2, col]
        hb = ax.hexbin(
            px,
            py,
            C=lam[mask],
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap="viridis",
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"{title} — lam")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  snapshot plot → {outpath}", flush=True)


def phi_toroidal_circulation_adaptive(cl, r_shell, dr=0.8):
    """Toroidal (z-axis) phi circulation at shell r_shell for an AdaptiveCellCluster.

    Upper-triangle via global cell index — each face counted once from
    the cell with the smaller global index.  Level-aware face midpoints
    so AMR cells at level>0 land at the correct radius.
    """
    pos = np.array(cl.leaf_pos)  # (N, 3)
    phi = np.array(cl.leaf_phi).T  # (N, 14) — transpose from (14, N)
    nb_g_arr = np.array(cl.leaf_nb)  # (N, 14) global indices
    leaf_g = np.array(cl.leaf_global_indices)  # (N,)
    lev = np.array(cl.leaf_level, dtype=np.float64)  # (N,)

    # Face midpoint offset: half-hop scaled to cell size (D1=1 at level 0)
    cell_size = 1.0 / (2.0**lev)  # (N,)
    mid_off = 0.5 * _FACE_HOP[None, :, :] * cell_size[:, None, None]  # (N, 14, 3)
    midpoints = pos[:, None, :] + mid_off  # (N, 14, 3)

    r_mid = np.linalg.norm(midpoints, axis=-1)  # (N, 14)
    on_shell = (r_mid >= r_shell - dr) & (r_mid <= r_shell + dr)
    has_nb = nb_g_arr >= 0

    # Upper-triangle: only count from cell with smaller global index
    upper = on_shell & has_nb & (leaf_g[:, None] < nb_g_arr)

    # Azimuthal unit vector at each face midpoint (toroidal around z-axis)
    mx, my = midpoints[:, :, 0], midpoints[:, :, 1]
    rho = np.sqrt(mx**2 + my**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    az_x = -my / rho_s  # (N, 14)
    az_y = mx / rho_s

    # Project face normal (_FACE_NORM shape (14, 3)) onto azimuthal direction
    proj = _FACE_NORM[None, :, 0] * az_x + _FACE_NORM[None, :, 1] * az_y  # (N, 14)

    return float(np.sum(phi[upper] * proj[upper]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=100.0, help="sphere radius")
    parser.add_argument(
        "--u-central",
        type=float,
        default=1.0,
        help="initial u at origin cell (not pinned)",
    )
    parser.add_argument(
        "--phi-noise",
        type=float,
        default=0.0,
        help="amplitude of initial phi noise for symmetry breaking",
    )
    parser.add_argument(
        "--lam-scale", type=float, default=0.05, help="step_lam scale factor"
    )
    parser.add_argument(
        "--cross-scale", type=float, default=0.01, help="step_cross scale factor"
    )
    parser.add_argument(
        "--sponge-frac", type=float, default=0.15, help="outer sponge fraction of R"
    )
    parser.add_argument(
        "--damp-max",
        type=float,
        default=0.003,
        help="max phi damp coefficient per step at boundary",
    )
    parser.add_argument(
        "--n-settle",
        type=int,
        default=20000,
        help="self-organisation steps (all three update rules)",
    )
    parser.add_argument(
        "--n-post", type=int, default=0, help="post-adapt settle (0 = n_settle//2)"
    )
    parser.add_argument(
        "--max-level", type=int, default=0, help="0=uniform, 1=one AMR level near knot"
    )
    parser.add_argument(
        "--split-frac",
        type=float,
        default=0.05,
        help="fraction of leaves to split on adapt",
    )
    parser.add_argument("--n-threads", type=int, default=10)
    parser.add_argument("--n-report", type=int, default=500)
    parser.add_argument(
        "--r-min-fit", type=float, default=None, help="min shell for power-law fit"
    )
    parser.add_argument(
        "--r-max-fit",
        type=float,
        default=None,
        help="max shell for power-law fit (default 0.70*R)",
    )
    parser.add_argument("--plot-out", type=str, default="shell_profile_adaptive.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--save-snapshot",
        type=str,
        default=None,
        help="save lattice to HDF5 file after settling (alat.save)",
    )
    parser.add_argument(
        "--phare-ic",
        type=str,
        default=None,
        help="path to ic_b.npz — use PHARE relaxed-spheromak as phi/u/lam IC",
    )
    parser.add_argument(
        "--phare-scale",
        type=float,
        default=20.0,
        help="DST units per npz unit (default 20: ±5 npz → ±100 DST)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="restart from HDF5 snapshot saved by --save-snapshot",
    )
    parser.add_argument(
        "--plot-cadence",
        type=int,
        default=0,
        help="generate a snapshot plot every N steps (0 = disabled)",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="snap",
        help="path prefix for cadence plots, e.g. 'out/ring'",
    )
    parser.add_argument(
        "--step-offset",
        type=int,
        default=0,
        help="global step offset added to local step for plot filenames",
    )
    args = parser.parse_args()

    knot_r = 2.0 / np.sqrt(3)  # ≈ 1.155 DST units at D1=1

    # ------------------------------------------------------------------ build or load
    if args.load:
        print(f"Loading snapshot {args.load} ...", flush=True)
        alat = _lat.AdaptiveLattice.load(args.load)
        alat.set_n_threads(args.n_threads)
        cl = alat[0]
        print(f"  N={cl.n_leaves}  max_level={cl.info.max_level}", flush=True)
    else:
        print(f"Building BCC sphere R={args.R} ...", flush=True)
        alat = _lat.AdaptiveLattice()
        alat.set_n_threads(args.n_threads)
        cl = alat[0]
        cl.info.max_level = args.max_level
        _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, lam0=0)
        alat.rebuild(0)
        print(f"  N={cl.n_leaves}  max_level={args.max_level}", flush=True)

        # ------------------------------------------------------------------ IC
        origin_j = find_origin(cl)

        if args.phare_ic:
            b_max = init_phare_ic(
                cl,
                args.phare_ic,
                scale=args.phare_scale,
                lam_target=9.0,
                u_max=1.0,
                amplitude=1.0,
            )
            print(
                f"  PHARE IC: {args.phare_ic}  scale={args.phare_scale}"
                f"  B_max={b_max:.4g}",
                flush=True,
            )
        else:
            u_arr = np.zeros(cl.n_leaves, dtype=np.float64)
            u_arr[origin_j] = args.u_central
            cl.set_leaf_u(u_arr)
            if args.phi_noise > 0.0:
                rng = np.random.default_rng(42)
                phi_ic = rng.uniform(-args.phi_noise, args.phi_noise, (14, cl.n_leaves))
                cl.set_leaf_phi(phi_ic)
            print(
                f"  u-knot IC (no forced lam, no pinned u)  origin_j={origin_j}"
                f"  u_central={args.u_central}  phi_noise={args.phi_noise}"
                f"  knot_r≈{knot_r:.3f}",
                flush=True,
            )

        cl.set_bc(_lat.BoundaryCondition.absorbing())

    origin_j = find_origin(cl)

    # Sponge mask: cells in the outer sponge_fraction of the radius.
    # Phi is damped directly each step in this region — equivalent to what
    # DampedLattice does.  Without this, step_cross turns the finite domain
    # into a phi pump (outgoing phi has nowhere to go).
    r_all = get_r_physical(cl)
    sponge_inner = args.R * (1.0 - args.sponge_frac)
    sponge_mask = r_all >= sponge_inner  # (N,) bool
    # Quadratic damp coefficient: 0 at inner edge, damp_max at outer edge
    ramp = np.clip((r_all - sponge_inner) / (args.R - sponge_inner), 0.0, 1.0)
    damp_phi = (args.damp_max * ramp**2).astype(np.float64)  # (N,)
    print(
        f"  sponge r>[{sponge_inner:.1f}, {args.R:.1f}]"
        f"  n_sponge={sponge_mask.sum()}  damp_max={args.damp_max}",
        flush=True,
    )

    # ------------------------------------------------------------------ settle (uniform)
    dt = compute_dt(cl)
    print(
        f"  dt={dt:.4e}  self-organising {args.n_settle} steps"
        f"  (lam_scale={args.lam_scale}  cross_scale={args.cross_scale}) ...",
        flush=True,
    )
    t0 = time.time()
    for s in range(1, args.n_settle + 1):
        alat.step(0, dt)
        alat.step_lam(0, dt, args.lam_scale)
        if args.cross_scale > 0.0:
            alat.step_cross(0, dt, args.cross_scale)
        # Direct phi damping in sponge shell — absorbs outgoing phi so step_cross
        # cannot pump the domain indefinitely.
        phi = np.array(cl.leaf_phi)  # (14, N)
        phi[:, sponge_mask] *= 1.0 - damp_phi[sponge_mask]
        cl.set_leaf_phi(phi)
        if s % args.n_report == 0:
            u_arr = np.array(cl.leaf_u)
            lam_arr = np.array(cl.leaf_lam)
            phi_arr = np.array(cl.leaf_phi)
            u_sum = float(u_arr.sum())
            u_orig = float(u_arr[origin_j])
            lam_max = float(lam_arr.max())
            lam_orig = float(lam_arr[origin_j])
            phi_e = float(np.mean(phi_arr**2))
            phi_max = float(np.abs(phi_arr).max())
            print(
                f"  {s}/{args.n_settle}  u_sum={u_sum:.4e}  u_orig={u_orig:.4e}"
                f"  lam_max={lam_max:.4e}  lam_orig={lam_orig:.4e}"
                f"  phi_E={phi_e:.4e}  phi_max={phi_max:.4e}",
                flush=True,
            )
        if args.plot_cadence > 0 and s % args.plot_cadence == 0:
            g = args.step_offset + s
            _plot_snapshot(cl, g, f"{args.plot_prefix}_{g:06d}.png")
    print(f"  self-organisation done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------ AMR adapt
    if args.max_level >= 1:
        print("Running AMR adapt ...", flush=True)
        sc = auto_split_criterion(cl, split_frac=args.split_frac, coarsen_factor=0.1)
        print(
            f"  criterion: hi={sc.hi:.4f}  lo={sc.lo:.4f}  ref={sc.ref:.4f}", flush=True
        )
        cl.set_split_criterion(sc)
        alat.adapt(0)
        alat.rebuild(0)
        print(f"  N after adapt={cl.n_leaves}", flush=True)

        dt = compute_dt(cl)
        n_post = args.n_post if args.n_post > 0 else args.n_settle // 2
        print(f"  new dt={dt:.4e}  post-adapt settle {n_post} steps ...", flush=True)
        t0 = time.time()
        for s in range(1, n_post + 1):
            alat.step(0, dt)
            alat.step_lam(0, dt, args.lam_scale)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale)
            phi = np.array(cl.leaf_phi)
            phi[:, sponge_mask] *= 1.0 - damp_phi[sponge_mask]
            cl.set_leaf_phi(phi)
            if s % args.n_report == 0:
                u_arr = np.array(cl.leaf_u)
                lam_arr = np.array(cl.leaf_lam)
                phi_arr = np.array(cl.leaf_phi)
                u_sum = float(u_arr.sum())
                u_orig = float(u_arr[origin_j])
                lam_max = float(lam_arr.max())
                lam_orig = float(lam_arr[origin_j])
                phi_e = float(np.mean(phi_arr**2))
                phi_max = float(np.abs(phi_arr).max())
                print(
                    f"  {s}/{n_post}  n_leaves={cl.n_leaves}"
                    f"  u_sum={u_sum:.4e}  u_orig={u_orig:.4e}"
                    f"  lam_max={lam_max:.4e}  lam_orig={lam_orig:.4e}"
                    f"  phi_E={phi_e:.4e}  phi_max={phi_max:.4e}",
                    flush=True,
                )
        print(f"  post-adapt settle done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------ snapshot
    if args.save_snapshot:
        alat.save(args.save_snapshot)
        print(f"snapshot saved → {args.save_snapshot}", flush=True)

    # ------------------------------------------------------------------ shell profile
    r_min_fit = args.r_min_fit if args.r_min_fit is not None else knot_r + 2.0
    r_max_fit = args.r_max_fit if args.r_max_fit is not None else args.R * 0.70
    r_shells = np.arange(1, int(args.R * 0.82) + 1, dtype=float)

    print(
        f"\nShell circulation profile  (fit range r=[{r_min_fit:.1f}, {r_max_fit:.1f}])"
    )
    print(f"{'r':>5}  {'circ':>12}  {'in fit?'}")
    print("  " + "-" * 30)

    circs = []
    for r in r_shells:
        c = phi_toroidal_circulation_adaptive(cl, r)
        in_fit = r_min_fit <= r <= r_max_fit
        print(f"{r:5.1f}  {c:+12.6e}  {'*' if in_fit else ''}")
        circs.append(c)
    circs = np.array(circs)

    fit_mask = (r_shells >= r_min_fit) & (r_shells <= r_max_fit) & (circs > 0)
    if fit_mask.sum() < 3:
        print("\nInsufficient positive exterior shells for power-law fit.")
        n_fit = A_fit = None
    else:
        log_r = np.log(r_shells[fit_mask])
        log_c = np.log(circs[fit_mask])
        coeffs = np.polyfit(log_r, log_c, 1)
        n_fit = -coeffs[0]
        A_fit = np.exp(coeffs[1])
        resid = log_c - np.polyval(coeffs, log_r)
        r2 = 1.0 - resid.var() / log_c.var()
        print(f"\nPower-law fit  circ(r) = {A_fit:.4e} * r^(-{n_fit:.3f})")
        print(
            f"  n={n_fit:.3f}  (dipole=2.000)  R²={r2:.4f}  over {fit_mask.sum()} shells"
        )

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    tag = f"R={args.R}  max_level={args.max_level}  N={cl.n_leaves}"
    fig.suptitle(f"Shell φ circulation — {tag}", fontsize=10)

    ax = axes[0]
    ext = r_shells >= knot_r
    inn = r_shells < knot_r + 0.5
    ax.plot(r_shells[inn], circs[inn], "o-", color="steelblue", ms=4, label="interior")
    ax.plot(r_shells[ext], circs[ext], "o-", color="tomato", ms=4, label="exterior")
    if n_fit is not None:
        r_line = r_shells[(r_shells >= r_min_fit) & (r_shells <= r_max_fit)]
        ax.plot(
            r_line,
            A_fit * r_line ** (-n_fit),
            "k--",
            lw=1,
            label=f"fit r^(-{n_fit:.2f})",
        )
    ax.axvline(knot_r, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("r (shells)")
    ax.set_ylabel("φ toroidal circulation")
    ax.set_title("Linear scale")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    pos_mask = circs > 0
    ax2.loglog(
        r_shells[pos_mask], circs[pos_mask], "o", color="tomato", ms=4, label="circ > 0"
    )
    neg_mask = circs < 0
    ax2.loglog(
        r_shells[neg_mask],
        -circs[neg_mask],
        "s",
        color="steelblue",
        ms=4,
        label="-circ < 0",
    )
    if n_fit is not None:
        r_ref = np.array([r_min_fit, r_max_fit])
        ax2.loglog(
            r_ref, A_fit * r_ref ** (-n_fit), "k--", lw=1.2, label=f"r^(-{n_fit:.2f})"
        )
        ax2.loglog(
            r_ref, A_fit * r_ref ** (-2.0), "k:", lw=1.0, alpha=0.5, label="r^(-2) ref"
        )
    ax2.axvline(knot_r, color="k", lw=0.7, ls=":", alpha=0.5)
    ax2.set_xlabel("r (shells)")
    ax2.set_ylabel("|φ toroidal circulation|")
    ax2.set_title("Log-log (fit vs dipole reference)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved → {args.plot_out}")


if __name__ == "__main__":
    main()
