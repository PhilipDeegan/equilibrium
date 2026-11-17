"""
Shell circulation profile diagnostic.

Two modes:

  --load snapshot.npz
      Load a pre-existing proton candidate, optionally settle, then measure.

  (no --load)
      Build a fresh DampedLattice with a single u-bump at origin (not pinned,
      no forced lam, optional tiny phi noise for symmetry breaking), run all
      three steps — step + step_lam + step_cross — and measure.

      This is the physically honest IC: if a toroidal winding self-organises
      from the u-gradient it comes from the lattice dynamics alone.

Key diagnostics: u_sum (substrate conservation), u_orig (knot depletion),
lam_orig (does winding accumulate at origin?), lam_max (where?).

Run (honest IC):
  PYTHONPATH=$PWD python3 run/substrate/probe_shell_profile.py \
    --R 40 --u-central 5.0 --phi-noise 1e-4 \
    --n-settle 50000 --n-report 1000 --n-threads 10

Run (from snapshot):
  PYTHONPATH=$PWD python3 run/substrate/probe_shell_profile.py \
    --load proton_candidate_R40.npz --R 40
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.flux_tube import phi_toroidal_circulation
from equilibrium.substrate.lattice import DampedLattice


def _plot_snapshot(lat, step, outpath, thickness=2.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = lat.pos
    phi_mag = np.sqrt((lat.phi**2).sum(axis=1))
    u = lat.u
    lam = lat.lam

    R_approx = float(np.linalg.norm(pos, axis=1).max())
    gridsize = max(40, int(R_approx * 2))

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
    print(f"  snapshot → {outpath}", flush=True)


def _load_phare_ic(lat, npz_path, scale, lam_target, u_max):
    """Set u and lam from the PHARE relaxed-spheromak |B| field. phi left at zero."""
    from scipy.interpolate import RegularGridInterpolator

    from equilibrium.substrate.discrete import LAM_MAX

    d = np.load(npz_path)

    def _interp(arr):
        return RegularGridInterpolator(
            (d["x"], d["y"], d["z"]),
            arr,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    ib = [_interp(d["bx"]), _interp(d["by"]), _interp(d["bz"])]
    pts = lat.pos / scale
    B = np.stack([ib[k](pts) for k in range(3)], axis=1)
    b_mag = np.linalg.norm(B, axis=1)
    b_max = float(b_mag.max()) or 1.0

    lat.lam[:] = np.minimum(lam_target * b_mag / b_max, LAM_MAX)
    lat.u[:] = u_max * b_mag / b_max
    return b_mag, b_max


parser = argparse.ArgumentParser()
parser.add_argument(
    "--load", type=str, default=None, help="snapshot .npz; omit to build from scratch"
)
parser.add_argument(
    "--phare-ic",
    type=str,
    default=None,
    help="PHARE ic_b.npz — sets u/lam from |B|, phi=0",
)
parser.add_argument(
    "--phare-scale",
    type=float,
    default=20.0,
    help="DST units per npz unit (default 20: ±5 npz → ±100 DST)",
)
parser.add_argument(
    "--lam-target", type=float, default=9.0, help="peak lam for PHARE/build IC"
)
parser.add_argument("--R", type=float, default=40.0, help="lattice sphere radius")
parser.add_argument(
    "--u-central",
    type=float,
    default=1.0,
    help="initial u at origin (build mode, not pinned)",
)
parser.add_argument(
    "--phi-noise",
    type=float,
    default=0.0,
    help="initial phi noise amplitude for symmetry breaking",
)
parser.add_argument(
    "--n-settle", type=int, default=0, help="settle steps before measuring"
)
parser.add_argument(
    "--n-report", type=int, default=500, help="print diagnostics every N settle steps"
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument(
    "--r-max-fit",
    type=float,
    default=None,
    help="max shell for power-law fit (default 0.70*R)",
)
parser.add_argument(
    "--r-min-fit",
    type=float,
    default=None,
    help="min shell for power-law fit (default ring_r+2)",
)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--plot-out", type=str, default="shell_profile.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument(
    "--plot-cadence",
    type=int,
    default=0,
    help="save snapshot plot every N settle steps (0=disabled)",
)
parser.add_argument(
    "--plot-prefix",
    type=str,
    default="tmp/snap",
    help="path prefix for cadence plots, e.g. tmp/shell_snap",
)


def main():
    args = parser.parse_args()

    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    r_all = np.linalg.norm(lat.pos, axis=1)
    centre = np.zeros(3)

    origin_idx = int(np.argmin(r_all))

    if args.load:
        data = np.load(args.load)
        lat.u[:] = data["u"]
        lat.phi[:] = data["phi"]
        lat.lam[:] = data["lam"]
        A_tiles = list(data["A_tiles"].astype(int))
        B_tiles = list(data["B_tiles"].astype(int)) if "B_tiles" in data else []
        ring_r = float(np.mean([r_all[i] for i in A_tiles + B_tiles]))
        print(f"Loaded {args.load}: N={lat.N}  R={args.R}  ring_r≈{ring_r:.2f}")
    elif args.phare_ic:
        b_mag, b_max = _load_phare_ic(
            lat,
            args.phare_ic,
            scale=args.phare_scale,
            lam_target=args.lam_target,
            u_max=1.0,
        )
        # ring_r: radius of peak |B| away from origin
        away = r_all > 1.0
        ring_r = float(r_all[away][b_mag[away].argmax()])
        print(
            f"PHARE IC {args.phare_ic}: N={lat.N}  R={args.R}"
            f"  scale={args.phare_scale}  b_max={b_max:.4g}  ring_r≈{ring_r:.2f}"
        )
    else:
        lat.u[origin_idx] = args.u_central
        if args.phi_noise > 0.0:
            rng = np.random.default_rng(42)
            lat.phi[:] = rng.uniform(-args.phi_noise, args.phi_noise, lat.phi.shape)
        ring_r = 1.0  # first hex-face shell from origin
        print(
            f"Build mode: N={lat.N}  R={args.R}  u_central={args.u_central}"
            f"  phi_noise={args.phi_noise}  origin_idx={origin_idx}"
        )

    if args.n_settle > 0:
        dt = lat.stable_dt()
        print(
            f"Settling {args.n_settle} steps"
            f"  (lam_scale={args.lam_scale}  cross_scale={args.cross_scale}"
            f"  n_threads={args.n_threads}) ..."
        )
        for s in range(1, args.n_settle + 1):
            lat.step(dt, n_threads=args.n_threads)
            lat.step_lam(dt, scale=args.lam_scale)
            if args.cross_scale > 0.0:
                lat.step_cross(dt, scale=args.cross_scale)
            if s % args.n_report == 0:
                u_sum = float(lat.u.sum())
                u_orig = float(lat.u[origin_idx])
                lam_orig = float(lat.lam[origin_idx])
                lam_max = float(lat.lam.max())
                phi_e = float(np.mean(lat.phi**2))
                phi_max = float(np.abs(lat.phi).max())
                print(
                    f"  {s}/{args.n_settle}  u_sum={u_sum:.4e}  u_orig={u_orig:.4e}"
                    f"  lam_orig={lam_orig:.4e}  lam_max={lam_max:.4e}"
                    f"  phi_E={phi_e:.4e}  phi_max={phi_max:.4e}",
                    flush=True,
                )
            if args.plot_cadence > 0 and s % args.plot_cadence == 0:
                _plot_snapshot(lat, s, f"{args.plot_prefix}_{s:06d}.png")
        print("Done.")

    r_min_fit = args.r_min_fit if args.r_min_fit is not None else ring_r + 2.0
    r_max_fit = args.r_max_fit if args.r_max_fit is not None else args.R * 0.70
    r_shells = np.arange(1, int(args.R * 0.82) + 1, dtype=float)

    print(
        f"\nShell circulation profile  (fit range r=[{r_min_fit:.1f}, {r_max_fit:.1f}])"
    )
    print(f"{'r':>5}  {'circ':>12}  {'in fit?'}")
    print("  " + "-" * 30)

    circs = []
    for r in r_shells:
        c = phi_toroidal_circulation(lat, centre, r)
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

    title = args.load if args.load else f"build  u_central={args.u_central}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Shell φ circulation — {title}  R={args.R}", fontsize=10)

    ax = axes[0]
    ext = r_shells >= ring_r
    inn = r_shells < ring_r + 0.5
    ax.plot(r_shells[inn], circs[inn], "o-", color="steelblue", ms=4, label="interior")
    ax.plot(r_shells[ext], circs[ext], "o-", color="tomato", ms=4, label="exterior")
    if n_fit is not None:
        r_fit_line = r_shells[(r_shells >= r_min_fit) & (r_shells <= r_max_fit)]
        ax.plot(
            r_fit_line,
            A_fit * r_fit_line ** (-n_fit),
            "k--",
            lw=1,
            label=f"fit r^(-{n_fit:.2f})",
        )
    ax.axvline(ring_r, color="k", lw=0.7, ls=":", alpha=0.5)
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
    ax2.axvline(ring_r, color="k", lw=0.7, ls=":", alpha=0.5)
    ax2.set_xlabel("r (shells)")
    ax2.set_ylabel("|φ toroidal circulation|")
    ax2.set_title("Log-log (fit vs dipole reference)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved → {args.plot_out}")


if __name__ == "__main__":
    main()
