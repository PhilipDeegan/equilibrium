"""
DST spheromak → dipole decay experiment.

Fully analytical ICs — no external data files required:
  phi  : Taylor force-free spheromak  (j₁(kr)/r projected onto face fluxes)
  lam  : Taylor u-profile              (j₁(kr)/r, peaked at r≈R/2.16)

step_cross drives topology change (DST analogue of resistivity).
step_lam allows substrate tension to redistribute.

The expected sequence mirrors PHARE t=0→3:
  1. Interior circulation (r < R_sph) decays as winding topology opens
  2. Exterior circulation (r > R_sph) grows as the dipole field emerges
  3. H plateaus when the dipole remnant stabilises

Run:
    PYTHONPATH=$PWD python3 run/substrate/dst_spheromak_to_dipole.py
    PYTHONPATH=$PWD python3 run/substrate/dst_spheromak_to_dipole.py --cs 0.003 --R 8
    PYTHONPATH=$PWD python3 run/substrate/dst_spheromak_to_dipole.py --plot-cadence 5000
"""

import argparse
import sys

import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice

# ---------------------------------------------------------------------------
# Taylor spheromak phi IC (inline — no dependency on explore_taylor.py)
# ---------------------------------------------------------------------------


def init_taylor_spheromak_phi(lat: SubstrateLattice, centre=None, R=5.0, B0=1.0) -> int:
    if centre is None:
        centre = np.zeros(3, dtype=float)
    centre = np.asarray(centre, dtype=float)
    k = 4.493 / R
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
    pos_i = lat.pos[:, np.newaxis, :]
    pos_j = lat.pos[nb_safe]
    mid = 0.5 * (pos_i + pos_j) - centre
    dx, dy, dz = mid[..., 0], mid[..., 1], mid[..., 2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    in_sphere = lat._valid & (r > 1e-10) & (r <= R)
    r_s = np.where(in_sphere, r, 1.0)
    rho = k * r_s
    j1r = spherical_jn(1, rho)
    comm = np.where(in_sphere, B0 * j1r / r_s, 0.0)
    bx = comm * (dz * dx / r_s - dy)
    by = comm * (dz * dy / r_s + dx)
    bz = comm * (-(dx**2 + dy**2) / r_s)
    diff = pos_j - pos_i
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)
    lat.phi = np.where(
        in_sphere, bx * n_hat[..., 0] + by * n_hat[..., 1] + bz * n_hat[..., 2], 0.0
    )
    return int(in_sphere.sum())


# ---------------------------------------------------------------------------
# Toroidal phi circulation diagnostic (inline — matches test_spheromak_decay.py)
# ---------------------------------------------------------------------------


def _phi_circulation(lat, centre, r_hop, dr=0.8):
    c = np.asarray(centre, dtype=float)
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
    pos_j = lat.pos[nb_safe]
    pos_i = lat.pos[:, np.newaxis, :]
    mid = 0.5 * (pos_i + pos_j) - c
    r_mid = np.linalg.norm(mid, axis=-1)
    on_shell = lat._valid & (r_mid >= r_hop - dr) & (r_mid <= r_hop + dr)
    upper = on_shell & (np.arange(len(lat.pos))[:, None] < nb_safe)
    mx, my = mid[..., 0], mid[..., 1]
    rho = np.sqrt(mx**2 + my**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    diff = pos_j - pos_i
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)
    proj = n_hat[..., 0] * (-my / rho_s) + n_hat[..., 1] * (mx / rho_s)
    return float(np.sum(lat.phi[upper] * proj[upper]))


# ---------------------------------------------------------------------------
# Analytical lam IC: Taylor u-profile  u(r) = j₁(kr)/r
# ---------------------------------------------------------------------------


def init_taylor_lam(lat, R, lam_peak=5.0, lam_floor=0.0):
    k = 4.493 / R
    r = np.linalg.norm(lat.pos, axis=1)
    lam = np.full(lat.N, lam_floor)
    mask = (r > 1e-6) & (r <= R)
    vals = spherical_jn(1, k * r[mask]) / r[mask]
    vals = np.clip(vals, 0.0, None)
    if vals.max() > 0:
        lam[mask] = lam_floor + lam_peak * vals / vals.max()
    lat.lam = lam
    print(
        f"  lam range [{lat.lam.min():.4f}, {lat.lam.max():.4f}]"
        f"  peak at r≈{r[lam.argmax()]:.2f} cells"
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_snapshot(lat, step, R_sph, outpath, thickness=2.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = lat.pos  # (N, 3)
    phi_mag = np.sqrt((lat.phi**2).sum(axis=1))  # (N,) — magnitude per tile
    u = lat.u  # (N,)
    lam = lat.lam  # (N,)

    # ~1 hex per 1.5 lattice cells across the full diameter
    r_max = np.linalg.norm(pos, axis=1).max()
    gridsize = max(20, int(r_max * 1.3))

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    fig.suptitle(f"Step {step}  R_sph={R_sph}", fontsize=11)

    slices = [
        ("XY (z≈0)", np.abs(pos[:, 2]) < thickness, 0, 1, "x", "y"),
        ("XZ (y≈0)", np.abs(pos[:, 1]) < thickness, 0, 2, "x", "z"),
        ("YZ (x≈0)", np.abs(pos[:, 0]) < thickness, 1, 2, "y", "z"),
    ]

    rows = [
        (phi_mag, "inferno", "|φ|", None, None),
        (u, "seismic", "u", None, None),
        (lam, "viridis", "lam", None, None),
    ]

    for row_idx, (data, cmap, label, vmin, vmax) in enumerate(rows):
        if cmap == "seismic":
            vm = max(abs(data.min()), abs(data.max())) or 1.0
            vmin, vmax = -vm, vm
        for col_idx, (title, mask, ix, iy, xl, yl) in enumerate(slices):
            ax = axes[row_idx, col_idx]
            px, py = pos[mask, ix], pos[mask, iy]
            d = data[mask]
            if len(px) < 3:
                ax.set_visible(False)
                continue
            hb = ax.hexbin(
                px,
                py,
                C=d,
                reduce_C_function=np.mean,
                gridsize=gridsize,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(hb, ax=ax)
            ax.set_title(f"{title} — {label}")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  snapshot → {outpath}", flush=True)


def _plot_summary(
    history, lat, r_all, R_sph, centre, outpath, r_min_fit=None, r_max_fit=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array([h[0] for h in history])
    H_hist = np.array([h[1] for h in history])
    cin_hist = np.array([h[4] for h in history])
    cout_hist = np.array([h[5] for h in history])

    # Dense shell circulation at final state
    r_max_plot = int(r_all.max() * 0.80)
    r_shells = np.arange(1, r_max_plot + 1, dtype=float)
    circs = np.array([_phi_circulation(lat, centre, r) for r in r_shells])

    # Power-law fit on exterior positive shells
    if r_min_fit is None:
        r_min_fit = R_sph + 1.0
    if r_max_fit is None:
        r_max_fit = r_all.max() * 0.65

    fit_mask = (r_shells >= r_min_fit) & (r_shells <= r_max_fit) & (circs > 0)
    n_fit = A_fit = None
    if fit_mask.sum() >= 3:
        log_r = np.log(r_shells[fit_mask])
        log_c = np.log(circs[fit_mask])
        coeffs = np.polyfit(log_r, log_c, 1)
        n_fit = -coeffs[0]
        A_fit = np.exp(coeffs[1])
        resid = log_c - np.polyval(coeffs, log_r)
        r2 = 1.0 - resid.var() / log_c.var() if log_c.var() > 0 else float("nan")
        print(
            f"\nPower-law fit  circ(r) = {A_fit:.4e} * r^(-{n_fit:.3f})"
            f"  (dipole=2.000)  R²={r2:.4f}  over {fit_mask.sum()} shells"
        )

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    tag = f"R_sph={R_sph}  N={lat.N}"
    fig.suptitle(f"DST spheromak → dipole — {tag}", fontsize=11)

    # --- row 0, col 0: shell circulation linear ---
    ax = axes[0, 0]
    inn = r_shells <= R_sph
    ext = r_shells > R_sph
    ax.plot(
        r_shells[inn], circs[inn], "o-", color="steelblue", ms=3, label="interior (r≤R)"
    )
    ax.plot(
        r_shells[ext], circs[ext], "o-", color="tomato", ms=3, label="exterior (r>R)"
    )
    if n_fit is not None:
        r_line = r_shells[(r_shells >= r_min_fit) & (r_shells <= r_max_fit)]
        ax.plot(
            r_line,
            A_fit * r_line ** (-n_fit),
            "k--",
            lw=1.2,
            label=f"fit r^(-{n_fit:.2f})",
        )
    ax.axvline(R_sph, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("r (cells)")
    ax.set_ylabel("φ toroidal circulation")
    ax.set_title("Shell circulation (linear)")
    ax.legend(fontsize=8)

    # --- row 0, col 1: log-log ---
    ax = axes[0, 1]
    pos_m = circs > 0
    neg_m = circs < 0
    if pos_m.any():
        ax.loglog(
            r_shells[pos_m], circs[pos_m], "o", color="tomato", ms=3, label="circ > 0"
        )
    if neg_m.any():
        ax.loglog(
            r_shells[neg_m],
            -circs[neg_m],
            "s",
            color="steelblue",
            ms=3,
            label="-circ < 0",
        )
    if n_fit is not None:
        r_ref = np.array([r_min_fit, r_max_fit])
        ax.loglog(
            r_ref, A_fit * r_ref ** (-n_fit), "k--", lw=1.2, label=f"r^(-{n_fit:.2f})"
        )
        ax.loglog(
            r_ref,
            A_fit * r_ref ** (-2.0),
            "k:",
            lw=1.0,
            alpha=0.5,
            label="r^(-2) dipole ref",
        )
    ax.axvline(R_sph, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("r (cells)")
    ax.set_ylabel("|φ toroidal circulation|")
    ax.set_title("Log-log + dipole reference")
    ax.legend(fontsize=8)

    # --- row 0, col 2: radial u profile ---
    ax = axes[0, 2]
    rb_vals, um_vals, urs_vals = [], [], []
    for rb in range(1, int(r_all.max() * 0.80), 1):
        m = (r_all >= rb - 0.5) & (r_all < rb + 0.5)
        if m.sum():
            rb_vals.append(rb)
            um_vals.append(float(lat.u[m].mean()))
            urs_vals.append(float(np.sqrt(np.mean(lat.u[m] ** 2))))
    rb_vals = np.array(rb_vals)
    um_vals = np.array(um_vals)
    urs_vals = np.array(urs_vals)
    ax.plot(rb_vals, um_vals, "b-", lw=1.2, label="u mean")
    ax.fill_between(
        rb_vals,
        um_vals - urs_vals,
        um_vals + urs_vals,
        alpha=0.2,
        color="blue",
        label="±rms",
    )
    ax.axvline(R_sph, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("r (cells)")
    ax.set_ylabel("u")
    ax.set_title("Radial u profile (final)")
    ax.legend(fontsize=8)

    # --- row 1, col 0: H vs step ---
    ax = axes[1, 0]
    ax.plot(steps, H_hist, "k-", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("H = Σu² + ½Σφ²")
    ax.set_title("Total energy H vs step")

    # --- row 1, col 1: circ_in / circ_out vs step ---
    ax = axes[1, 1]
    ax.plot(steps, cin_hist, "-", color="steelblue", lw=1.2, label="circ_in  (Σ r≤R)")
    ax.plot(steps, cout_hist, "-", color="tomato", lw=1.2, label="circ_out (Σ r>R)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("Σ φ toroidal circulation")
    ax.set_title("Interior vs exterior circulation vs step")
    ax.legend(fontsize=8)

    # --- row 1, col 2: |circ_in| + |circ_out| log scale ---
    ax = axes[1, 2]
    abs_in = np.abs(cin_hist)
    abs_out = np.abs(cout_hist)
    if (abs_in > 0).any():
        ax.semilogy(steps, abs_in, "-", color="steelblue", lw=1.2, label="|circ_in|")
    if (abs_out > 0).any():
        ax.semilogy(steps, abs_out, "-", color="tomato", lw=1.2, label="|circ_out|")
    ax.set_xlabel("step")
    ax.set_ylabel("|Σ circ|")
    ax.set_title("Log circulation magnitude vs step")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"summary plot → {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

R_SCAN = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18)


def run(
    R_sph: float = 6.0,
    B0: float = 1.0,
    lam_peak: float = 5.0,
    lam_floor: float = 0.0,
    cs: float = 0.001,
    lam_scale: float = 0.05,
    radius_cells: float = 28.0,
    n_steps: int = 20000,
    report: int = 500,
    sponge: float = 0.15,
    damp_max: float = 0.003,
    n_threads: int = 6,
    plot_out: str = "spheromak_decay.png",
    plot_cadence: int = 0,
    plot_prefix: str = "snap_decay",
    no_plot: bool = False,
):
    print(f"\n=== DST spheromak → dipole  R={R_sph}  cs={cs}  lam_peak={lam_peak} ===")

    lat = DampedLattice(
        radius_cells=radius_cells, sponge_fraction=sponge, damp_max=damp_max
    )
    centre = np.zeros(3)
    r_all = np.linalg.norm(lat.pos, axis=1)
    in_sph = r_all <= R_sph

    print("Initialising phi (Taylor spheromak) …")
    n_faces = init_taylor_spheromak_phi(lat, centre=centre, R=R_sph, B0=B0)
    print(f"  faces_init={n_faces}  phi_max={lat.phi.max():.4f}")

    print("Initialising lam (Taylor u-profile) …")
    init_taylor_lam(lat, R=R_sph, lam_peak=lam_peak, lam_floor=lam_floor)

    dt = lat.stable_dt()
    print(f"  N={lat.N}  dt={dt:.6f}")

    # header
    r_cols = "  ".join(f"r={r:2d}" for r in R_SCAN)
    print(
        f"\n{'step':>6}  {'H':>10}  {'phi_rms':>9}  {'u_rms<R':>9}  "
        f"{'circ_in':>10}  {'circ_out':>10}  {r_cols}"
    )
    print("  " + "-" * (60 + 8 * len(R_SCAN)))

    history = []  # (step, H, phi_rms, u_rms, circ_in, circ_out)

    for step in range(n_steps + 1):
        if step % report == 0:
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            u_rms = float(np.sqrt(np.mean(lat.u[in_sph] ** 2)))
            shell_c = [_phi_circulation(lat, centre, r) for r in R_SCAN]
            circ_in = sum(c for r, c in zip(R_SCAN, shell_c) if r <= R_sph)
            circ_out = sum(c for r, c in zip(R_SCAN, shell_c) if r > R_sph)
            shells = "  ".join(f"{c:+.2e}" for c in shell_c)
            print(
                f"{step:6d}  {H:10.4e}  {phi_rms:9.4e}  {u_rms:9.4e}  "
                f"{circ_in:+10.3f}  {circ_out:+10.3f}  {shells}",
                flush=True,
            )
            history.append((step, H, phi_rms, u_rms, circ_in, circ_out))

        if not no_plot and plot_cadence > 0 and step % plot_cadence == 0:
            _plot_snapshot(lat, step, R_sph, f"{plot_prefix}_{step:07d}.png")

        if step < n_steps:
            lat.step(dt, n_threads)
            if cs > 0.0:
                lat.step_cross(dt, scale=cs, n_threads=n_threads)
            if lam_scale > 0.0:
                lat.step_lam(dt, scale=lam_scale, n_threads=n_threads)

    # final radial u profile (text)
    print("\n  Radial u profile (final):")
    for rb in range(1, int(radius_cells * 0.8), 2):
        m = (r_all >= rb - 0.5) & (r_all < rb + 0.5)
        if m.sum():
            um = float(lat.u[m].mean())
            urs = float(np.sqrt(np.mean(lat.u[m] ** 2)))
            tag = " <-- moat?" if um < -0.01 else ""
            print(f"    r={rb:3d}  u_mean={um:+.5f}  u_rms={urs:.5f}{tag}")

    if not no_plot:
        _plot_summary(history, lat, r_all, R_sph, centre, plot_out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=6.0, help="spheromak radius (cells)")
    p.add_argument("--B0", type=float, default=1.0, help="phi amplitude")
    p.add_argument("--lam-peak", type=float, default=5.0, help="peak lam inside sphere")
    p.add_argument(
        "--lam-floor", type=float, default=0.0, help="background lam (vacuum)"
    )
    p.add_argument("--cs", type=float, default=0.001, help="step_cross scale (0=off)")
    p.add_argument(
        "--lam-scale", type=float, default=0.05, help="step_lam scale (0=frozen)"
    )
    p.add_argument("--radius", type=float, default=28.0, help="lattice radius (cells)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--report", type=int, default=500)
    p.add_argument("--sponge", type=float, default=0.15)
    p.add_argument("--damp-max", type=float, default=0.003)
    p.add_argument("--n-threads", type=int, default=6)
    p.add_argument(
        "--plot-out",
        type=str,
        default="spheromak_decay.png",
        help="summary plot filename",
    )
    p.add_argument(
        "--plot-cadence", type=int, default=1000, help="snapshot every N steps (0=off)"
    )
    p.add_argument(
        "--plot-prefix",
        type=str,
        default="snap_decay",
        help="prefix for cadence snapshots",
    )
    p.add_argument("--no-plot", action="store_true", help="skip all plotting")
    a = p.parse_args()

    run(
        R_sph=a.R,
        B0=a.B0,
        lam_peak=a.lam_peak,
        lam_floor=a.lam_floor,
        cs=a.cs,
        lam_scale=a.lam_scale,
        radius_cells=a.radius,
        n_steps=a.steps,
        report=a.report,
        sponge=a.sponge,
        damp_max=a.damp_max,
        n_threads=a.n_threads,
        plot_out=a.plot_out,
        plot_cadence=a.plot_cadence,
        plot_prefix=a.plot_prefix,
        no_plot=a.no_plot,
    )
