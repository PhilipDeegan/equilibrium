"""
DST Harris sheet — minimal Python model testbed.

Two IC modes:

  u_bump (default) — double sech² u bump + lam slaved to u.
    Purpose: u/lam coupling at macro scale.

  phi_slab — two disc-shaped lam regions with opposing phi circulation.
    Each slab carries aggregate phi in the ±x direction (the DST-native
    analog of the Harris Bx field).  No imposed u — everything driven by
    the phi pattern.  step_cross allows topology change at the interface.
    This is the physically correct DST reconnection IC: the "magnetic field"
    emerges from two opposing proton aggregates, not from an imposed B(y).

Run:
  PYTHONPATH=$PWD python3 run/substrate/harris.py
  PYTHONPATH=$PWD python3 run/substrate/harris.py --ic phi_slab --B0 1.0 --cross-scale 0.01
  PYTHONPATH=$PWD python3 run/substrate/harris.py --ic phi_slab --B0 1.0 --lam-mode dynamic
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.math.geo.kelvin import KelvinLattice
from equilibrium.substrate.lattice import SubstrateLattice, SubstrateTile

# BCC face directions — lex order matches KelvinLattice.all_offsets
# Hex faces 0-7: (sx,sy,sz) ∈ {±1}³; sq faces 8-13: ±x, ±y, ±z axes
_HEX_DIR = np.array(
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
_SQ_DIR = np.array(
    [
        [-1, 0, 0],
        [+1, 0, 0],
        [0, -1, 0],
        [0, +1, 0],
        [0, 0, -1],
        [0, 0, +1],
    ],
    dtype=np.float64,
)
_FACE_DIR = np.vstack([_HEX_DIR, _SQ_DIR])  # (14, 3)


# ---------------------------------------------------------------------------
# BCC slab builder
# ---------------------------------------------------------------------------


def build_bcc_slab(Lx: float, Ly: float, Lz: float, d1: float = 1.0):
    """
    Construct a SubstrateLattice on a periodic BCC slab.

    Box dimensions are rounded to the nearest integer number of BCC conventional
    cells (a = d1*2/√3).  All three axes are periodic.

    Returns (lat, Lx_actual, Ly_actual, Lz_actual).
    """
    geo = KelvinLattice(lattice_spacing=d1)
    a = geo.a  # d1 * 2/sqrt(3)

    nx = max(1, round(Lx / a))
    ny = max(1, round(Ly / a))
    nz = max(1, round(Lz / a))
    Lx_a, Ly_a, Lz_a = nx * a, ny * a, nz * a
    box = np.array([Lx_a, Ly_a, Lz_a])

    ix = np.arange(nx, dtype=float)
    iy = np.arange(ny, dtype=float)
    iz = np.arange(nz, dtype=float)
    Ix, Iy, Iz = np.meshgrid(ix, iy, iz, indexing="ij")
    grid = np.stack([Ix.ravel(), Iy.ravel(), Iz.ravel()], axis=1)

    pts_corner = grid * a
    pts_bc = (grid + 0.5) * a
    pts = np.vstack([pts_corner, pts_bc])
    N = len(pts)

    lat = SubstrateLattice()
    lat.pos = pts
    lat.ell = np.ones(N, dtype=np.float64)
    lat.lam = np.zeros(N, dtype=np.float64)
    lat.u = np.zeros(N, dtype=np.float64)
    lat.phi = np.zeros((N, 14), dtype=np.float64)
    lat.fixed_u = np.zeros(N, dtype=bool)
    lat.fixed_phi = np.zeros((N, 14), dtype=bool)
    lat.tiles = [SubstrateTile(i, lat) for i in range(N)]

    # Neighbour wiring with periodic wrap
    scale = 2.0 / a

    def key(p):
        return tuple(np.round(np.asarray(p) * scale).astype(np.int64))

    idx_map = {key(p): i for i, p in enumerate(pts)}

    for i, tile in enumerate(lat.tiles):
        p = pts[i]
        for k, off in enumerate(geo.all_offsets):
            pn = (p + off) % box
            nb_key = key(pn)
            if nb_key in idx_map:
                tile.neighbours[k] = lat.tiles[idx_map[nb_key]]

    lat._dirty = True
    return lat, Lx_a, Ly_a, Lz_a


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="DST Harris sheet — Python model testbed")
parser.add_argument("--Lx", type=float, default=25.0, help="box x width (d1 units)")
parser.add_argument("--Ly", type=float, default=50.0, help="box y width (d1 units)")
parser.add_argument("--Lz", type=float, default=3.5, help="box z thickness (d1 units)")
parser.add_argument("--d1", type=float, default=1.0)
parser.add_argument(
    "--sheet-width", type=float, default=5.0, help="Harris half-width w (d1)"
)
parser.add_argument("--u-bg", type=float, default=0.4, help="background u")
parser.add_argument(
    "--u-peak", type=float, default=1.0, help="u peak at current sheets"
)
parser.add_argument("--g", type=float, default=1.0, help="lam-u coupling: lam_IC = g*u")
parser.add_argument(
    "--ic",
    type=str,
    default="u_bump",
    choices=["u_bump", "phi_slab"],
    help="IC mode: u_bump=sech² compression; phi_slab=opposing phi circulation",
)
parser.add_argument("--B0", type=float, default=1.0, help="phi_slab: phi amplitude")
parser.add_argument(
    "--lam-bg", type=float, default=1.0, help="phi_slab: uniform background lam"
)
parser.add_argument(
    "--lam-mode",
    type=str,
    default="slave",
    choices=["slave", "dynamic"],
    help="slave: re-impose lam=g*u each step; dynamic: evolve via step_lam",
)
parser.add_argument(
    "--lam-scale", type=float, default=0.01, help="step_lam scale (dynamic)"
)
parser.add_argument(
    "--cross-scale",
    type=float,
    default=0.0,
    help="step_cross scale (0=off); topology change",
)
parser.add_argument("--n-steps", type=int, default=5000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument(
    "--perturb",
    type=float,
    default=0.0,
    help="perturbation amplitude as fraction of u_peak (0 = none)",
)
parser.add_argument(
    "--perturb-mode",
    type=str,
    default="u_island",
    choices=["u_island", "phi"],
    help="u_island: modulate u sech² with cos(kx*x); phi: divergence-free phi seed",
)
parser.add_argument("--dt", type=float, default=None, help="override CFL timestep")
parser.add_argument(
    "--plot-out", type=str, default="harris_test.png", help="output plot filename"
)
parser.add_argument("--no-plot", action="store_true", help="skip plot")


def main():
    args = parser.parse_args()

    lat, Lx_a, Ly_a, Lz_a = build_bcc_slab(args.Lx, args.Ly, args.Lz, args.d1)
    N = lat.N
    pos = lat.pos
    y = pos[:, 1]
    y0 = Ly_a / 4.0
    w = args.sheet_width

    print(f"Box: {Lx_a:.2f} x {Ly_a:.2f} x {Lz_a:.2f}  N={N}  y0={y0:.2f}  w={w}")

    # -----------------------------------------------------------------------
    # IC
    # -----------------------------------------------------------------------
    y1 = Ly_a - y0  # second sheet at 3Ly/4 (y coords are 0 → Ly_a)

    def dy_periodic(y_coord, y_null):
        """Minimum periodic distance, wraps to [-Ly/2, Ly/2]."""
        d = y_coord - y_null
        return d - Ly_a * np.round(d / Ly_a)

    # Precompute face geometry (needed for phi_slab and snapshot)
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
    pos_j = lat.pos[nb_safe]  # (N,14,3)
    face_diff = pos_j - lat.pos[:, np.newaxis, :]  # (N,14,3)
    face_len = np.linalg.norm(face_diff, axis=-1, keepdims=True)
    face_nhat = face_diff / np.where(face_len > 0, face_len, 1.0)  # (N,14,3)
    face_mid_y = 0.5 * (lat.pos[:, 1, np.newaxis] + pos_j[:, :, 1])  # (N,14)

    if args.ic == "u_bump":
        u_init = (
            args.u_bg
            + args.u_peak / np.cosh(dy_periodic(y, y0) / w) ** 2
            + args.u_peak / np.cosh(dy_periodic(y, y1) / w) ** 2
        )
        lat.u = u_init.copy()
        lat.lam = args.g * lat.u.copy()

        # Perturbation
        if args.perturb > 0.0:
            kx = 2.0 * np.pi / Lx_a
            x_c = pos[:, 0]
            island_x = np.cos(kx * (x_c + Lx_a / 2.0))

            if args.perturb_mode == "u_island":
                for y_null in (y0, y1):
                    yr = dy_periodic(pos[:, 1], y_null)
                    lat.u += (
                        args.perturb * args.u_peak / np.cosh(yr / w) ** 2 * island_x
                    )
                lat.lam = args.g * lat.u.copy()
                print(
                    f"u_island perturbation: amp={args.perturb*args.u_peak:.4f}"
                    f"  u=[{lat.u.min():.4f}, {lat.u.max():.4f}]"
                )
            else:
                dP = args.perturb * args.u_peak
                for sign, y_null in ((+1, y0), (-1, y1)):
                    yr = dy_periodic(pos[:, 1], y_null)
                    env = np.exp(-(yr**2) / w**2)
                    px = (
                        sign
                        * (-2.0 * yr / w**2)
                        * dP
                        * np.cos(kx * (x_c + Lx_a / 2.0))
                        * env
                    )
                    py = sign * kx * dP * np.sin(kx * (x_c + Lx_a / 2.0)) * env
                    for k in range(14):
                        lat.phi[:, k] += px * _FACE_DIR[k, 0] + py * _FACE_DIR[k, 1]
                print(
                    f"phi perturbation: dP={dP:.4f}"
                    f"  phi_rms={float(np.sqrt(np.mean(lat.phi**2))):.4e}"
                )

        print(f"IC: u=[{u_init.min():.4f}, {u_init.max():.4f}]")
        print(
            f"    sigma_bg={1/(1+args.g*args.u_bg):.4f}"
            f"  sigma_sheet={1/(1+args.g*(args.u_bg+args.u_peak)):.4f}"
        )

    else:  # phi_slab
        # Two opposing phi circulation regions — DST-native Harris sheet.
        # phi[i,k] = B0 * tanh(dy_periodic(y_mid, y0)/w) * n_hat_x
        # This gives +B0*n_hat_x between the sheets (y0 < y < y1)
        #        and -B0*n_hat_x outside (y < y0 or y > y1).
        # Current sheets are the tanh transitions at y=y0 and y=y1.
        dy_face = face_mid_y - y0
        dy_face = dy_face - Ly_a * np.round(dy_face / Ly_a)
        tanh_prof = np.tanh(dy_face / w)  # (N,14): +1 between, -1 outside

        lat.phi = np.where(lat._valid, args.B0 * tanh_prof * face_nhat[:, :, 0], 0.0)
        lat.u[:] = 0.0
        lat.lam[:] = args.lam_bg

        if args.perturb > 0.0:
            kx = 2.0 * np.pi / Lx_a
            x_c = pos[:, 0]
            # seed a u_island at each sheet to break x-symmetry
            for y_null in (y0, y1):
                yr = dy_periodic(pos[:, 1], y_null)
                lat.u += (
                    args.perturb
                    * args.B0
                    / np.cosh(yr / w) ** 2
                    * np.cos(kx * (x_c + Lx_a / 2.0))
                )
            print(f"phi_slab perturb: u=[{lat.u.min():.4f}, {lat.u.max():.4f}]")

        phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
        print(
            f"phi_slab IC: B0={args.B0}  lam_bg={args.lam_bg}"
            f"  phi_rms={phi_rms:.4e}"
        )

    dt = args.dt if args.dt is not None else lat.stable_dt()
    print(
        f"dt={dt:.4e}  lam_mode={args.lam_mode}  cross_scale={args.cross_scale}"
        f"  lam_scale={args.lam_scale}"
    )

    # Diagnostic masks
    sheet_mask = (np.abs(y - y0) < 2 * args.d1) | (np.abs(y - y1) < 2 * args.d1)
    mid_mask = np.abs(y - Ly_a / 2.0) < 2 * args.d1

    u_vol_0 = float(lat.u.sum())
    n_sheet = sheet_mask.sum()
    n_mid = mid_mask.sum()
    print(f"sheet cells: {n_sheet}  mid cells: {n_mid}")

    # Snapshot steps for 2D maps: t=0, t=25%, t=final
    snap_steps = sorted(set([0, args.n_steps // 4, args.n_steps]))
    snapshots = {}  # step → (Xu, Yu, Ug, Sg)

    def _midlayer_mask():
        z = pos[:, 2]
        zmid = (z.max() + z.min()) / 2.0
        ztol = (z.max() - z.min()) / 6.0
        return np.abs(z - zmid) < ztol

    _zmask = _midlayer_mask()

    def _to_grid(xv, yv, v, nr=120):
        from scipy.interpolate import griddata

        xi = np.linspace(xv.min(), xv.max(), nr)
        yi = np.linspace(yv.min(), yv.max(), nr)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((xv, yv), v, (Xi, Yi), method="linear")
        return Xi, Yi, Zi

    def _phi_x_cell():
        """Per-cell mean phi projected onto x — DST analog of local Bx."""
        valid = lat._valid
        proj = (lat.phi * face_nhat[:, :, 0] * valid).sum(axis=1)
        count = valid.sum(axis=1).clip(1)
        return proj / count

    def take_snapshot():
        sig = 1.0 / (1.0 + lat.lam)
        phi_x = _phi_x_cell()
        xv, yv = pos[_zmask, 0], pos[_zmask, 1]
        Xu, Yu, Ug = _to_grid(xv, yv, lat.u[_zmask])
        _, _, Sg = _to_grid(xv, yv, sig[_zmask])
        _, _, Pg = _to_grid(xv, yv, phi_x[_zmask])
        return Xu, Yu, Ug, Sg, Pg

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    ts_steps, ts_u_sheet, ts_u_mid, ts_ratio, ts_phi_E = [], [], [], [], []

    hdr = (
        f"{'step':>7}  {'u_sheet':>9}  {'u_std':>7}  {'u_mid':>9}  "
        f"{'lam_sheet':>10}  {'sig_sheet':>10}  {'lam/u':>8}  "
        f"{'phi_E':>10}  {'Δu_vol':>12}"
    )
    print(f"\n{hdr}")
    print("  " + "-" * len(hdr))

    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            u_s = float(lat.u[sheet_mask].mean())
            u_std = float(lat.u[sheet_mask].std())
            u_m = float(lat.u[mid_mask].mean())
            lam_s = float(lat.lam[sheet_mask].mean())
            sig_s = float((1.0 / (1.0 + lat.lam))[sheet_mask].mean())
            ratio = lam_s / u_s if u_s > 1e-10 else float("nan")
            phi_E = float(np.sum(lat.phi**2))
            u_vol = float(lat.u.sum())
            print(
                f"{step:7d}  {u_s:9.5f}  {u_std:7.4f}  {u_m:9.5f}  "
                f"{lam_s:10.5f}  {sig_s:10.5f}  {ratio:8.4f}  "
                f"{phi_E:10.4e}  {u_vol - u_vol_0:+12.4e}"
            )
            ts_steps.append(step)
            ts_u_sheet.append(u_s)
            ts_u_mid.append(u_m)
            ts_ratio.append(ratio)
            ts_phi_E.append(phi_E)

        if step in snap_steps:
            snapshots[step] = take_snapshot()

        if step < args.n_steps:
            lat.step(dt)
            if args.cross_scale > 0.0:
                lat.step_cross(dt, scale=args.cross_scale)
            if args.lam_mode == "slave":
                lat.lam = args.g * lat.u
            else:
                lat.step_lam(dt, scale=args.lam_scale)

    print("done")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        n_rows = 3 if args.ic == "phi_slab" else 2
        n_snaps = len([s for s in snap_steps if s in snapshots])
        fig = plt.figure(figsize=(15, 4 * n_rows))
        fig.suptitle(
            f"DST Harris sheet — ic={args.ic}  lam_mode={args.lam_mode}"
            + (f"  cross={args.cross_scale}" if args.cross_scale > 0 else ""),
            fontsize=11,
        )
        gs = gridspec.GridSpec(
            n_rows,
            1 + n_snaps,
            figure=fig,
            width_ratios=[1.2] + [1] * n_snaps,
            hspace=0.38,
            wspace=0.35,
        )

        # Col 0: time series
        ax_ts = fig.add_subplot(gs[:, 0])
        t_arr = np.array(ts_steps) * dt
        ax_ts.plot(t_arr, ts_u_sheet, label="u_sheet", color="C0")
        ax_ts.plot(t_arr, ts_u_mid, label="u_mid", color="C1", linestyle="--")
        ax_ts.set_xlabel("time")
        ax_ts.set_ylabel("u", color="C0")
        ax_ts.tick_params(axis="y", labelcolor="C0")
        ax_ts.legend(loc="upper right", fontsize=8)
        ax_r = ax_ts.twinx()
        ax_r.plot(t_arr, ts_ratio, label="lam/u at sheet", color="C3", linestyle=":")
        ax_r.set_ylabel("lam / u", color="C3")
        ax_r.tick_params(axis="y", labelcolor="C3")
        ax_r.legend(loc="lower right", fontsize=8)
        ax_ts.set_title("Time series")

        # Cols 1-N: 2D maps at each snapshot
        u_all = np.concatenate(
            [snapshots[s][2].ravel() for s in snap_steps if s in snapshots]
        )
        s_all = np.concatenate(
            [snapshots[s][3].ravel() for s in snap_steps if s in snapshots]
        )
        p_all = np.concatenate(
            [snapshots[s][4].ravel() for s in snap_steps if s in snapshots]
        )
        u_vmin, u_vmax = float(np.nanpercentile(u_all, 2)), float(
            np.nanpercentile(u_all, 98)
        )
        s_vmin, s_vmax = float(np.nanpercentile(s_all, 2)), float(
            np.nanpercentile(s_all, 98)
        )
        p_vm = float(np.nanpercentile(np.abs(p_all), 98)) or 1e-6

        for ci, snap in enumerate(s for s in snap_steps if s in snapshots):
            Xu, Yu, Ug, Sg, Pg = snapshots[snap]
            lbl = f"t={snap * dt:.1f}"

            ax_u = fig.add_subplot(gs[0, ci + 1])
            im_u = ax_u.pcolormesh(
                Xu,
                Yu,
                Ug,
                cmap="inferno",
                vmin=u_vmin,
                vmax=u_vmax,
                shading="auto",
                rasterized=True,
            )
            fig.colorbar(im_u, ax=ax_u, shrink=0.85, label="u")
            ax_u.axhline(y0, color="w", lw=0.8, ls="--", alpha=0.7)
            ax_u.axhline(Ly_a - y0, color="w", lw=0.8, ls="--", alpha=0.7)
            ax_u.set_title(f"u  —  {lbl}")
            ax_u.set_xlabel("x")
            ax_u.set_ylabel("y")

            ax_s = fig.add_subplot(gs[1, ci + 1])
            im_s = ax_s.pcolormesh(
                Xu,
                Yu,
                Sg,
                cmap="viridis",
                vmin=s_vmin,
                vmax=s_vmax,
                shading="auto",
                rasterized=True,
            )
            fig.colorbar(im_s, ax=ax_s, shrink=0.85, label="σ = 1/(1+λ)")
            ax_s.axhline(y0, color="w", lw=0.8, ls="--", alpha=0.7)
            ax_s.axhline(Ly_a - y0, color="w", lw=0.8, ls="--", alpha=0.7)
            ax_s.set_title(f"σ  —  {lbl}")
            ax_s.set_xlabel("x")
            ax_s.set_ylabel("y")

            if args.ic == "phi_slab":
                ax_p = fig.add_subplot(gs[2, ci + 1])
                im_p = ax_p.pcolormesh(
                    Xu,
                    Yu,
                    Pg,
                    cmap="RdBu_r",
                    vmin=-p_vm,
                    vmax=p_vm,
                    shading="auto",
                    rasterized=True,
                )
                fig.colorbar(im_p, ax=ax_p, shrink=0.85, label="φ_x (Bx analog)")
                ax_p.axhline(y0, color="k", lw=0.8, ls="--", alpha=0.7)
                ax_p.axhline(Ly_a - y0, color="k", lw=0.8, ls="--", alpha=0.7)
                ax_p.set_title(f"φ_x  —  {lbl}")
                ax_p.set_xlabel("x")
                ax_p.set_ylabel("y")

        plt.savefig(args.plot_out, dpi=150, bbox_inches="tight")
        print(f"plot saved → {args.plot_out}")


if __name__ == "__main__":
    main()
