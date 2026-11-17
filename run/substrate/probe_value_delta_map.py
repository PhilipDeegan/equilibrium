"""
probe_value_delta_map.py  —  IC calibration via coordinate-descent relaxation.

For each target variable (u / phi / lam):
  1. Start from the base IC.
  2. Reset all fields to IC; override the target field with the current estimate.
  3. Run n_short substrate steps.
  4. Extract the new estimate of the target field (global; convergence is checked in
     the focus region r ≤ center_extent).
  5. Repeat until rms change in the focus region falls below tol, or max_iter reached.

The "delta map" shows, for each cell in the focus region, how much each field moved
(converged − IC) and how those movements correlate across field pairs.

Typical invocation:
    PYTHONPATH=$PWD python3 run/substrate/probe_value_delta_map.py \\
        --R 20 --lam 1.0 --u-central 1.0 \\
        --vars u lam --n-short 10 --max-iter 300 --tol 1e-7 \\
        --plot-dir tmp/delta_map
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument(
    "--lam",
    type=float,
    default=1.0,
    help="base IC winding (lam_0 at knot centre BFS r=1)",
)
parser.add_argument(
    "--u-central",
    type=float,
    default=1.0,
    help="Coulomb amplitude: u(r) = u_central / r",
)
parser.add_argument(
    "--n-short", type=int, default=10, help="substrate steps per relaxation iteration"
)
parser.add_argument(
    "--max-iter",
    type=int,
    default=300,
    help="max coordinate-descent iterations per variable",
)
parser.add_argument(
    "--tol", type=float, default=1e-7, help="convergence: rms delta in focus region"
)
parser.add_argument(
    "--center-extent",
    type=float,
    default=5.0,
    help="focus region radius in cells (convergence check + delta-map display)",
)
parser.add_argument(
    "--vars",
    nargs="+",
    choices=["u", "phi", "lam"],
    default=["u"],
    help="variables to relax independently (default: u only)",
)
parser.add_argument(
    "--phi-mode",
    choices=["wave", "sink"],
    default="wave",
    help="step type: wave=step_cpp only; sink=step_cpp + step_lam_sink",
)
parser.add_argument("--sink-scale", type=float, default=1e-3)
parser.add_argument(
    "--lam-scale",
    type=float,
    default=0.0,
    help="if >0 also run step_lam each iteration step",
)
parser.add_argument(
    "--pin-u",
    action="store_true",
    help="freeze u at IC during relaxation steps so phi finds its driven steady state",
)
parser.add_argument("--bc", choices=["sponge", "open"], default="sponge")
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--sponge-fraction", type=float, default=0.15)
parser.add_argument(
    "--kill-bc",
    action="store_true",
    help="hard-zero phi beyond kill-fraction*R every step — perfect absorber, no reflection",
)
parser.add_argument(
    "--kill-fraction",
    type=float,
    default=0.8,
    help="radius fraction beyond which phi is zeroed when --kill-bc is set (default 0.8)",
)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--plot-dir", type=str, default="probe_plots")
parser.add_argument("--plot-prefix", type=str, default="delta_map_")
parser.add_argument(
    "--save",
    type=str,
    default="",
    help="path prefix for .npz snapshot of final converged lattice (e.g. tmp/lattice/phi_ic)",
)
parser.add_argument(
    "--load",
    type=str,
    default="",
    help="path to a .npz snapshot to use as the starting estimate for --vars fields (resumes a previous run)",
)

# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------


def make_lattice(args):
    if args.bc == "sponge":
        return DampedLattice(
            radius_cells=args.R,
            sponge_fraction=args.sponge_fraction,
            damp_max=args.damp_max,
        )
    return SubstrateLattice(radius_cells=args.R)


def make_kill_mask(lat, args):
    """Precompute the hard-kill phi mask (r > kill_fraction * R)."""
    if not args.kill_bc:
        return None
    r_all = np.linalg.norm(lat.pos, axis=1)
    return r_all > args.kill_fraction * args.R


def build_ic(lat, u_central, lam_0):
    """Set base IC: Coulomb u-profile + set_winding lam + phi=0."""
    pos = lat.pos
    r_all = np.linalg.norm(pos, axis=1)
    chi_0 = lam_0 / (1.0 + lam_0)
    kA = lat.tiles[int(np.argmin(r_all))]
    lat.lam[:] = 0.0
    lat.set_winding(kA, chi_0=chi_0)
    for t in lat.tiles:
        r = float(np.linalg.norm(t.pos))
        lat.u[t.idx] = u_central / max(r, 0.5)
    lat.phi[:] = 0.0


def run_step(lat, dt, args, kill_mask=None):
    """One composite step (wave + optional lam sub-steps + optional hard BC kill)."""
    if isinstance(lat, DampedLattice):
        lat.step(dt, n_threads=args.n_threads)
    else:
        lat.step_cpp(dt, args.n_threads)
    if args.phi_mode == "sink":
        lat.step_lam_sink(dt, scale=args.sink_scale)
    if args.lam_scale:
        lat.step_lam(dt, scale=args.lam_scale)
    if kill_mask is not None:
        lat.phi[kill_mask] = 0.0


# ---------------------------------------------------------------------------
# Core relaxation
# ---------------------------------------------------------------------------


def relax_variable(
    lat, var_name, ic_u, ic_phi, ic_lam, dt, focus_mask, args, kill_mask=None, init=None
):
    """
    Coordinate-descent relaxation for one field variable.

    init: optional dict with keys 'u'/'phi'/'lam' to override the starting
          estimate for var_name (used when resuming from a saved snapshot).

    Returns (converged_field, delta_history) where delta_history[k] is the
    rms change of the target field in the focus region at iteration k.
    """
    current = {
        "u": ic_u.copy(),
        "phi": ic_phi.copy(),
        "lam": ic_lam.copy(),
    }
    if init and var_name in init:
        current[var_name] = init[var_name].copy()
    deltas = []
    n_focus = int(focus_mask.sum())

    for it in range(args.max_iter):
        # Reset all fields to IC, then override the target
        lat.u[...] = ic_u
        lat.phi[...] = ic_phi
        lat.lam[...] = ic_lam
        if var_name == "u":
            lat.u[...] = current["u"]
        elif var_name == "phi":
            lat.phi[...] = current["phi"]
        else:
            lat.lam[...] = current["lam"]

        if args.pin_u:
            lat.fixed_u[:] = True
        for _ in range(args.n_short):
            run_step(lat, dt, args, kill_mask)
        if args.pin_u:
            lat.fixed_u[:] = False

        # Extract new estimate and compute delta in focus region
        if var_name == "u":
            new_val = lat.u.copy()
            diff = new_val[focus_mask] - current["u"][focus_mask]
        elif var_name == "phi":
            new_val = lat.phi.copy()
            diff = (new_val[focus_mask] - current["phi"][focus_mask]).ravel()
        else:
            new_val = lat.lam.copy()
            diff = new_val[focus_mask] - current["lam"][focus_mask]

        delta = float(np.sqrt(np.mean(diff**2)))
        deltas.append(delta)
        current[var_name] = new_val

        if var_name == "phi":
            val_rms = float(np.sqrt(np.mean(new_val[focus_mask] ** 2)))
        else:
            val_rms = float(np.mean(np.abs(new_val[focus_mask])))

        print(
            f"  iter {it+1:4d}  delta_{var_name}={delta:.4e}"
            f"  |{var_name}|_focus={val_rms:.4e}  n_focus={n_focus}",
            flush=True,
        )

        if it >= 1 and delta < args.tol:
            print(f"  -> converged at iter {it+1}")
            break

    return current[var_name], deltas


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _z_axis_sel(pos, r_cut=1.5):
    """Cells near the z-axis, sorted by z coordinate."""
    near = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2) < r_cut
    order = np.argsort(pos[near, 2])
    return near, order


def _field_z(field, near, order, var_name):
    """Extract z-axis slice; reduce phi to rms across faces."""
    vals = field[near][order]
    if var_name == "phi":
        return np.sqrt((vals**2).mean(axis=1))
    return vals


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_convergence(delta_histories, plot_dir, prefix):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4))
    for var_name, deltas in delta_histories.items():
        ax.semilogy(
            range(1, len(deltas) + 1), deltas, marker=".", markersize=3, label=var_name
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("rms delta in focus region")
    ax.set_title("IC relaxation convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f"{prefix}convergence.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  [plot] {fname}", flush=True)


def plot_profiles(ic_map, converged, pos, center_extent, plot_dir, prefix):
    """Z-axis profiles: IC vs converged, and delta in [-extent, +extent]."""
    plt = _mpl()
    near, order = _z_axis_sel(pos)
    z_s = pos[near, 2][order]
    in_win = np.abs(z_s) <= center_extent

    n = len(converged)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    fig.suptitle(f"z-axis profiles  (focus +/-{center_extent} cells)", fontsize=11)

    for col, (var_name, conv_field) in enumerate(converged.items()):
        ic_vals = _field_z(ic_map[var_name], near, order, var_name)
        conv_vals = _field_z(conv_field, near, order, var_name)

        ax = axes[0, col]
        ax.plot(z_s, ic_vals, "b--", lw=1.0, label="IC")
        ax.plot(z_s, conv_vals, "r-", lw=1.2, label="converged")
        ax.axvline(-center_extent, color="k", lw=0.4, ls=":")
        ax.axvline(+center_extent, color="k", lw=0.4, ls=":")
        ax.set_xlabel("z")
        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name}: full z-axis")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        delta = conv_vals - ic_vals
        ax.plot(z_s[in_win], delta[in_win], "g-", lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("z")
        ax.set_ylabel(f"delta_{var_name}")
        ax.set_title(f"delta_{var_name}  [-{center_extent}, +{center_extent}]")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f"{prefix}profiles.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  [plot] {fname}", flush=True)


def plot_radial_profiles(ic_map, converged, pos, center_extent, plot_dir, prefix):
    """Spherical-mean radial profiles within the focus region."""
    plt = _mpl()
    r_all = np.linalg.norm(pos, axis=1)
    focus = r_all <= center_extent
    r_f = r_all[focus]
    r_bins = np.linspace(0, center_extent, 20)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    n = len(converged)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(f"radial profiles in focus region (r <= {center_extent})", fontsize=11)

    for col, (var_name, conv_field) in enumerate(converged.items()):
        ic_f = ic_map[var_name]
        if var_name == "phi":
            ic_vals = np.sqrt((ic_f[focus] ** 2).mean(axis=1))
            conv_vals = np.sqrt((conv_field[focus] ** 2).mean(axis=1))
        else:
            ic_vals = ic_f[focus]
            conv_vals = conv_field[focus]

        def _bin(vals, rf=r_f, rb=r_bins, rm=r_mid):
            out = np.full(len(rm), np.nan)
            for i in range(len(rm)):
                m = (rf >= rb[i]) & (rf < rb[i + 1])
                if m.any():
                    out[i] = vals[m].mean()
            return out

        ax = axes[0, col]
        ax.plot(r_mid, _bin(ic_vals), "b--", lw=1.0, label="IC")
        ax.plot(r_mid, _bin(conv_vals), "r-", lw=1.2, label="converged")
        ax.plot(r_mid, _bin(conv_vals - ic_vals), "g-", lw=0.9, label="delta")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("r")
        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name} radial mean")
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f"{prefix}radial.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  [plot] {fname}", flush=True)


def plot_delta_map(ic_map, converged, pos, focus_mask, plot_dir, prefix):
    """
    Delta map: for cells in focus region, plot delta_A vs delta_B coloured by r.
    Diagonal panels show the distribution of each delta.
    Skipped if fewer than two variables were probed.
    """
    if len(converged) < 2:
        return

    plt = _mpl()

    deltas = {}
    for var_name, conv_field in converged.items():
        ic_f = ic_map[var_name]
        if var_name == "phi":
            d = np.sqrt(((conv_field[focus_mask] - ic_f[focus_mask]) ** 2).mean(axis=1))
        else:
            d = conv_field[focus_mask] - ic_f[focus_mask]
        deltas[var_name] = d

    r_focus = np.linalg.norm(pos[focus_mask], axis=1)
    var_names = list(deltas.keys())
    n = len(var_names)

    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n), squeeze=False)
    fig.suptitle("Delta map: delta(converged - IC) in focus region", fontsize=11)

    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            ax = axes[i, j]
            if i == j:
                ax.hist(deltas[vi], bins=30, color="steelblue", alpha=0.7)
                ax.set_xlabel(f"delta_{vi}")
                ax.set_title(f"delta_{vi} distribution")
            else:
                sc = ax.scatter(
                    deltas[vj], deltas[vi], c=r_focus, cmap="plasma", s=10, alpha=0.7
                )
                plt.colorbar(sc, ax=ax, label="r")
                ax.set_xlabel(f"delta_{vj}")
                ax.set_ylabel(f"delta_{vi}")
                ax.set_title(f"delta_{vi} vs delta_{vj}")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f"{prefix}delta_map.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  [plot] {fname}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    lat = make_lattice(args)
    pos = lat.pos
    dt = lat.stable_dt()
    r_all = np.linalg.norm(pos, axis=1)
    focus_mask = r_all <= args.center_extent

    build_ic(lat, args.u_central, args.lam)
    ic_u = lat.u.copy()
    ic_phi = lat.phi.copy()
    ic_lam = lat.lam.copy()

    print(
        f"N={lat.N}  R={args.R}  dt={dt:.4e}"
        f"  focus_cells={int(focus_mask.sum())} (r<={args.center_extent})"
    )
    print(
        f"  u_central={args.u_central}  lam_0={args.lam}"
        f"  chi_0={args.lam/(1+args.lam):.4f}"
    )
    print(
        f"  vars={args.vars}  n_short={args.n_short}"
        f"  max_iter={args.max_iter}  tol={args.tol}"
    )
    kill_mask = make_kill_mask(lat, args)
    print(
        f"  phi_mode={args.phi_mode}  pin_u={args.pin_u}  bc={args.bc}  n_threads={args.n_threads}"
        + (f"  kill_bc=r>{args.kill_fraction*args.R:.1f}" if args.kill_bc else "")
    )
    print()

    init = None
    if args.load:
        data = np.load(args.load if args.load.endswith(".npz") else args.load + ".npz")
        init = {k: data[k] for k in ("u", "phi", "lam") if k in data}
        print(
            f"  loaded initial estimate from {args.load}  (fields: {list(init.keys())})"
        )

    ic_map = {"u": ic_u, "phi": ic_phi, "lam": ic_lam}
    delta_histories = {}
    converged = {}

    for var_name in args.vars:
        print(f"-- relaxing {var_name} " + "-" * 50)
        conv_field, deltas = relax_variable(
            lat, var_name, ic_u, ic_phi, ic_lam, dt, focus_mask, args, kill_mask, init
        )
        converged[var_name] = conv_field
        delta_histories[var_name] = deltas
        print()

    print("-- summary " + "-" * 60)
    for var_name, conv_field in converged.items():
        ic_f = ic_map[var_name]
        if var_name == "phi":
            diff_rms = float(
                np.sqrt(np.mean((conv_field[focus_mask] - ic_f[focus_mask]) ** 2))
            )
            diff_max = float(np.max(np.abs(conv_field[focus_mask] - ic_f[focus_mask])))
        else:
            diff = conv_field[focus_mask] - ic_f[focus_mask]
            diff_rms = float(np.sqrt(np.mean(diff**2)))
            diff_max = float(np.max(np.abs(diff)))
        n_iters = len(delta_histories[var_name])
        flag = delta_histories[var_name][-1] < args.tol
        print(
            f"  {var_name}: iters={n_iters}  converged={flag}"
            f"  delta_rms={diff_rms:.4e}  delta_max={diff_max:.4e}"
        )

    print()
    plot_convergence(delta_histories, args.plot_dir, args.plot_prefix)
    plot_profiles(
        ic_map, converged, pos, args.center_extent, args.plot_dir, args.plot_prefix
    )
    plot_radial_profiles(
        ic_map, converged, pos, args.center_extent, args.plot_dir, args.plot_prefix
    )
    plot_delta_map(ic_map, converged, pos, focus_mask, args.plot_dir, args.plot_prefix)

    if args.save:
        # Restore converged fields into lat so save() captures the final state
        for var_name, conv_field in converged.items():
            if var_name == "u":
                lat.u[...] = conv_field
            elif var_name == "phi":
                lat.phi[...] = conv_field
            else:
                lat.lam[...] = conv_field
        # Fields not probed stay at IC values
        if "u" not in converged:
            lat.u[...] = ic_u
        if "phi" not in converged:
            lat.phi[...] = ic_phi
        if "lam" not in converged:
            lat.lam[...] = ic_lam
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        path = args.save + ".npz"
        np.savez_compressed(
            path,
            u=lat.u,
            phi=lat.phi,
            lam=lat.lam,
            pos=lat.pos,
            R=np.float64(args.R),
            dt=np.float64(dt),
        )
        print(f"converged lattice saved -> {path}")
        ic_path = args.save + "_ic.npz"
        np.savez_compressed(
            ic_path,
            u=ic_u,
            phi=ic_phi,
            lam=ic_lam,
            pos=lat.pos,
            R=np.float64(args.R),
            dt=np.float64(dt),
        )
        print(f"IC lattice saved        -> {ic_path}")

    print("done")


if __name__ == "__main__":
    main()
