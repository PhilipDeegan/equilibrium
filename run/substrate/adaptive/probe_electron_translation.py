"""
DST electron translation — inertia as omnibalance + cost of translation.

The electron (double-ring dipole, poloidal Taylor component, no lam) is seeded
in its omnibalance state: uniform u + W-weighted divergence-free phi — an
exact fixed point of the reversible step.  A static u tilt (pressure gradient)
is then applied across the domain and pinned at a battery ring just inside the
sponge, and the winding's centre is tracked.

Questions, per config (cross_scale × kick axis):
  1. Does external pressure couple to the winding at all?  With lam = 0 the
     wave step and (per sign pattern) step_cross are linear, so superposition
     predicts NO coupling — the tilt adds an independent background solution
     and the winding stays put.  Any measured drift comes from the
     sign-gate nonlinearity of step_cross (it fires on inward phi only) or
     is diagnostic contamination.
  2. Is coupling gated by the interior crossing?  cross=0 vs cross>0.
  3. Isotropy (omnibalance check): kick along the dipole axis (z) vs
     equatorial (x) — a truly omnibalanced structure responds identically.

A null result everywhere is itself the headline: the current operator set
has no channel by which pressure translates a lam-free winding — the
translation gate (step_cross v2) is missing physics, not a tuning problem.

The phase-tilt (momentum) kick used for the 8-tile ring probes does not
apply here: the dipole is a zero-frequency structure with no oscillation
phase to tilt.  Constructing a translating dipole state is part of the
step_cross v2 design question.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_electron_translation.py
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_electron_translation.py \
      --R 30 --cross-scales 0.0,0.01,0.05 --kick-axes x,z --gradient 1e-3
"""

import argparse
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_electron_dipole_ic,
    phi_poloidal_scan,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument(
    "--sph-frac",
    type=float,
    default=0.2,
    help="dipole radius fraction (default 0.2 → scale 4.0 at R=20, matching R=80 runs)",
)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--sigma", type=float, default=None)
parser.add_argument(
    "--cross-scales",
    type=str,
    default="0.0,0.01,0.02",
    help="comma list of step_cross scales to sweep",
)
parser.add_argument(
    "--kick-axes",
    type=str,
    default="x,z",
    help="comma list of u-gradient axes (z = dipole axis, x = equatorial)",
)
parser.add_argument(
    "--gradient", type=float, default=5e-4, help="u tilt per cell along kick axis"
)
parser.add_argument(
    "--no-pin",
    action="store_true",
    help="do not pin the tilt at the battery ring inside the sponge",
)
parser.add_argument("--n-base", type=int, default=500, help="pre-kick baseline steps")
parser.add_argument("--n-kick", type=int, default=3000, help="steps with tilt applied")
parser.add_argument(
    "--n-coast", type=int, default=2000, help="steps after tilt removed"
)
parser.add_argument("--report", type=int, default=250)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument(
    "--com-thresh",
    type=float,
    default=0.15,
    help="winding-cell mask: prms > thresh * prms_max",
)
parser.add_argument("--plot-out", type=str, default="probe_electron_translation.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--data-out", type=str, default=None)

S_SCAN_FRAC = (0.5, 1.0, 1.5)  # tube shells for winding survival, in units of scale

_AX = {"x": 0, "y": 1, "z": 2}


def winding_com(cl, thresh_frac):
    """Centre of the winding: prms-weighted COM over cells above threshold."""
    prms = np.sqrt((np.array(cl.leaf_phi) ** 2).mean(axis=0))
    pmax = float(prms.max())
    if pmax < 1e-30:
        return np.zeros(3), 0.0
    mask = prms > thresh_frac * pmax
    w = prms[mask]
    com = (np.array(cl.leaf_pos)[mask] * w[:, None]).sum(axis=0) / w.sum()
    return com, pmax


def run_config(args, cross, kick_axis):
    ax_i = _AX[kick_axis]
    scale = args.R * args.sph_frac
    s_scan = tuple(f * scale for f in S_SCAN_FRAC)

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    pos = np.array(cl.leaf_pos)
    r_all = np.linalg.norm(pos, axis=1)
    r_inner = float(r_all.max()) * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    _, r_axis = init_electron_dipole_ic(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=args.B0,
        lam_target=0.0,
        u_bg=0.0,
        amplitude=args.amplitude,
        sigma=args.sigma,
    )
    dt = compute_dt(cl)

    tilt = args.gradient * pos[:, ax_i]
    battery = (r_all > r_inner - 2.0) & (r_all <= r_inner)

    history = []  # (t, phase, com3, circ_pol_max, prms_max)

    def record(step, phase):
        com, pmax = winding_com(cl, args.com_thresh)
        mid, n_hat, phi_f = face_flux_geometry(cl, com)
        pol = phi_poloidal_scan(mid, n_hat, phi_f, r_axis, s_scan)
        history.append((step * dt, phase, com, float(np.abs(pol).max()), pmax))
        c = history[-1]
        print(
            f"  {step:6d} {phase:>5}  com=({c[2][0]:+8.4f},{c[2][1]:+8.4f},"
            f"{c[2][2]:+8.4f})  circ_pol={c[3]:+.3e}  prms_max={c[4]:.3e}",
            flush=True,
        )

    def advance(n, phase, step0):
        for s in range(1, n + 1):
            alat.step(0, dt)
            alat.step_sponge(0)
            if cross > 0.0:
                alat.step_cross(0, dt, cross)
            if s % args.report == 0:
                record(step0 + s, phase)
        return step0 + n

    print(f"\n=== config: cross={cross}  kick_axis={kick_axis}  g={args.gradient} ===")
    record(0, "base")
    step = advance(args.n_base, "base", 0)

    # apply tilt (+ pin at the battery ring so the sponge cannot erode it)
    cl.set_leaf_u(np.array(cl.leaf_u) + tilt)
    if not args.no_pin:
        cl.set_leaf_fixed_u(battery.astype(np.uint8))
    step = advance(args.n_kick, "kick", step)

    # remove tilt, unpin, coast
    if not args.no_pin:
        cl.set_leaf_fixed_u(np.zeros(cl.n_leaves, dtype=np.uint8))
    cl.set_leaf_u(np.array(cl.leaf_u) - tilt)
    step = advance(args.n_coast, "coast", step)

    # velocity fits along the kick axis (second half of each phase)
    def fit_v(phase):
        rows = [(t, c[ax_i]) for t, p, c, _, _ in history if p == phase]
        if len(rows) < 3:
            return 0.0
        rows = rows[len(rows) // 2 :]
        t = np.array([r[0] for r in rows])
        x = np.array([r[1] for r in rows])
        return float(np.polyfit(t, x, 1)[0])

    com0 = history[0][2]
    com_end_kick = next(c for t, p, c, _, _ in reversed(history) if p == "kick")
    res = {
        "cross": cross,
        "axis": kick_axis,
        "v_base": fit_v("base"),
        "v_kick": fit_v("kick"),
        "v_coast": fit_v("coast"),
        "d_kick": float(com_end_kick[ax_i] - com0[ax_i]),
        "pol_frac": history[-1][3] / max(history[0][3], 1e-30),
        "history": history,
        "ax_i": ax_i,
        "dt": dt,
    }
    print(
        f"  → v_base={res['v_base']:+.3e}  v_kick={res['v_kick']:+.3e}"
        f"  v_coast={res['v_coast']:+.3e}  Δ_kick={res['d_kick']:+.4f}"
        f"  mobility v/g={res['v_kick'] / args.gradient:+.3e}"
        f"  pol_survival={res['pol_frac']:.3f}"
    )
    return res


def main():
    args = parser.parse_args()
    cross_scales = [float(v) for v in args.cross_scales.split(",")]
    axes = [a.strip() for a in args.kick_axes.split(",")]

    results = [
        run_config(args, cross, ax)
        for cross, ax in itertools.product(cross_scales, axes)
    ]

    print(
        f"\n{'cross':>7}  {'axis':>4}  {'v_base':>10}  {'v_kick':>10}  "
        f"{'v_coast':>10}  {'mobility':>10}  {'pol_surv':>8}"
    )
    for r in results:
        print(
            f"{r['cross']:7.3f}  {r['axis']:>4}  {r['v_base']:+10.3e}"
            f"  {r['v_kick']:+10.3e}  {r['v_coast']:+10.3e}"
            f"  {r['v_kick'] / args.gradient:+10.3e}  {r['pol_frac']:8.3f}"
        )

    if args.data_out:
        np.savez(
            args.data_out,
            summary=np.array(
                [
                    (
                        r["cross"],
                        _AX[r["axis"]],
                        r["v_base"],
                        r["v_kick"],
                        r["v_coast"],
                        r["d_kick"],
                        r["pol_frac"],
                    )
                    for r in results
                ]
            ),
            gradient=args.gradient,
        )
        print(f"data saved → {args.data_out}")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Electron translation probe  R={args.R}  g={args.gradient}"
        f"  (kick t=[{args.n_base}dt, {args.n_base + args.n_kick}dt])",
        fontsize=10,
    )
    cmap = plt.cm.tab10
    for i, r in enumerate(results):
        t = np.array([h[0] for h in r["history"]])
        d = np.array([h[2][r["ax_i"]] for h in r["history"]])
        pol = np.array([h[3] for h in r["history"]])
        lbl = f"cross={r['cross']}, {r['axis']}"
        ax1.plot(t, d - d[0], color=cmap(i % 10), lw=1.5, label=lbl)
        ax2.semilogy(t, np.abs(pol), color=cmap(i % 10), lw=1.2, label=lbl)
    dt0 = results[0]["dt"]
    for ax in (ax1, ax2):
        ax.axvline(args.n_base * dt0, color="k", lw=0.5, ls="--")
        ax.axvline((args.n_base + args.n_kick) * dt0, color="k", lw=0.5, ls="--")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("winding COM displacement (kick axis)")
    ax1.set_title("Translation response")
    ax1.legend(fontsize=7)
    ax2.set_ylabel("|circ_pol| max (log)")
    ax2.set_xlabel("time")
    ax2.set_title("Winding survival")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved → {args.plot_out}")


if __name__ == "__main__":
    main()
