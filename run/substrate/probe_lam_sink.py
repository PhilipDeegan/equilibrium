"""
probe_lam_sink.py  —  φ/λ coupling probe, two free knots on the z-axis.

Phase 1 (warmup): DampedLattice wave step; let u/phi settle.
Phase 2 (coupled): step + optional lam sub-steps; track diagnostics.

Phi modes (--phi-mode):
  wave — pure substrate wave
  chi  — wave + chi-gradient drive in step()
  sink — wave + lam-sink absorption/re-emission   [default]
  all  — all three
  none — phi frozen at 0; isolates lam-force from wave coupling
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice
from tst.substrate import axis_sel, knot_mask, lam_com, mpl_agg, sph_bin, u_slab

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=25.0)
parser.add_argument("--sep", type=float, default=6.0)
parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--u-central", type=float, default=1.0)
parser.add_argument("--n-warm", type=int, default=5000)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--sink-scale", type=float, default=1e-3)
parser.add_argument("--lam-scale", type=float, default=0.0)
parser.add_argument("--force-scale", type=float, default=1e-3)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--report", type=int, default=2000)
parser.add_argument(
    "--phi-mode",
    choices=["wave", "sink", "all", "none"],
    default="sink",
    help=(
        "wave: pure wave; "
        "sink: wave+lam-sink [default]; "
        "all: wave+sink; "
        "none: phi=0 always (isolates lam-force from wave)"
    ),
)
parser.add_argument(
    "--plot-cadence",
    type=int,
    default=0,
    help="save snapshot plots every N coupled steps (0=off)",
)
parser.add_argument("--plot-dir", type=str, default="probe_plots")
parser.add_argument(
    "--plot-prefix",
    type=str,
    default="",
    help="filename prefix for plots, e.g. 'sink_' or 'wave_'",
)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def save_snapshot(lat, step, plot_dir, half, prefix=""):
    """2×2 figure: z-axis profiles (u/λ, φ_z) and spherical profiles (u/λ, φ anisotropy)."""
    plt = mpl_agg()
    pos = lat.pos
    r_all = np.linalg.norm(pos, axis=1)
    r_max = float(r_all.max())
    r_bins = np.linspace(0, r_max, 40)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    near, order = axis_sel(pos)
    z_s = pos[near, 2][order]
    u_s = lat.u[near][order]
    lam_s = lat.lam[near][order]
    phi_s = lat.phi[near][order]

    # face 12 = −z sq,  face 13 = +z sq
    phi_z_net = phi_s[:, 13] - phi_s[:, 12]
    phi_z_mag = np.sqrt(phi_s[:, 12] ** 2 + phi_s[:, 13] ** 2)

    phi_rms_cell = np.sqrt((lat.phi**2).mean(axis=1))
    phi_rms_hex = np.sqrt((lat.phi[:, :8] ** 2).mean(axis=1))
    phi_rms_sq = np.sqrt((lat.phi[:, 8:] ** 2).mean(axis=1))

    u_r = sph_bin(lat.u, r_all, r_bins)
    lam_r = sph_bin(lat.lam, r_all, r_bins)
    p_all = sph_bin(phi_rms_cell, r_all, r_bins)
    p_hex = sph_bin(phi_rms_hex, r_all, r_bins)
    p_sq = sph_bin(phi_rms_sq, r_all, r_bins)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"step {step}", fontsize=12)

    # top-left: u and λ along z-axis
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(z_s, u_s, "b-", lw=1.2, label="u")
    ax2.plot(z_s, lam_s, "r-", lw=1.2, label="λ")
    for v in (half, -half):
        ax.axvline(v, color="k", lw=0.5, ls="--")
    ax.set_xlabel("z")
    ax.set_ylabel("u", color="b")
    ax2.set_ylabel("λ", color="r")
    ax.set_title("z-axis: u and λ")
    ax.legend(
        ax.get_lines() + ax2.get_lines(), ["u", "λ"], loc="upper right", fontsize=8
    )

    # top-right: φ_z along z-axis
    ax = axes[0, 1]
    ax.plot(z_s, phi_z_net, "g-", lw=1.2, label="φ_z net (13−12)")
    ax.plot(z_s, phi_z_mag, "g--", lw=0.8, alpha=0.6, label="|φ_z|")
    ax.axhline(0, color="k", lw=0.4)
    for v in (half, -half):
        ax.axvline(v, color="k", lw=0.5, ls="--")
    ax.set_xlabel("z")
    ax.set_ylabel("φ_z")
    ax.set_title("z-axis: φ_z (sq faces ±z)")
    ax.legend(fontsize=8)

    # bottom-left: spherical mean u and λ vs r
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(r_mid, u_r, "b-", lw=1.2, label="<u>")
    ax2.plot(r_mid, lam_r, "r-", lw=1.2, label="<λ>")
    ax.set_xlabel("r")
    ax.set_ylabel("<u>", color="b")
    ax2.set_ylabel("<λ>", color="r")
    ax.set_title("spherical mean: u and λ vs r")
    ax.legend(ax.get_lines() + ax2.get_lines(), ["<u>", "<λ>"], fontsize=8)

    # bottom-right: φ anisotropy hex vs sq
    ax = axes[1, 1]
    ax.plot(r_mid, p_all, "k-", lw=1.2, label="all faces")
    ax.plot(r_mid, p_hex, "b--", lw=0.9, label="hex (0–7)")
    ax.plot(r_mid, p_sq, "r--", lw=0.9, label="sq  (8–13)")
    ax.set_xlabel("r")
    ax.set_ylabel("φ rms")
    ax.set_title("φ anisotropy: hex vs sq faces")
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f"{prefix}snap_{step:07d}.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  [plot] {fname}", flush=True)


def save_timeseries(history, plot_dir, prefix=""):
    if not history:
        return
    plt = mpl_agg()
    steps = np.array([h["step"] for h in history])
    lam_sum = np.array([h["lam_sum"] for h in history])
    u_sum = np.array([h["u_sum"] for h in history])
    cycle_q = np.array([h["cycle_q"] for h in history])
    u_btw = np.array([h["u_btw"] for h in history])
    u_out = np.array([h["u_out"] for h in history])
    sep = np.array([h["sep"] for h in history])
    phi_rms = np.array([h["phi_rms"] for h in history])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("probe_lam_sink — time series", fontsize=12)

    axes[0, 0].plot(steps, lam_sum)
    axes[0, 0].set_title("λ_sum (total winding)")
    axes[0, 0].set_ylabel("λ_sum")

    axes[0, 1].plot(steps, phi_rms)
    axes[0, 1].set_title("φ rms (global)")
    axes[0, 1].set_ylabel("φ rms")

    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(steps, u_sum, "b-", lw=1.2, label="u_sum")
    ax2.plot(steps, lam_sum, "r--", lw=0.9, label="λ_sum")
    ax.set_title("global u_sum and λ_sum")
    ax.set_ylabel("u_sum", color="b")
    ax2.set_ylabel("λ_sum", color="r")
    ax.legend(ax.get_lines() + ax2.get_lines(), ["u_sum", "λ_sum"], fontsize=8)

    axes[1, 1].plot(steps, cycle_q - cycle_q[0])
    axes[1, 1].axhline(0, color="k", lw=0.5)
    axes[1, 1].set_title("cycle drift Δ(u_sum + λ_sum)")
    axes[1, 1].set_ylabel("drift")

    axes[2, 0].plot(steps, u_btw, label="u_between")
    axes[2, 0].plot(steps, u_out, label="u_outer")
    axes[2, 0].set_title("u-field zones")
    axes[2, 0].set_ylabel("u")
    axes[2, 0].legend()

    axes[2, 1].plot(steps, sep)
    axes[2, 1].set_title("knot COM separation")
    axes[2, 1].set_ylabel("separation")

    for ax in axes.flat:
        ax.set_xlabel("step")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f"{prefix}timeseries.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  [plot] {fname}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()
    phi_mode = args.phi_mode

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    dt = lat.stable_dt()

    half = args.sep / 2.0
    chi_0 = args.lam / (1.0 + args.lam)

    def nearest(target):
        return lat.tiles[int(np.argmin(np.linalg.norm(pos - target, axis=1)))]

    kA = nearest(np.array([0.0, 0.0, half]))
    kB = nearest(np.array([0.0, 0.0, -half]))

    # Superpose two independent winding profiles (element-wise max to preserve both cores).
    lat.set_winding(kA, chi_0=chi_0)
    lam_A = lat.lam.copy()
    lat.lam[:] = 0.0
    lat.set_winding(kB, chi_0=chi_0)
    lat.lam[:] = np.maximum(lam_A, lat.lam)

    # u: superposition of two 1/r wells — no fixed_u, let everything evolve freely.
    for t in lat.tiles:
        rA = float(np.linalg.norm(t.pos - kA.pos))
        rB = float(np.linalg.norm(t.pos - kB.pos))
        lat.u[t.idx] = args.u_central / max(rA, 0.5) + args.u_central / max(rB, 0.5)

    z_btw_lo, z_btw_hi = -half, +half
    z_outA_lo, z_outA_hi = +half, +half + args.sep
    z_outB_lo, z_outB_hi = -half - args.sep, -half

    print(
        f"N={lat.N}  R={args.R}  sep={args.sep}  lam_0={args.lam:.4f}  "
        f"dt={dt:.4e}  phi_mode={phi_mode}"
    )
    print(
        f"  sink_scale={args.sink_scale}  force_scale={args.force_scale}"
        f"  lam_scale={args.lam_scale}"
    )
    print(f"knot A: {kA.pos}   B: {kB.pos}")
    print(f"lam_sum_init = {lat.lam.sum():.4f}")

    # -----------------------------------------------------------------------
    # Phase 1 — warmup (always wave step so u/phi settle)
    # -----------------------------------------------------------------------
    print(f"\nwarmup {args.n_warm} steps …", flush=True)
    for s in range(1, args.n_warm + 1):
        lat.step(dt, n_threads=args.n_threads)
        if s % max(1, args.n_warm // 4) == 0:
            print(
                f"  {s}/{args.n_warm}  u_mean={lat.u.mean():.4f}"
                f"  phi_rms={np.sqrt(np.mean(lat.phi**2)):.4e}",
                flush=True,
            )
    print("warmup done")

    if phi_mode == "none":
        lat.phi[:] = 0.0
        print("phi_mode=none: zeroed phi after warmup")

    # -----------------------------------------------------------------------
    # Post-warmup reference
    # -----------------------------------------------------------------------
    lam_sum_0 = float(lat.lam.sum())
    u_sum_0 = float(lat.u.sum())
    cycle_q_0 = u_sum_0 + lam_sum_0  # tracked to expose u drain vs lam growth
    phi_rms_0 = float(np.sqrt(np.mean(lat.phi**2)))
    comA_0 = lam_com(pos, lat.lam * knot_mask(pos, kA.pos))
    comB_0 = lam_com(pos, lat.lam * knot_mask(pos, kB.pos))
    sep_0 = float(np.linalg.norm(comA_0 - comB_0))

    print(
        f"\npost-warmup: lam_sum={lam_sum_0:.4f}  u_sum={u_sum_0:.4f}"
        f"  cycle_q={cycle_q_0:.4f}  phi_rms={phi_rms_0:.4e}  sep={sep_0:.4f}\n"
    )

    if args.plot_cadence:
        save_snapshot(lat, 0, args.plot_dir, half, prefix=args.plot_prefix)

    hdr = (
        f"{'step':>8}  {'lam_sum':>10}  {'Δlam':>9}  "
        f"{'u_sum':>12}  {'Δcycle':>10}  "
        f"{'u_between':>10}  {'u_outer':>8}  {'u_asym':>8}  "
        f"{'sep':>8}  {'Δsep':>7}  {'phi_rms':>9}"
    )
    print(hdr)
    print("  " + "-" * len(hdr))

    history = []

    # -----------------------------------------------------------------------
    # Phase 2 — coupled step() + lam sub-steps
    # -----------------------------------------------------------------------
    for s in range(args.n_steps + 1):

        if s % args.report == 0:
            mA = knot_mask(pos, kA.pos)
            mB = knot_mask(pos, kB.pos)
            comA = lam_com(pos, lat.lam * mA)
            comB = lam_com(pos, lat.lam * mB)
            sep = float(np.linalg.norm(comA - comB))

            lam_sum = float(lat.lam.sum())
            u_sum = float(lat.u.sum())
            cycle_q = u_sum + lam_sum
            u_btw = u_slab(pos, lat.u, z_btw_lo, z_btw_hi)
            u_outA = u_slab(pos, lat.u, z_outA_lo, z_outA_hi)
            u_outB = u_slab(pos, lat.u, z_outB_lo, z_outB_hi)
            u_out = (u_outA + u_outB) / 2.0
            u_asym = u_btw - u_out
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))

            print(
                f"{s:8d}  {lam_sum:10.4f}  {lam_sum-lam_sum_0:+9.4f}  "
                f"{u_sum:12.2f}  {cycle_q-cycle_q_0:+10.4f}  "
                f"{u_btw:10.6f}  {u_out:8.6f}  {u_asym:+8.6f}  "
                f"{sep:8.4f}  {sep-sep_0:+7.4f}  {phi_rms:9.4e}",
                flush=True,
            )

            history.append(
                dict(
                    step=s,
                    lam_sum=lam_sum,
                    u_sum=u_sum,
                    cycle_q=cycle_q,
                    u_btw=u_btw,
                    u_out=u_out,
                    u_asym=u_asym,
                    sep=sep,
                    phi_rms=phi_rms,
                )
            )

        if args.plot_cadence and s > 0 and s % args.plot_cadence == 0:
            save_snapshot(lat, s, args.plot_dir, half, prefix=args.plot_prefix)

        if s == args.n_steps:
            break

        # --- phi step ---
        if phi_mode == "none":
            lat.phi[:] = 0.0
        else:
            lat.step(dt, n_threads=args.n_threads)
            if phi_mode in ("sink", "all"):
                lat.step_lam_sink(dt, scale=args.sink_scale)

        # --- lam sub-steps ---
        if args.lam_scale:
            lat.step_lam(dt, scale=args.lam_scale)
        if args.force_scale:
            lat.step_lam_force(dt, scale=args.force_scale)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    lam_sum_f = float(lat.lam.sum())
    sep_f = float(
        np.linalg.norm(
            lam_com(pos, lat.lam * knot_mask(pos, kA.pos))
            - lam_com(pos, lat.lam * knot_mask(pos, kB.pos))
        )
    )
    print(
        f"\nlam growth:      {lam_sum_0:.4f} → {lam_sum_f:.4f}  (Δ={lam_sum_f-lam_sum_0:+.4f})"
    )
    print(f"knot separation: {sep_0:.4f} → {sep_f:.4f}  (Δ={sep_f-sep_0:+.4f})")

    save_timeseries(history, args.plot_dir, prefix=args.plot_prefix)
    print("done")


if __name__ == "__main__":
    main()
