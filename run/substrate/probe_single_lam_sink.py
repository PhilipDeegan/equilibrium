"""
probe_single_lam_sink.py  —  φ/λ coupling probe, single knot at origin.

BC options (--bc):
  sponge    — DampedLattice: absorbing quadratic sponge layer [default]
  radiation — RadiationBCLattice: preserves 1/r Coulomb tail, absorbs AC waves
  open      — SubstrateLattice: no BC (hard reflective boundary)

Goal: compare how the system evolves under each BC; converge toward something
semiconservative where the Coulomb u-field is preserved but wave energy escapes.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import (
    DampedLattice,
    RadiationBCLattice,
    SubstrateLattice,
)
from tst.substrate import axis_sel, mpl_agg, sph_bin

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=25.0)
parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--u-central", type=float, default=1.0)
parser.add_argument("--n-warm", type=int, default=25000)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument(
    "--bc",
    choices=["sponge", "radiation", "open"],
    default="sponge",
    help="boundary condition (default: sponge)",
)
parser.add_argument(
    "--damp-max",
    type=float,
    default=0.003,
    help="sponge/radiation BC peak damping coefficient",
)
parser.add_argument(
    "--sponge-fraction",
    type=float,
    default=0.15,
    help="fraction of radius used as BC layer",
)
parser.add_argument(
    "--phi-mode",
    choices=["wave", "chi", "sink", "all", "none"],
    default="sink",
    help="wave: pure wave; chi: +chi-drive; sink: +lam-sink [default]; all: all; none: phi=0",
)
parser.add_argument("--sink-scale", type=float, default=1e-3)
parser.add_argument("--lam-scale", type=float, default=0.0)
parser.add_argument(
    "--force-scale",
    type=float,
    default=0.0,
    help="lam-force (∇u pull) scale — default 0 to avoid blob collapse",
)
parser.add_argument("--chi-scale", type=float, default=0.0)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--report", type=int, default=2000)
parser.add_argument(
    "--plot-cadence",
    type=int,
    default=0,
    help="save snapshot plots every N coupled steps (0=off)",
)
parser.add_argument("--plot-dir", type=str, default="probe_plots")
parser.add_argument("--plot-prefix", type=str, default="")
parser.add_argument(
    "--pin-warmup",
    action="store_true",
    help="fix u at the knot core during warmup so the Coulomb field builds up; "
    "releases before the coupled phase (required for radiation BC to differ from sponge)",
)
parser.add_argument(
    "--pin-release-steps",
    type=int,
    default=0,
    help="ramp core u linearly from pinned value to 0 over N steps before releasing (0=instant)",
)
parser.add_argument(
    "--n-free-warm",
    type=int,
    default=0,
    help="free wave steps after pin release to let phi shock damp; BC is NOT re-frozen (keeps original Coulomb fit)",
)
parser.add_argument(
    "--sink-amp",
    type=float,
    default=0.0,
    help="chi amplification in step_lam_sink: 1.0=original (1+chi) feedback, 0.0=flat re-emission (default)",
)
parser.add_argument(
    "--save",
    type=str,
    default="",
    help="path prefix for .npz snapshot saved at end of run (e.g. tmp/lattice/rad)",
)


# ---------------------------------------------------------------------------
# BC factory
# ---------------------------------------------------------------------------


def make_lattice(args):
    if args.bc == "sponge":
        return DampedLattice(
            radius_cells=args.R,
            sponge_fraction=args.sponge_fraction,
            damp_max=args.damp_max,
        )
    elif args.bc == "radiation":
        return RadiationBCLattice(
            radius_cells=args.R,
            bc_fraction=args.sponge_fraction,
            damp_max=args.damp_max,
        )
    else:  # open
        return SubstrateLattice(radius_cells=args.R)


def wave_step(lat, dt, n_threads, chi_scale):
    """Dispatch to the correct step method for each BC type."""
    if isinstance(lat, (DampedLattice, RadiationBCLattice)):
        lat.step(dt, n_threads=n_threads, chi_scale=chi_scale)
    elif chi_scale:
        lat.step(dt, chi_scale=chi_scale)
    else:
        lat.step_cpp(dt, n_threads)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_snapshot(lat, step, plot_dir, prefix, R):
    plt = mpl_agg()
    pos = lat.pos
    r_all = np.linalg.norm(pos, axis=1)
    r_bins = np.linspace(0, R, 40)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    near, order = axis_sel(pos)
    z_s = pos[near, 2][order]
    u_s = lat.u[near][order]
    lam_s = lat.lam[near][order]
    phi_s = lat.phi[near][order]
    phi_z_net = phi_s[:, 13] - phi_s[:, 12]
    phi_z_mag = np.sqrt(phi_s[:, 12] ** 2 + phi_s[:, 13] ** 2)

    u_r = sph_bin(lat.u, r_all, r_bins)
    lam_r = sph_bin(lat.lam, r_all, r_bins)
    p_all = sph_bin(np.sqrt((lat.phi**2).mean(axis=1)), r_all, r_bins)
    p_hex = sph_bin(np.sqrt((lat.phi[:, :8] ** 2).mean(axis=1)), r_all, r_bins)
    p_sq = sph_bin(np.sqrt((lat.phi[:, 8:] ** 2).mean(axis=1)), r_all, r_bins)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"step {step}", fontsize=12)

    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(z_s, u_s, "b-", lw=1.2, label="u")
    ax2.plot(z_s, lam_s, "r-", lw=1.2, label="λ")
    ax.set_xlabel("z")
    ax.set_ylabel("u", color="b")
    ax2.set_ylabel("λ", color="r")
    ax.set_title("z-axis: u and λ")
    ax.legend(
        ax.get_lines() + ax2.get_lines(), ["u", "λ"], loc="upper right", fontsize=8
    )

    ax = axes[0, 1]
    ax.plot(z_s, phi_z_net, "g-", lw=1.2, label="φ_z net")
    ax.plot(z_s, phi_z_mag, "g--", lw=0.8, alpha=0.6, label="|φ_z|")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("z")
    ax.set_ylabel("φ_z")
    ax.set_title("z-axis: φ_z (sq faces ±z)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(r_mid, u_r, "b-", lw=1.2, label="<u>")
    ax2.plot(r_mid, lam_r, "r-", lw=1.2, label="<λ>")
    ax.set_xlabel("r")
    ax.set_ylabel("<u>", color="b")
    ax2.set_ylabel("<λ>", color="r")
    ax.set_title("spherical mean: u and λ vs r")
    ax.legend(ax.get_lines() + ax2.get_lines(), ["<u>", "<λ>"], fontsize=8)

    ax = axes[1, 1]
    ax.plot(r_mid, p_all, "k-", lw=1.2, label="all")
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


def save_timeseries(history, plot_dir, prefix, R):
    if not history:
        return
    plt = mpl_agg()
    steps = np.array([h["step"] for h in history])
    lam_sum = np.array([h["lam_sum"] for h in history])
    u_sum = np.array([h["u_sum"] for h in history])
    cycle_q = np.array([h["cycle_q"] for h in history])
    u_core = np.array([h["u_core"] for h in history])
    u_mid = np.array([h["u_mid"] for h in history])
    u_far = np.array([h["u_far"] for h in history])
    phi_rms = np.array([h["phi_rms"] for h in history])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("probe_single_lam_sink — time series", fontsize=12)

    axes[0, 0].plot(steps, lam_sum)
    axes[0, 0].set_title("λ_sum")
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

    axes[2, 0].plot(steps, u_core, label="core (r<2)")
    axes[2, 0].plot(steps, u_mid, label="mid (R/3–2R/3)")
    axes[2, 0].plot(steps, u_far, label="far (0.75–0.9 R)")
    axes[2, 0].set_title("u radial shells")
    axes[2, 0].set_ylabel("u")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].axis("off")

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
    R = args.R

    lat = make_lattice(args)
    pos = lat.pos
    dt = lat.stable_dt()
    r_all = np.linalg.norm(pos, axis=1)

    chi_0 = args.lam / (1.0 + args.lam)

    def nearest(target):
        return lat.tiles[int(np.argmin(np.linalg.norm(pos - target, axis=1)))]

    kA = nearest(np.zeros(3))
    lat.set_winding(kA, chi_0=chi_0)

    for t in lat.tiles:
        r = float(np.linalg.norm(t.pos))
        lat.u[t.idx] = args.u_central / max(r, 0.5)

    # radial diagnostic shells
    m_core = r_all < 2.0
    m_mid = (r_all >= R / 3) & (r_all < 2 * R / 3)
    m_far = (r_all >= 0.75 * R) & (r_all < 0.90 * R)

    if args.pin_warmup:
        lat.fixed_u[m_core] = True

    print(
        f"N={lat.N}  R={R}  lam_0={args.lam:.4f}  dt={dt:.4e}"
        f"  bc={args.bc}  phi_mode={phi_mode}"
    )
    print(
        f"  sink_scale={args.sink_scale}  sink_amp={args.sink_amp}  force_scale={args.force_scale}"
        f"  chi_scale={args.chi_scale}  lam_scale={args.lam_scale}"
    )
    print(
        f"  damp_max={args.damp_max}  sponge_fraction={args.sponge_fraction}"
        f"  pin_warmup={args.pin_warmup}"
    )
    print(f"knot: {kA.pos}   lam_sum_init={lat.lam.sum():.4f}")

    # -----------------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------------
    print(f"\nwarmup {args.n_warm} steps …", flush=True)
    for s in range(1, args.n_warm + 1):
        wave_step(lat, dt, args.n_threads, chi_scale=0.0)
        if s % max(1, args.n_warm // 4) == 0:
            print(
                f"  {s}/{args.n_warm}  u_mean={lat.u.mean():.4f}"
                f"  phi_rms={np.sqrt(np.mean(lat.phi**2)):.4e}",
                flush=True,
            )

    if args.bc == "radiation":
        lat.freeze_bc()
        print(f"radiation BC frozen: A={lat._A:.6f}  u_bg={lat._u_bg:.6f}")

    if args.pin_warmup:
        if args.pin_release_steps > 0:
            u_core_pinned = lat.u[m_core].copy()
            print(
                f"gradual pin release over {args.pin_release_steps} steps …", flush=True
            )
            for k in range(1, args.pin_release_steps + 1):
                alpha = 1.0 - k / args.pin_release_steps
                lat.u[m_core] = u_core_pinned * alpha
                wave_step(lat, dt, args.n_threads, chi_scale=0.0)
            print("gradual release done")
        lat.fixed_u[m_core] = False
        print("pin released")

    if args.n_free_warm > 0:
        print(f"\nfree warmup {args.n_free_warm} steps …", flush=True)
        for s in range(1, args.n_free_warm + 1):
            wave_step(lat, dt, args.n_threads, chi_scale=0.0)
            if s % max(1, args.n_free_warm // 4) == 0:
                print(
                    f"  {s}/{args.n_free_warm}  u_mean={lat.u.mean():.4f}"
                    f"  phi_rms={np.sqrt(np.mean(lat.phi**2)):.4e}",
                    flush=True,
                )
        print("free warmup done")

    if phi_mode == "none":
        lat.phi[:] = 0.0
        print("phi_mode=none: zeroed phi after warmup")

    print("warmup done")

    # -----------------------------------------------------------------------
    # Post-warmup reference
    # -----------------------------------------------------------------------
    lam_sum_0 = float(lat.lam.sum())
    u_sum_0 = float(lat.u.sum())
    cycle_q_0 = u_sum_0 + lam_sum_0
    phi_rms_0 = float(np.sqrt(np.mean(lat.phi**2)))

    print(
        f"\npost-warmup: lam_sum={lam_sum_0:.4f}  u_sum={u_sum_0:.4f}"
        f"  cycle_q={cycle_q_0:.4f}  phi_rms={phi_rms_0:.4e}\n"
    )

    if args.save:
        ic_path = args.save + "_ic.npz"
        np.savez_compressed(
            ic_path,
            u=lat.u,
            phi=lat.phi,
            lam=lat.lam,
            pos=lat.pos,
            R=np.float64(R),
            dt=np.float64(dt),
        )
        print(f"IC saved → {ic_path}")

    if args.plot_cadence:
        save_snapshot(lat, 0, args.plot_dir, args.plot_prefix, R)

    hdr = (
        f"{'step':>8}  {'lam_sum':>10}  {'Δlam':>9}  "
        f"{'u_sum':>12}  {'Δcycle':>10}  "
        f"{'u_core':>8}  {'u_mid':>8}  {'u_far':>8}  {'phi_rms':>9}"
    )
    print(hdr)
    print("  " + "-" * len(hdr))

    history = []

    # -----------------------------------------------------------------------
    # Coupled phase
    # -----------------------------------------------------------------------
    for s in range(args.n_steps + 1):
        if s % args.report == 0:
            lam_sum = float(lat.lam.sum())
            u_sum = float(lat.u.sum())
            cycle_q = u_sum + lam_sum
            u_core = float(lat.u[m_core].mean()) if m_core.any() else 0.0
            u_mid = float(lat.u[m_mid].mean()) if m_mid.any() else 0.0
            u_far = float(lat.u[m_far].mean()) if m_far.any() else 0.0
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))

            print(
                f"{s:8d}  {lam_sum:10.4f}  {lam_sum-lam_sum_0:+9.4f}  "
                f"{u_sum:12.2f}  {cycle_q-cycle_q_0:+10.4f}  "
                f"{u_core:8.4f}  {u_mid:8.4f}  {u_far:8.4f}  {phi_rms:9.4e}",
                flush=True,
            )

            history.append(
                dict(
                    step=s,
                    lam_sum=lam_sum,
                    u_sum=u_sum,
                    cycle_q=cycle_q,
                    u_core=u_core,
                    u_mid=u_mid,
                    u_far=u_far,
                    phi_rms=phi_rms,
                )
            )

        if args.plot_cadence and s > 0 and s % args.plot_cadence == 0:
            save_snapshot(lat, s, args.plot_dir, args.plot_prefix, R)

        if s == args.n_steps:
            break

        # --- phi step ---
        if phi_mode == "none":
            lat.phi[:] = 0.0
        else:
            chi_s = args.chi_scale if phi_mode in ("chi", "all") else 0.0
            wave_step(lat, dt, args.n_threads, chi_scale=chi_s)
            if phi_mode in ("sink", "all"):
                lat.step_lam_sink(dt, scale=args.sink_scale, amp=args.sink_amp)

        # --- lam sub-steps ---
        if args.lam_scale:
            lat.step_lam(dt, scale=args.lam_scale)
        if args.force_scale:
            lat.step_lam_force(dt, scale=args.force_scale)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    lam_sum_f = float(lat.lam.sum())
    u_sum_f = float(lat.u.sum())
    print(
        f"\nlam growth: {lam_sum_0:.4f} → {lam_sum_f:.4f}  (Δ={lam_sum_f-lam_sum_0:+.4f})"
    )
    print(f"u_sum:      {u_sum_0:.4f} → {u_sum_f:.4f}  (Δ={u_sum_f-u_sum_0:+.4f})")
    print(f"cycle drift: {(u_sum_f + lam_sum_f) - cycle_q_0:+.4f}")

    save_timeseries(history, args.plot_dir, args.plot_prefix, R)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        path = args.save + ".npz"
        np.savez_compressed(
            path,
            u=lat.u,
            phi=lat.phi,
            lam=lat.lam,
            pos=lat.pos,
            R=np.float64(R),
            dt=np.float64(dt),
            lam_sum_init=np.float64(lam_sum_0),
            lam_sum_final=np.float64(lam_sum_f),
            u_sum_final=np.float64(u_sum_f),
        )
        print(f"lattice saved → {path}")

    print("done")


if __name__ == "__main__":
    main()
