"""
Probe: Hopf-linked spheromak decay (step_cross) inside a Saha-equilibrated vacuum.

Combines two previously-separate testbeds:
  - run/substrate/spheromak_decay.py   (Ring A + Ring B, step_cross topology
    opening, cold/quiet vacuum)
  - run/substrate/probe_saha_spheromak.py   (ReciprocalBC + thermal noise bath,
    single ring, no decay mechanism)

Question: the cross_scale calibration (cs=0.005 viable, cs=0.010 too aggressive)
was done in a cold vacuum that decays down to a *numerical* noise floor
(H ~ 1e-10) over 200k steps. The Saha vacuum has a much larger *physical* noise
floor (u_std ~ 0.3-0.4 from thermal boundary driving). This probe checks whether
circ_B/H can still be distinguished from that physical noise floor, and if so
for how long / at what cross_scale.

A third, undecorated "control" ring (no lam, no imprinted phi) is built far from
the real spheromak and measured with the same diagnostic — it gives a live,
empirical noise floor for the circulation measurement under the current thermal
bath, so "is the signal still separated from the noise" is a direct comparison
rather than a guess.

Run:
  PYTHONPATH=$PWD python3 run/substrate/probe_saha_spheromak_decay.py
  PYTHONPATH=$PWD python3 run/substrate/probe_saha_spheromak_decay.py --cross-scale 0.004 --n-steps 40000
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.substrate.flux_tube import (
    _TOROIDAL_PATTERNS,
    _trace_loop,
    find_toroidal_ring,
    find_toroidal_spheromak,
    init_resonant_loop,
    init_spheromak,
    phi_ring_circulation,
)
from equilibrium.substrate.lattice import DampedLattice

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Reciprocal BC — thermal bath (same as probe_saha_spheromak.py)
# ---------------------------------------------------------------------------
class ReciprocalBC:
    def __init__(self, lat, u_target, rng):
        nb = lat.nb
        boundary = np.any(nb < 0, axis=1) & ~lat.fixed_u
        self._idx = np.where(boundary)[0]
        self._target = float(u_target)
        self._rng = rng

    def step(self, lat, noise_amp=0.0):
        deficit = self._target - float(lat.u.sum())
        n = len(self._idx)
        if n == 0:
            return
        w = self._rng.exponential(size=n)
        w /= w.sum()
        inj = deficit * w
        if noise_amp > 0:
            noise = noise_amp * self._rng.standard_normal(size=n)
            noise -= noise.mean()
            inj += noise
        lat.u[self._idx] += inj


def find_ring_far(lat, target_pos, axis, exclude):
    """Find a toroidal ring near target_pos, avoiding tiles in `exclude`."""
    face_seq, _ = _TOROIDAL_PATTERNS[axis]
    dists = np.linalg.norm(lat.pos - np.array(target_pos), axis=1)
    for start in np.argsort(dists)[:200]:
        tiles, faces = _trace_loop(lat, int(start), face_seq)
        if tiles is not None and not (set(tiles) & exclude):
            return tiles, faces
    return None, None


def main():
    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=20.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument(
        "--lam-rings", type=float, default=1.0, help="lam on ring tiles (0=vacuum)"
    )
    parser.add_argument("--lam-scale", type=float, default=0.05)
    parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
    parser.add_argument(
        "--cross-scale", type=float, default=0.005, help="step_cross scale (0=off)"
    )
    parser.add_argument("--axis", default="z")
    parser.add_argument("--rings", choices=["both", "single"], default="both")
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--report", type=int, default=1000)
    parser.add_argument("--sponge", type=float, default=0.15)
    parser.add_argument("--damp-max", type=float, default=0.003)
    parser.add_argument(
        "--u-bg", type=float, default=1.0, help="Saha vacuum background level"
    )
    parser.add_argument(
        "--noise", type=float, default=0.05, help="thermal noise amplitude at boundary"
    )
    parser.add_argument("--out", type=str, default="saha_spheromak_decay.png")
    parser.add_argument("--save-snapshot", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Build lattice + IC
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    lat = DampedLattice(
        radius_cells=args.R,
        u_bg=args.u_bg,
        sponge_fraction=args.sponge,
        damp_max=args.damp_max,
    )
    lat.u[:] = args.u_bg
    dt = lat.stable_dt()

    if args.rings == "both":
        A_tiles, A_faces, B_tiles, B_faces = find_toroidal_spheromak(
            lat, axis=args.axis
        )
        if A_tiles is None:
            print("ERROR: no toroidal spheromak found — lattice too small?")
            sys.exit(1)
        omega_A, omega_B = init_spheromak(
            lat,
            A_tiles,
            A_faces,
            B_tiles,
            B_faces,
            amplitude=args.amplitude,
            frozen=False,
            phi_only=True,
        )
        print(
            f"Ring A: {len(A_tiles)} tiles  Ring B: {len(B_tiles)} tiles  "
            f"omega_A={omega_A:.4f}  omega_B={omega_B:.4f}"
        )
    else:
        A_tiles, A_faces = find_toroidal_ring(lat, axis=args.axis)
        if A_tiles is None:
            print("ERROR: no toroidal ring found — lattice too small?")
            sys.exit(1)
        B_tiles = []
        omega_A = init_resonant_loop(
            lat, A_tiles, A_faces, amplitude=args.amplitude, phi_only=True
        )
        print(f"Ring A: {len(A_tiles)} tiles (single)  omega_A={omega_A:.4f}")

    ring_idx = set(A_tiles) | set(B_tiles)
    if args.lam_rings > 0.0:
        for i in ring_idx:
            lat.lam[i] = args.lam_rings

    centre_A = lat.pos[A_tiles].mean(axis=0)
    centre_B = lat.pos[B_tiles].mean(axis=0) if B_tiles else None

    # Control ring: same size/shape, far from the real structure, no lam/phi imprinted.
    # Gives a live empirical noise floor for the circulation diagnostic under the
    # current thermal bath.
    ctrl_target = np.array([0.55 * args.R, 0.0, 0.0])
    ctrl_tiles, ctrl_faces = find_ring_far(
        lat, ctrl_target, args.axis, exclude=ring_idx
    )
    if ctrl_tiles is None:
        print("WARN: no control ring found — noise floor comparison disabled")
        ctrl_tiles = None
    else:
        centre_ctrl = lat.pos[ctrl_tiles].mean(axis=0)
        print(
            f"Control ring: {len(ctrl_tiles)} tiles  centre={centre_ctrl.round(2)}  "
            f"r={np.linalg.norm(centre_ctrl):.2f}  (no lam/phi imprinted)"
        )

    print(
        f"N={lat.N}  R={args.R}  cross_scale={args.cross_scale}  lam_scale={args.lam_scale}  "
        f"u_bg={args.u_bg}  noise={args.noise}  dt={dt:.4e}"
    )

    bc = ReciprocalBC(lat, u_target=float(lat.u.sum()), rng=rng)

    # ---------------------------------------------------------------------------
    # History
    # ---------------------------------------------------------------------------
    n_rep = args.n_steps // args.report + 1
    t_hist = np.zeros(n_rep)
    H_hist = np.zeros(n_rep)
    phi_rms_h = np.zeros(n_rep)
    u_mean_h = np.zeros(n_rep)
    u_std_h = np.zeros(n_rep)
    lam_A_h = np.zeros(n_rep)
    lam_B_h = np.zeros(n_rep)
    circ_A_h = np.zeros(n_rep)
    circ_B_h = np.zeros(n_rep)
    circ_ctrl_h = np.zeros(n_rep)

    has_B = bool(B_tiles)

    def record(si, step):
        t_hist[si] = step * dt
        H_hist[si] = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
        phi_rms_h[si] = float(np.sqrt(np.mean(lat.phi**2)))
        u_mean_h[si] = float(lat.u.mean())
        u_std_h[si] = float(lat.u.std())
        lam_A_h[si] = float(lat.lam[A_tiles].mean())
        lam_B_h[si] = float(lat.lam[B_tiles].mean()) if has_B else 0.0
        circ_A_h[si] = phi_ring_circulation(lat, A_tiles, centre_A, axis=args.axis)
        circ_B_h[si] = (
            phi_ring_circulation(lat, B_tiles, centre_B, axis=args.axis)
            if has_B
            else 0.0
        )
        if ctrl_tiles is not None:
            circ_ctrl_h[si] = phi_ring_circulation(
                lat, ctrl_tiles, centre_ctrl, axis=args.axis
            )

    hdr = f"{'step':>6}  {'H':>10}  {'phi_rms':>10}  {'u_std':>7}  {'lam_A':>6}"
    hdr += f"  {'lam_B':>6}" if has_B else ""
    hdr += f"  {'circ_A':>9}"
    hdr += f"  {'circ_B':>9}" if has_B else ""
    hdr += f"  {'circ_ctrl':>9}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    si = 0
    record(si, 0)

    def _print_row(si, step):
        row = f"{step:6d}  {H_hist[si]:10.4e}  {phi_rms_h[si]:10.4e}  {u_std_h[si]:7.4f}  {lam_A_h[si]:6.4f}"
        if has_B:
            row += f"  {lam_B_h[si]:6.4f}"
        row += f"  {circ_A_h[si]:+9.3e}"
        if has_B:
            row += f"  {circ_B_h[si]:+9.3e}"
        row += f"  {circ_ctrl_h[si]:+9.3e}"
        print(row, flush=True)

    _print_row(0, 0)

    for step in range(1, args.n_steps + 1):
        lat.step(dt)
        if args.cross_scale > 0.0:
            lat.step_cross(dt, scale=args.cross_scale)
        if args.lam_mode == "dynamic":
            lat.step_lam(dt, scale=args.lam_scale)
        bc.step(lat, noise_amp=args.noise)

        if step % args.report == 0:
            si += 1
            record(si, step)
            _print_row(si, step)

    # ---------------------------------------------------------------------------
    # Save snapshot
    # ---------------------------------------------------------------------------
    if args.save_snapshot:
        save_kwargs = dict(
            phi=lat.phi,
            lam=lat.lam,
            u=lat.u,
            A_tiles=np.array(list(A_tiles), dtype=int),
        )
        if has_B:
            save_kwargs["B_tiles"] = np.array(list(B_tiles), dtype=int)
        np.savez(args.save_snapshot, **save_kwargs)
        print(f"snapshot saved -> {args.save_snapshot}")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Saha-vacuum spheromak decay  R={args.R}  cross={args.cross_scale}  "
        f"lam_scale={args.lam_scale}  noise={args.noise}",
        fontsize=11,
    )
    ts = t_hist[: si + 1]

    ax = axes[0, 0]
    ax.semilogy(ts, H_hist[: si + 1], color="steelblue", label="H")
    ax.semilogy(ts, phi_rms_h[: si + 1], color="darkorange", label="phi_rms")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude (log)")
    ax.set_title("Total energy / phi_rms")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(ts, u_std_h[: si + 1], color="seagreen", label="u_std (Saha bath)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("u_std")
    ax.set_title("Thermal bath amplitude")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(ts, np.abs(circ_A_h[: si + 1]), color="steelblue", label="|circ_A|")
    if has_B:
        ax.plot(ts, np.abs(circ_B_h[: si + 1]), color="tomato", label="|circ_B|")
    ax.plot(
        ts,
        np.abs(circ_ctrl_h[: si + 1]),
        color="gray",
        ls="--",
        label="|circ_ctrl| (noise floor)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("|circulation| (log)")
    ax.set_title("Ring circulation vs empirical noise floor")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(ts, lam_A_h[: si + 1], color="steelblue", label="lam_A")
    if has_B:
        ax.plot(ts, lam_B_h[: si + 1], color="tomato", label="lam_B")
    ax.set_xlabel("time")
    ax.set_ylabel("lam")
    ax.set_title("Ring lam")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nplot saved -> {args.out}")


if __name__ == "__main__":
    main()
