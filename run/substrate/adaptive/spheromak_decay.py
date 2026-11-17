"""
DST spheromak decay — adaptive lattice testbed.

Mirrors run/substrate/spheromak_decay.py on the AdaptiveLattice C++ backend.

Hopf-linked IC (Ring A: 8-tile toroidal, Ring B: 4-tile plaquette) with phi_only=True
to avoid the parametric instability from independent resonant u patterns on hex-adjacent
rings.  step_cross opens topology; step_lam relaxes the winding field.

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay.py --R 40 --n-steps 20000
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay.py --load snap.npz --cross-scale 0.005

Snapshot workflow:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay.py \\
      --R 40 --n-steps 20000 --save-snapshot proton_adaptive_R40.npz
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay.py \\
      --load proton_adaptive_R40.npz --n-steps 20000 --save-snapshot proton_adaptive_R40_decay2.npz
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    find_toroidal_spheromak,
    init_toroidal_winding,
    phi_ring_circulation,
    phi_toroidal_circulation,
)

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
    "--lam-relax-rate",
    type=float,
    default=0.0,
    help="rate for step_lam_relax: dlam/dt = rate*(alpha*|phi| - lam)  (0=off)",
)
parser.add_argument(
    "--lam-alpha",
    type=float,
    default=1.0,
    help="phi→lam coupling: lam_target = alpha * |phi|",
)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--axis", default="z")
parser.add_argument("--rings", choices=["both", "single"], default="both")
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15, help="sponge fraction of R")
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--max-level", type=int, default=0)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument("--plot-out", type=str, default="spheromak_decay_adaptive.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--save-snapshot", type=str, default=None)
parser.add_argument("--load", type=str, default=None)

R_SCAN = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _g_to_j_map(cl):
    return {int(g): j for j, g in enumerate(cl.leaf_global_indices)}


def _radial_profile(pos, values, centre, r_edges):
    r = np.linalg.norm(pos - centre, axis=1)
    out = []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        m = (r >= r0) & (r < r1)
        out.append(float(values[m].mean()) if m.any() else 0.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    if args.max_level > 0:
        # Ring tile indices (A_idx, B_idx) are stored as leaf-j positions which
        # go stale the moment adapt() splits or coarsens cells.  dt also needs
        # recomputing after each adapt (L_min changes with level).  Both must
        # be fixed before AMR is enabled here.
        raise NotImplementedError(
            "max_level > 0 not yet supported: ring index and dt staleness after adapt()"
        )

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]

    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    # Graduated quadratic sponge BC (volumetric, matches DampedLattice)
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    centre = np.zeros(3, dtype=float)

    # -------------------------------------------------------------------
    # IC init
    # -------------------------------------------------------------------
    if args.load:
        data = np.load(args.load)
        phi_arr = data["phi"]
        if phi_arr.shape[0] != 14:  # loaded from uniform lattice (N,14)
            phi_arr = phi_arr.T
        cl.set_leaf_phi(phi_arr)
        cl.set_leaf_lam(data["lam"])
        cl.set_leaf_u(data["u"])
        A_js = list(data["A_tiles"].astype(int))
        B_js = list(data["B_tiles"].astype(int)) if "B_tiles" in data else []
        ring_idx = set(A_js + B_js)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        print(
            f"Loaded {args.load}: N={cl.n_leaves}  R={args.R}  "
            f"N_A={len(A_js)}  N_B={len(B_js)}  r≈{ring_r:.2f}"
        )

    elif args.rings == "both":
        print(f"N={cl.n_leaves}  R={args.R}  rings=both")
        A_js, A_faces, B_js, B_faces = find_toroidal_spheromak(cl, axis=args.axis)
        if A_js is None:
            print("ERROR: no toroidal spheromak found — lattice too small?")
            sys.exit(1)
        omega_A = init_toroidal_winding(
            cl, A_js, A_faces, amplitude=args.amplitude, phi_only=True
        )
        omega_B = init_toroidal_winding(
            cl, B_js, B_faces, amplitude=args.amplitude, phi_only=True
        )
        ring_idx = set(A_js + B_js)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        print(f"Ring A: {len(A_js)} tiles  Ring B: {len(B_js)} tiles  r≈{ring_r:.2f}")
        print(f"ω_A={omega_A:.4f}  ω_B={omega_B:.4f}")
        if args.lam_rings > 0.0:
            lam = np.array(cl.leaf_lam)
            for i in ring_idx:
                lam[i] = args.lam_rings
            cl.set_leaf_lam(lam)

    else:  # single
        print(f"N={cl.n_leaves}  R={args.R}  rings=single")
        A_js, A_faces = find_toroidal_ring(cl, axis=args.axis)
        if A_js is None:
            print("ERROR: no toroidal ring found — lattice too small?")
            sys.exit(1)
        B_js = []
        omega_A = init_toroidal_winding(
            cl, A_js, A_faces, amplitude=args.amplitude, phi_only=True
        )
        ring_idx = set(A_js)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        print(f"Ring A: {len(A_js)} tiles (single)  r≈{ring_r:.2f}  ω_A={omega_A:.4f}")
        if args.lam_rings > 0.0:
            lam = np.array(cl.leaf_lam)
            for i in ring_idx:
                lam[i] = args.lam_rings
            cl.set_leaf_lam(lam)

    A_idx = list(A_js)
    B_idx = list(B_js)
    has_B = bool(B_idx)

    dt = compute_dt(cl)
    phi_arr = np.array(cl.leaf_phi)  # (14, N)
    phi_rms = float(np.sqrt(np.mean(phi_arr**2)))
    print(
        f"lam_rings={args.lam_rings}  lam_mode={args.lam_mode}  lam_scale={args.lam_scale}"
    )
    print(f"cross_scale={args.cross_scale}  phi_rms={phi_rms:.4e}  dt={dt:.4e}")

    r_edges = np.arange(0, args.R * 0.85, 1.0)
    r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])

    n_snaps = args.n_steps // args.report + 1
    times = np.zeros(n_snaps)
    H_hist = np.zeros(n_snaps)
    circ_hist = np.zeros((n_snaps, len(R_SCAN)))
    phi_rms_h = np.zeros(n_snaps)
    lam_ring_h = np.zeros(n_snaps)
    lam_A_h = np.zeros(n_snaps)
    lam_B_h = np.zeros(n_snaps)
    circ_A_h = np.zeros(n_snaps)
    circ_B_h = np.zeros(n_snaps)

    snap_steps = sorted({0, args.n_steps // 4, args.n_steps // 2, args.n_steps})
    snap_profiles = {}
    snap_2d = {}

    def record(si, step):
        phi_a = np.array(cl.leaf_phi)  # (14, N)
        u_a = np.array(cl.leaf_u)
        lam_a = np.array(cl.leaf_lam)
        times[si] = step * dt
        H_hist[si] = float(np.sum(u_a**2) + 0.5 * np.sum(phi_a**2))
        phi_rms_h[si] = float(np.sqrt(np.mean(phi_a**2)))
        lam_ring_h[si] = float(np.mean([lam_a[i] for i in ring_idx]))
        lam_A_h[si] = float(np.mean([lam_a[i] for i in A_idx])) if A_idx else 0.0
        lam_B_h[si] = float(np.mean([lam_a[i] for i in B_idx])) if B_idx else 0.0
        circ_A_h[si] = phi_ring_circulation(cl, A_idx, centre)
        circ_B_h[si] = phi_ring_circulation(cl, B_idx, centre)
        for ki, r in enumerate(R_SCAN):
            circ_hist[si, ki] = phi_toroidal_circulation(cl, centre, r)

    has_B = bool(B_idx)
    r_cols = "  ".join(f"r={r:2d}" for r in R_SCAN)
    lam_hdr = "  lam_A   lam_B" if has_B else "  lam_A"
    circ_hdr = "  circ_A     circ_B" if has_B else "  circ_A"
    print(
        f"\n{'step':>6}  {'H':>10}  {'phi_rms':>10}  {'lam_ring':>8}{lam_hdr}{circ_hdr}  {r_cols}"
    )
    print("  " + "-" * (65 + 8 * len(R_SCAN)))

    si = 0
    for step in range(args.n_steps + 1):
        if step % args.report == 0:
            record(si, step)
            cs = "  ".join(f"{circ_hist[si, ki]:+.2e}" for ki in range(len(R_SCAN)))
            lam_extra = (
                f"  {lam_A_h[si]:6.4f}  {lam_B_h[si]:6.4f}"
                if has_B
                else f"  {lam_A_h[si]:6.4f}"
            )
            circ_extra = (
                f"  {circ_A_h[si]:+.3e}  {circ_B_h[si]:+.3e}"
                if has_B
                else f"  {circ_A_h[si]:+.3e}"
            )
            print(
                f"{step:6d}  {H_hist[si]:10.4e}  {phi_rms_h[si]:10.4e}"
                f"  {lam_ring_h[si]:8.4f}{lam_extra}{circ_extra}  {cs}",
                flush=True,
            )
            si += 1

        if step in snap_steps:
            u_a = np.array(cl.leaf_u)
            lam_a = np.array(cl.leaf_lam)
            pos_a = np.array(cl.leaf_pos)
            up = _radial_profile(pos_a, u_a, centre, r_edges)
            lp = _radial_profile(pos_a, lam_a, centre, r_edges)
            snap_profiles[step] = (up, lp)

        if step == 0 or step == args.n_steps:
            snap_2d[step] = {
                "phi": np.array(cl.leaf_phi).copy(),
                "lam": np.array(cl.leaf_lam).copy(),
                "u": np.array(cl.leaf_u).copy(),
            }

        if step < args.n_steps:
            alat.step(0, dt)
            alat.step_sponge(0)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale)
            if args.lam_mode == "dynamic":
                alat.step_lam(0, dt, args.lam_scale)
            if args.lam_relax_rate > 0.0:
                step_lam_relax(cl, dt, args.lam_relax_rate, args.lam_alpha)

    # -------------------------------------------------------------------
    # Save snapshot
    # -------------------------------------------------------------------
    if args.save_snapshot:
        save_kwargs = dict(
            phi=np.array(cl.leaf_phi).T,  # (N, 14) — matches uniform lattice convention
            lam=np.array(cl.leaf_lam),
            u=np.array(cl.leaf_u),
            pos=np.array(cl.leaf_pos),
            A_tiles=np.array(A_idx, dtype=int),
        )
        if B_idx:
            save_kwargs["B_tiles"] = np.array(B_idx, dtype=int)
        np.savez(args.save_snapshot, **save_kwargs)
        print(f"snapshot saved → {args.save_snapshot}")

    if args.no_plot:
        return

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    n_prof = len(snap_profiles)
    fig = plt.figure(figsize=(5 + 3 * n_prof, 10))
    fig.suptitle(
        f"DST spheromak decay (adaptive) — cross={args.cross_scale}"
        f"  lam_mode={args.lam_mode}  lam_rings={args.lam_rings}",
        fontsize=10,
    )
    gs = gridspec.GridSpec(
        3,
        1 + n_prof,
        figure=fig,
        width_ratios=[1.6] + [1.0] * n_prof,
        hspace=0.48,
        wspace=0.38,
    )

    ax_H = fig.add_subplot(gs[0, 0])
    ax_circ = fig.add_subplot(gs[1, 0])
    ax_rc = fig.add_subplot(gs[2, 0])

    ax_H.semilogy(times[:si], H_hist[:si], color="steelblue")
    ax_H.set_ylabel("H (log)")
    ax_H.set_xlabel("time")
    ax_H.set_title("Total energy")

    cmap_in = plt.cm.Blues(np.linspace(0.4, 0.9, len(R_SCAN)))
    cmap_ex = plt.cm.Reds(np.linspace(0.4, 0.9, len(R_SCAN)))
    for ki, r in enumerate(R_SCAN):
        color = cmap_in[ki] if r <= ring_r + 1 else cmap_ex[ki]
        ls = "-" if r <= ring_r + 1 else "--"
        ax_circ.plot(times[:si], circ_hist[:si, ki], color=color, ls=ls, label=f"r={r}")
    ax_circ.axhline(0, color="k", lw=0.5)
    ax_circ.set_ylabel("φ toroidal circ.")
    ax_circ.set_xlabel("time")
    ax_circ.set_title("Shell circulation: interior (blue) / exterior (red)")
    ax_circ.legend(fontsize=6, ncol=2)

    ax_rc.plot(times[:si], circ_A_h[:si], color="steelblue", label="ring A")
    if has_B:
        ax_rc.plot(times[:si], circ_B_h[:si], color="tomato", label="ring B")
    ax_rc.axhline(0, color="k", lw=0.5)
    ax_rc.set_ylabel("φ circ.")
    ax_rc.set_xlabel("time")
    ax_rc.set_title("Per-ring φ circulation (A=proton / B=electron)")
    ax_rc.legend(fontsize=7)

    pos_a = np.array(cl.leaf_pos)
    for col, (step, (up, lp)) in enumerate(sorted(snap_profiles.items())):
        t_val = step * dt
        ax_u = fig.add_subplot(gs[0, 1 + col])
        ax_u.plot(r_mids, up, color="steelblue")
        ax_u.axvline(ring_r, color="k", lw=0.7, ls="--")
        ax_u.set_xlabel("r")
        ax_u.set_title(f"u(r)  t={t_val:.1f}")
        if col == 0:
            ax_u.set_ylabel("u")

        ax_l = fig.add_subplot(gs[1, 1 + col])
        ax_l.plot(r_mids, lp, color="darkorange")
        ax_l.axvline(ring_r, color="k", lw=0.7, ls="--")
        ax_l.set_xlabel("r")
        ax_l.set_title(f"lam(r)  t={t_val:.1f}")
        if col == 0:
            ax_l.set_ylabel("lam")

        ax_s = fig.add_subplot(gs[2, 1 + col])
        ax_s.plot(r_mids, 1.0 / (1.0 + lp), color="mediumseagreen")
        ax_s.axvline(ring_r, color="k", lw=0.7, ls="--")
        ax_s.set_xlabel("r")
        ax_s.set_title(f"σ(r)  t={t_val:.1f}")
        if col == 0:
            ax_s.set_ylabel("σ")

    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved → {args.plot_out}")


if __name__ == "__main__":
    main()
