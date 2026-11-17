"""
DST spheromak decay with lam-poles variant — adaptive lattice testbed.

Mirrors run/substrate/spheromak_decay_lam_poles.py on the AdaptiveLattice
C++ backend.

Key differences from test_spheromak_decay.py:
  --lam-poles  lam imposed on axial pole tiles (default 0 = emergent)
  --u-bg       sponge BC reservoir tension (default 0 = absorbing)
  --dc         ring A DC circulation mode
  pole_drain diagnostic: net u change at pole tiles per step

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay_lam_poles.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay_lam_poles.py \\
      --u-bg 1e-10 --lam-poles 0 --n-steps 20000
  PYTHONPATH=$PWD python3 run/substrate/adaptive/spheromak_decay_lam_poles.py \\
      --load snap.npz --cross-scale 0.005 --u-bg 1e-10
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    _MIRROR,
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
parser.add_argument(
    "--lam-poles", type=float, default=0.0, help="lam on pole tiles (0=emergent)"
)
parser.add_argument(
    "--pole-r", type=float, default=1.0, help="xy-radius cutoff for pole cluster"
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument(
    "--u-bg", type=float, default=0.0, help="sponge BC reservoir u (0=absorbing)"
)
parser.add_argument("--axis", default="z")
parser.add_argument("--rings", choices=["both", "single", "dc"], default="both")
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--max-level", type=int, default=0)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument(
    "--plot-out", type=str, default="spheromak_decay_lam_poles_adaptive.png"
)
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--save-snapshot", type=str, default=None)
parser.add_argument("--load", type=str, default=None)

R_SCAN = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _radial_profile(pos, values, centre, r_edges):
    r = np.linalg.norm(pos - centre, axis=1)
    out = []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        m = (r >= r0) & (r < r1)
        out.append(float(values[m].mean()) if m.any() else 0.0)
    return np.array(out)


def _pole_tiles(pos, ring_set, pole_r, axis="z"):
    """Tight axial clusters at the two poles of the dipole axis, excluding ring tiles."""
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    perp = [i for i in range(3) if i != ax]
    r_perp = np.sqrt(pos[:, perp[0]] ** 2 + pos[:, perp[1]] ** 2)
    on_axis = r_perp < pole_r
    a = pos[:, ax]
    pos_tiles = [int(i) for i in np.where(on_axis & (a > 0))[0] if i not in ring_set]
    neg_tiles = [int(i) for i in np.where(on_axis & (a < 0))[0] if i not in ring_set]
    return pos_tiles, neg_tiles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    if args.max_level > 0:
        raise NotImplementedError(
            "max_level > 0 not yet supported: ring index and dt staleness after adapt()"
        )

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]

    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=args.u_bg, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    centre = np.zeros(3, dtype=float)

    # -----------------------------------------------------------------------
    # IC init
    # -----------------------------------------------------------------------
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
        if args.lam_poles > 0.0:
            pos_p, neg_p = _pole_tiles(
                np.array(cl.leaf_pos), ring_idx, args.pole_r, args.axis
            )
            lam = np.array(cl.leaf_lam)
            for i in pos_p + neg_p:
                lam[i] = args.lam_poles
            cl.set_leaf_lam(lam)
            print(
                f"Pole lam: +={len(pos_p)} tiles  -={len(neg_p)} tiles  lam={args.lam_poles}"
            )

    elif args.rings == "single":
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
        if args.lam_poles > 0.0:
            pos_p, neg_p = _pole_tiles(
                np.array(cl.leaf_pos), ring_idx, args.pole_r, args.axis
            )
            lam = np.array(cl.leaf_lam)
            for i in pos_p + neg_p:
                lam[i] = args.lam_poles
            cl.set_leaf_lam(lam)
            print(
                f"Pole lam: +={len(pos_p)} tiles  -={len(neg_p)} tiles  lam={args.lam_poles}"
            )

    else:  # dc
        print(f"N={cl.n_leaves}  R={args.R}  rings=dc")
        A_js, A_faces = find_toroidal_ring(cl, axis=args.axis)
        if A_js is None:
            print("ERROR: no toroidal ring found — lattice too small?")
            sys.exit(1)
        B_js = []
        N_ring = len(A_js)
        phi = np.array(cl.leaf_phi)  # (14, N)
        for idx in range(N_ring):
            j_i = A_js[idx]
            j_nxt = A_js[(idx + 1) % N_ring]
            k = A_faces[idx]
            mk = int(_MIRROR[k])
            phi[k, j_i] = args.amplitude
            phi[mk, j_nxt] = -args.amplitude
        cl.set_leaf_phi(phi)
        ring_idx = set(A_js)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        div_ring = sum(sum(float(phi[k, i]) for k in range(14)) for i in ring_idx)
        print(f"Ring A: {len(A_js)} tiles (DC circulation)  r≈{ring_r:.2f}")
        print(
            f"amplitude={args.amplitude}  div(phi) on ring = {div_ring:.2e}  (should be ≈0)"
        )
        if args.lam_rings > 0.0:
            lam = np.array(cl.leaf_lam)
            for i in ring_idx:
                lam[i] = args.lam_rings
            cl.set_leaf_lam(lam)
        if args.lam_poles > 0.0:
            pos_p, neg_p = _pole_tiles(
                np.array(cl.leaf_pos), ring_idx, args.pole_r, args.axis
            )
            lam = np.array(cl.leaf_lam)
            for i in pos_p + neg_p:
                lam[i] = args.lam_poles
            cl.set_leaf_lam(lam)
            print(
                f"Pole lam: +={len(pos_p)} tiles  -={len(neg_p)} tiles  lam={args.lam_poles}"
            )

    A_idx = list(A_js)
    B_idx = list(B_js)
    has_B = bool(B_idx)

    # Pole tile indices for rate tracking
    _pos_p, _neg_p = _pole_tiles(
        np.array(cl.leaf_pos), ring_idx, args.pole_r, args.axis
    )
    pole_idx = np.array(_pos_p + _neg_p, dtype=int)

    dt = compute_dt(cl)
    phi_arr = np.array(cl.leaf_phi)
    phi_rms = float(np.sqrt(np.mean(phi_arr**2)))
    print(
        f"lam_poles={args.lam_poles}  lam_mode={args.lam_mode}  lam_scale={args.lam_scale}  u_bg={args.u_bg}"
    )
    print(f"cross_scale={args.cross_scale}  phi_rms={phi_rms:.4e}  dt={dt:.4e}")
    print(f"Pole tiles for rate tracking: {len(pole_idx)}")

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
    pole_drain_h = np.zeros(n_snaps)

    snap_steps = sorted({0, args.n_steps // 4, args.n_steps // 2, args.n_steps})
    snap_profiles = {}
    snap_2d = {}

    _u_pole_prev = float(np.array(cl.leaf_u)[pole_idx].sum()) if len(pole_idx) else 0.0
    _pole_acc = 0.0

    def record(si, step):
        nonlocal _pole_acc
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
        pole_drain_h[si] = _pole_acc / max(args.report, 1)
        _pole_acc = 0.0

    r_cols = "  ".join(f"r={r:2d}" for r in R_SCAN)
    lam_hdr = "  lam_A   lam_B" if has_B else "  lam_A"
    circ_hdr = "  circ_A     circ_B" if has_B else "  circ_A"
    print(
        f"\n{'step':>6}  {'H':>10}  {'phi_rms':>10}  {'lam_ring':>8}{lam_hdr}{circ_hdr}"
        f"  {'pole_Δu/step':>13}  {r_cols}"
    )
    print("  " + "-" * (95 + 8 * len(R_SCAN)))

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
                f"  {lam_ring_h[si]:8.4f}{lam_extra}{circ_extra}"
                f"  {pole_drain_h[si]:+13.4e}  {cs}",
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
            u_pole_before = (
                float(np.array(cl.leaf_u)[pole_idx].sum()) if len(pole_idx) else 0.0
            )
            alat.step(0, dt)
            if args.cross_scale > 0.0:
                alat.step_cross(0, dt, args.cross_scale)
            if args.lam_mode == "dynamic":
                alat.step_lam(0, dt, args.lam_scale)
            alat.step_sponge(0)
            _pole_acc += float(np.array(cl.leaf_u)[pole_idx].sum()) - u_pole_before

    # -----------------------------------------------------------------------
    # Save snapshot
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    n_prof = len(snap_profiles)
    fig = plt.figure(figsize=(5 + 3 * n_prof, 12))
    fig.suptitle(
        f"DST spheromak decay (adaptive) — cross={args.cross_scale}"
        f"  lam_mode={args.lam_mode}  lam_poles={args.lam_poles}  u_bg={args.u_bg}",
        fontsize=10,
    )
    gs = gridspec.GridSpec(
        4,
        1 + n_prof,
        figure=fig,
        width_ratios=[1.6] + [1.0] * n_prof,
        hspace=0.48,
        wspace=0.38,
    )

    ax_H = fig.add_subplot(gs[0, 0])
    ax_circ = fig.add_subplot(gs[1, 0])
    ax_rc = fig.add_subplot(gs[2, 0])
    ax_lam = fig.add_subplot(gs[3, 0])

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

    ax_lam.plot(times[:si], phi_rms_h[:si], color="steelblue", label="phi_rms")
    ax_lam.plot(times[:si], lam_ring_h[:si], color="darkorange", label="lam_ring")
    ax_lam.set_ylabel("amplitude")
    ax_lam.set_xlabel("time")
    ax_lam.set_title("phi_rms and ring lam")
    ax_lam.legend(fontsize=7)

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

    # -----------------------------------------------------------------------
    # Before / after 2D midplane maps
    # -----------------------------------------------------------------------
    if len(snap_2d) < 2:
        return

    from scipy.interpolate import griddata

    def midplane_slice(phi_arr, lam_arr, u_arr, clip_r=None):
        """Return (x, y, phi_tor, lam, u) for tiles near z=0."""
        z = pos_a[:, 2]
        ztol = (z.max() - z.min()) / 8.0
        zmid = (z.max() + z.min()) / 2.0
        mask = np.abs(z - zmid) < ztol
        if clip_r is not None:
            r2d = np.linalg.norm(pos_a[:, :2], axis=1)
            mask &= r2d < clip_r

        xv = pos_a[mask, 0]
        yv = pos_a[mask, 1]

        nb_g = np.array(cl.leaf_nb)  # (N, 14)
        leaf_g = cl.leaf_global_indices
        g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

        diff = np.array(
            [
                pos_a[g_to_j.get(int(nb_g[i, k]), i)] - pos_a[i]
                for i in range(pos_a.shape[0])
                for k in range(14)
            ]
        ).reshape(pos_a.shape[0], 14, 3)
        d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
        n_hat = diff / np.where(d_len > 0, d_len, 1.0)

        xi, yi = pos_a[:, 0, np.newaxis], pos_a[:, 1, np.newaxis]
        rho_s = np.sqrt(xi**2 + yi**2)
        rho_s = np.where(rho_s > 1e-10, rho_s, 1.0)
        pha_x = -yi / rho_s
        pha_y = xi / rho_s

        proj = n_hat[..., 0] * pha_x + n_hat[..., 1] * pha_y
        phi_tor = (phi_arr.T * proj).sum(axis=1) / np.maximum(
            (nb_g >= 0).sum(axis=1), 1
        )
        return xv, yv, phi_tor[mask], lam_arr[mask], u_arr[mask]

    clip = args.R * 0.75
    nr = 150
    xi_g = np.linspace(-clip, clip, nr)
    yi_g = np.linspace(-clip, clip, nr)
    Xi, Yi = np.meshgrid(xi_g, yi_g)

    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig2.suptitle(
        f"DST spheromak (adaptive) — before / after"
        f"  (cross={args.cross_scale}  t_final={args.n_steps*dt:.1f})",
        fontsize=11,
    )

    rows = {
        "phi_tor": (0, "φ toroidal (midplane)", "RdBu_r"),
        "lam": (1, "λ (winding)", "viridis"),
    }
    t_labs = {0: "before  t=0", args.n_steps: f"after  t={args.n_steps*dt:.1f}"}

    theta = np.linspace(0, 2 * np.pi, 64)

    for col, step in enumerate(sorted(snap_2d)):
        sd = snap_2d[step]
        xv, yv, phi_t, lam_v, u_v = midplane_slice(
            sd["phi"], sd["lam"], sd["u"], clip_r=clip
        )
        for field, (row, title, cmap) in rows.items():
            vals = phi_t if field == "phi_tor" else lam_v
            Zi = griddata((xv, yv), vals, (Xi, Yi), method="linear")
            ax = axes[row, col]
            vmax = np.nanpercentile(np.abs(Zi), 99) or 1e-6
            if field == "phi_tor":
                im = ax.pcolormesh(
                    Xi, Yi, Zi, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto"
                )
            else:
                im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, shading="auto")
            fig2.colorbar(im, ax=ax, shrink=0.8)
            ax.set_aspect("equal")
            ax.set_title(f"{title}\n{t_labs[step]}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.plot(
                ring_r * np.cos(theta), ring_r * np.sin(theta), "w--", lw=0.7, alpha=0.6
            )

    sd0 = snap_2d[0]
    sdN = snap_2d[args.n_steps]
    xv0, yv0, pt0, lam0, _ = midplane_slice(
        sd0["phi"], sd0["lam"], sd0["u"], clip_r=clip
    )
    xvN, yvN, ptN, lamN, _ = midplane_slice(
        sdN["phi"], sdN["lam"], sdN["u"], clip_r=clip
    )
    for field, (row, title, _) in rows.items():
        v0 = pt0 if field == "phi_tor" else lam0
        vN = ptN if field == "phi_tor" else lamN
        Z0 = griddata((xv0, yv0), v0, (Xi, Yi), method="linear")
        ZN = griddata((xvN, yvN), vN, (Xi, Yi), method="linear")
        dZ = ZN - Z0
        ax = axes[row, 2]
        vm = np.nanpercentile(np.abs(dZ), 99) or 1e-6
        im = ax.pcolormesh(Xi, Yi, dZ, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
        fig2.colorbar(im, ax=ax, shrink=0.8)
        ax.set_aspect("equal")
        ax.set_title(f"Δ{title}\n(after − before)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(
            ring_r * np.cos(theta), ring_r * np.sin(theta), "w--", lw=0.7, alpha=0.6
        )

    before_after = args.plot_out.replace(".png", "_2d.png")
    plt.tight_layout()
    plt.savefig(before_after, dpi=130, bbox_inches="tight")
    print(f"before/after plot saved → {before_after}")


if __name__ == "__main__":
    main()
