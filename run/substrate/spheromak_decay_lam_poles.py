"""
DST spheromak decay — Python model testbed.

The neutron in DST is a Hopf-linked pair of flux loops (Ring A: 8-tile toroidal
ring, Ring B: 4-tile plaquette threading through it).  This is the DST-native
spheromak — no MHD analytical fields imported.

step_cross is the topology-change mechanism: inward phi on one face redistributes
to the 6 edge-adjacent faces, allowing closed winding loops to open.  Its scale
sets the decay rate — the DST analog of the weak coupling constant.

Key observables:
  - phi_circulation at shells r=1..15 (interior vs exterior)
  - Interior circulation decaying = topology opening up
  - Exterior circulation growing = field leaking into surroundings (dipole field)

Run:
  PYTHONPATH=$PWD python3 run/substrate/spheromak_decay.py
  PYTHONPATH=$PWD python3 run/substrate/spheromak_decay.py --cross-scale 0.1
  PYTHONPATH=$PWD python3 run/substrate/spheromak_decay.py --cross-scale 0.0 --n-steps 5000

Snapshot workflow (exp 2 — freeze and test):
  # Run to a mid-decay state and save:
  PYTHONPATH=$PWD python3 run/substrate/spheromak_decay.py \
    --rings both --cross-scale 0.001 --n-steps 12000 --save-snapshot proton_candidate.npz
  # Reload and continue with step_cross off:
  PYTHONPATH=$PWD python3 run/substrate/spheromak_decay.py \
    --load proton_candidate.npz --cross-scale 0 --n-steps 8000
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.flux_tube import (
    _MIRROR,
    find_toroidal_ring,
    find_toroidal_spheromak,
    init_resonant_loop,
    init_spheromak,
    phi_ring_circulation,
    phi_toroidal_circulation,
)
from equilibrium.substrate.lattice import DampedLattice
from tst.substrate import radial_profile

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0, help="lattice sphere radius")
parser.add_argument("--amplitude", type=float, default=1.0, help="ring phi amplitude")
parser.add_argument(
    "--lam-poles",
    type=float,
    default=1.0,
    help="lam on pole tiles (z-axis ±z of ring plane)",
)
parser.add_argument(
    "--pole-r", type=float, default=1.0, help="xy-radius cutoff for pole cluster"
)
parser.add_argument(
    "--lam-scale", type=float, default=0.05, help="step_lam scale factor"
)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument(
    "--cross-scale", type=float, default=0.01, help="step_cross scale (0=off)"
)
parser.add_argument(
    "--u-bg", type=float, default=0.0, help="sponge BC reservoir u (0=absorbing)"
)
parser.add_argument("--axis", default="z", help="spheromak axis")
parser.add_argument(
    "--rings",
    choices=["both", "single", "dc"],
    default="both",
    help="both=Hopf-linked pair (neutron); single=ring A traveling wave; "
    "dc=ring A DC circulation (div-free static equilibrium)",
)
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=1, help="threads for step_cpp")
parser.add_argument("--plot-out", type=str, default="spheromak_decay.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument(
    "--save-snapshot",
    type=str,
    default=None,
    help="save phi/lam/u + ring tile indices to .npz at end of run",
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="load phi/lam/u + ring tile indices from .npz (skips IC init)",
)

R_SCAN = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pole_tiles(pos, ring_tiles, pole_r, axis="z"):
    """Tight axial clusters at the two poles of the dipole axis, excluding ring tiles."""
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    perp = [i for i in range(3) if i != ax]
    r_perp = np.sqrt(pos[:, perp[0]] ** 2 + pos[:, perp[1]] ** 2)
    on_axis = r_perp < pole_r
    a = pos[:, ax]
    ring_set = set(ring_tiles)
    pos_tiles = [int(i) for i in np.where(on_axis & (a > 0))[0] if i not in ring_set]
    neg_tiles = [int(i) for i in np.where(on_axis & (a < 0))[0] if i not in ring_set]
    return pos_tiles, neg_tiles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    lat = DampedLattice(
        radius_cells=args.R,
        sponge_fraction=args.sponge,
        damp_max=args.damp_max,
        u_bg=args.u_bg,
    )
    centre = np.zeros(3, dtype=float)
    r_all = np.linalg.norm(lat.pos, axis=1)

    # -----------------------------------------------------------------------
    # IC init — either load snapshot or build from scratch
    # -----------------------------------------------------------------------
    if args.load:
        data = np.load(args.load)
        lat.u[:] = data["u"]
        lat.phi[:] = data["phi"]
        lat.lam[:] = data["lam"]
        A_tiles = list(data["A_tiles"].astype(int))
        B_tiles = list(data["B_tiles"].astype(int)) if "B_tiles" in data else []
        ring_idx = set(A_tiles + B_tiles)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        print(
            f"Loaded {args.load}: N={lat.N}  R={args.R}  "
            f"N_A={len(A_tiles)}  N_B={len(B_tiles)}  r≈{ring_r:.2f}"
        )

    elif args.rings == "both":
        print(f"N={lat.N}  R={args.R}  rings=both")
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
        ring_idx = set(A_tiles + B_tiles)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        print(
            f"Ring A: {len(A_tiles)} tiles  Ring B: {len(B_tiles)} tiles  r≈{ring_r:.2f}"
        )
        print(f"ω_A={omega_A:.4f}  ω_B={omega_B:.4f}")
        if args.lam_poles > 0.0:
            pos_p, neg_p = _pole_tiles(lat.pos, list(ring_idx), args.pole_r, args.axis)
            for i in pos_p + neg_p:
                lat.lam[i] = args.lam_poles
            print(
                f"Pole lam: +={len(pos_p)} tiles  -={len(neg_p)} tiles  lam={args.lam_poles}"
            )

    elif args.rings == "single":
        print(f"N={lat.N}  R={args.R}  rings=single")
        A_tiles, A_faces = find_toroidal_ring(lat, axis=args.axis)
        if A_tiles is None:
            print("ERROR: no toroidal ring found — lattice too small?")
            sys.exit(1)
        B_tiles = []
        omega_A = init_resonant_loop(
            lat,
            A_tiles,
            A_faces,
            amplitude=args.amplitude,
            phi_only=True,
        )
        ring_idx = set(A_tiles)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        print(f"Ring A: {len(A_tiles)} tiles (single traveling wave)  r≈{ring_r:.2f}")
        print(f"ω_A={omega_A:.4f}")
        if args.lam_poles > 0.0:
            pos_p, neg_p = _pole_tiles(lat.pos, list(ring_idx), args.pole_r, args.axis)
            for i in pos_p + neg_p:
                lat.lam[i] = args.lam_poles
            print(
                f"Pole lam: +={len(pos_p)} tiles  -={len(neg_p)} tiles  lam={args.lam_poles}"
            )

    else:  # dc
        print(f"N={lat.N}  R={args.R}  rings=dc")
        A_tiles, A_faces = find_toroidal_ring(lat, axis=args.axis)
        if A_tiles is None:
            print("ERROR: no toroidal ring found — lattice too small?")
            sys.exit(1)
        B_tiles = []
        N_ring = len(A_tiles)
        for j in range(N_ring):
            i = A_tiles[j]
            nxt = A_tiles[(j + 1) % N_ring]
            k = A_faces[j]
            mk = int(_MIRROR[k])
            lat.phi[i, k] = args.amplitude
            lat.phi[nxt, mk] = -args.amplitude
        ring_idx = set(A_tiles)
        ring_r = float(np.mean([r_all[i] for i in ring_idx]))
        div_ring = sum(sum(lat.phi[i, k] for k in range(14)) for i in ring_idx)
        print(f"Ring A: {len(A_tiles)} tiles (DC circulation)  r≈{ring_r:.2f}")
        print(
            f"amplitude={args.amplitude}  div(phi) on ring = {div_ring:.2e}  (should be ≈0)"
        )
        if args.lam_poles > 0.0:
            pos_p, neg_p = _pole_tiles(lat.pos, list(ring_idx), args.pole_r, args.axis)
            for i in pos_p + neg_p:
                lat.lam[i] = args.lam_poles
            print(
                f"Pole lam: +={len(pos_p)} tiles  -={len(neg_p)} tiles  lam={args.lam_poles}"
            )

    # Unified ring index lists
    A_idx = list(A_tiles)
    B_idx = list(B_tiles)

    # Pole tile indices for rate tracking (from _pole_tiles if fresh IC, else z-axis scan)
    _pos_p, _neg_p = _pole_tiles(lat.pos, list(ring_idx), args.pole_r, args.axis)
    pole_idx = np.array(_pos_p + _neg_p, dtype=int)

    phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
    dt = lat.stable_dt()
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
    bc_flux_h = np.zeros(
        n_snaps
    )  # net u injected by BC per step (averaged over report interval)
    pole_drain_h = np.zeros(n_snaps)  # net u change at pole tiles per step

    snap_steps = sorted({0, args.n_steps // 4, args.n_steps // 2, args.n_steps})
    snap_profiles = {}
    snap_2d = {}

    _bc_acc = 0.0  # accumulated BC flux since last record
    _u_pole_prev = lat.u[pole_idx].sum() if len(pole_idx) else 0.0
    _pole_acc = 0.0  # accumulated pole drain since last record

    def record(si, step):
        nonlocal _bc_acc, _pole_acc, _u_pole_prev
        times[si] = step * dt
        H_hist[si] = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
        phi_rms_h[si] = float(np.sqrt(np.mean(lat.phi**2)))
        lam_ring_h[si] = float(np.mean([lat.lam[i] for i in ring_idx]))
        lam_A_h[si] = float(np.mean([lat.lam[i] for i in A_idx])) if A_idx else 0.0
        lam_B_h[si] = float(np.mean([lat.lam[i] for i in B_idx])) if B_idx else 0.0
        circ_A_h[si] = phi_ring_circulation(lat, A_idx, centre)
        circ_B_h[si] = phi_ring_circulation(lat, B_idx, centre)
        for ki, r in enumerate(R_SCAN):
            circ_hist[si, ki] = phi_toroidal_circulation(lat, centre, r)
        bc_flux_h[si] = _bc_acc / args.report
        pole_drain_h[si] = _pole_acc / args.report
        _bc_acc = 0.0
        _pole_acc = 0.0

    has_B = bool(B_idx)
    r_cols = "  ".join(f"r={r:2d}" for r in R_SCAN)
    lam_hdr = "  lam_A   lam_B" if has_B else "  lam_A"
    circ_hdr = "  circ_A     circ_B" if has_B else "  circ_A"
    print(
        f"\n{'step':>6}  {'H':>10}  {'phi_rms':>10}  {'lam_ring':>8}{lam_hdr}{circ_hdr}"
        f"  {'bc_inj/step':>12}  {'pole_Δu/step':>13}  {r_cols}"
    )
    print("  " + "-" * (100 + 8 * len(R_SCAN)))

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
                f"  {bc_flux_h[si]:+12.4e}  {pole_drain_h[si]:+13.4e}  {cs}",
                flush=True,
            )
            si += 1

        if step in snap_steps:
            up = radial_profile(lat.pos, lat.u, centre, r_edges)
            lp = radial_profile(lat.pos, lat.lam, centre, r_edges)
            snap_profiles[step] = (up, lp)

        if step == 0 or step == args.n_steps:
            snap_2d[step] = {
                "phi": lat.phi.copy(),
                "lam": lat.lam.copy(),
                "u": lat.u.copy(),
            }

        if step < args.n_steps:
            u_pole_before = lat.u[pole_idx].sum() if len(pole_idx) else 0.0
            lat.step(dt, n_threads=args.n_threads)
            _bc_acc += lat.bc_flux()
            _pole_acc += lat.u[pole_idx].sum() - u_pole_before
            if args.cross_scale > 0.0:
                lat.step_cross(dt, scale=args.cross_scale)
            if args.lam_mode == "dynamic":
                lat.step_lam(dt, scale=args.lam_scale)

    # -----------------------------------------------------------------------
    # Save snapshot
    # -----------------------------------------------------------------------
    if args.save_snapshot:
        save_kwargs = dict(
            phi=lat.phi,
            lam=lat.lam,
            u=lat.u,
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
        f"DST spheromak decay — cross={args.cross_scale}  lam_mode={args.lam_mode}"
        f"  lam_poles={args.lam_poles}  u_bg={args.u_bg}",
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
    ax_rc = fig.add_subplot(gs[2, 0])  # per-ring circulation
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

    # Per-ring circulation panel
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
        z = lat.pos[:, 2]
        ztol = (z.max() - z.min()) / 8.0
        zmid = (z.max() + z.min()) / 2.0
        mask = np.abs(z - zmid) < ztol
        if clip_r is not None:
            r2d = np.linalg.norm(lat.pos[:, :2], axis=1)
            mask &= r2d < clip_r

        xv = lat.pos[mask, 0]
        yv = lat.pos[mask, 1]

        nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
        pos_j = lat.pos[nb_safe]
        diff = pos_j - lat.pos[:, np.newaxis, :]
        d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
        n_hat = diff / np.where(d_len > 0, d_len, 1.0)

        xi, yi = lat.pos[:, 0, np.newaxis], lat.pos[:, 1, np.newaxis]
        rho_i = np.sqrt(xi**2 + yi**2)
        rho_s = np.where(rho_i > 1e-10, rho_i, 1.0)
        pha_x = -yi / rho_s
        pha_y = xi / rho_s

        proj = n_hat[..., 0] * pha_x + n_hat[..., 1] * pha_y
        valid = lat._valid
        phi_tor = (phi_arr * proj * valid).sum(axis=1) / np.maximum(
            valid.sum(axis=1), 1
        )

        return xv, yv, phi_tor[mask], lam_arr[mask], u_arr[mask]

    clip = args.R * 0.75
    nr = 150
    xi_g = np.linspace(-clip, clip, nr)
    yi_g = np.linspace(-clip, clip, nr)
    Xi, Yi = np.meshgrid(xi_g, yi_g)

    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig2.suptitle(
        f"DST spheromak — before / after  (cross={args.cross_scale}  t_final={args.n_steps*dt:.1f})",
        fontsize=11,
    )

    rows = {
        "phi_tor": (0, "φ toroidal (midplane)", "RdBu_r"),
        "lam": (1, "λ (winding)", "viridis"),
    }
    t_labs = {0: "before  t=0", args.n_steps: f"after  t={args.n_steps*dt:.1f}"}

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
            theta = np.linspace(0, 2 * np.pi, 64)
            ax.plot(
                ring_r * np.cos(theta), ring_r * np.sin(theta), "w--", lw=0.7, alpha=0.6
            )

    xv0, yv0, pt0, lam0, _ = midplane_slice(
        snap_2d[0]["phi"], snap_2d[0]["lam"], snap_2d[0]["u"], clip_r=clip
    )
    xvN, yvN, ptN, lamN, _ = midplane_slice(
        snap_2d[args.n_steps]["phi"],
        snap_2d[args.n_steps]["lam"],
        snap_2d[args.n_steps]["u"],
        clip_r=clip,
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
