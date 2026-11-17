"""
probe_poynting_ring.py (adaptive) — Poynting-flux visualisation for a single
oscillating ring "proton" (traveling-wave phi on a toroidal ring), on the
AdaptiveLattice C++ backend. Mirrors run/substrate/probe_poynting_ring.py
(uniform SubstrateLattice); see that file's docstring for the full E/S
construction rationale (E := -dphi_cross/dt, isolated from the free wave
step; B from the existing curl(phi) reconstruction; S = E x B).

Unlike the uniform probe, this one defaults to mode="square" with hops=4
(32-tile ring, ring_r a few cells wider than the minimal 8-tile loop) rather
than mode="minimal" -- the 8-tile loop is only ~2.8 cells across, inside the
BCC dispersion anisotropy band (project_bcc_dispersion_anisotropy /
project_ring_scale_convergence): any physics conclusion drawn from it is
confound-suspect, so this adaptive port should not default to it even though
the uniform script still does (for backward compatibility with its own
existing runs).

Usage:
    PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_poynting_ring.py
    PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_poynting_ring.py \\
        --R 40 --hops 6 --n-steps 3000 --n-snapshots 6 --cross-scale 0.01
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.mhd import _SQ_AXIS_PAIRS, _b_vec
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    _W_FACE,
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--R", type=float, default=100.0)
parser.add_argument(
    "--mode",
    choices=("minimal", "square"),
    default="square",
    help="ring construction: minimal = fixed 8-tile loop (~2.8 cells across "
    "-- inside the BCC dispersion anisotropy band, avoid for real physics "
    "conclusions). square (default here) = scaled loop via --hops "
    "(tst.substrate.adaptive.adaptive_flux_tube.find_toroidal_ring).",
)
parser.add_argument(
    "--hops",
    type=int,
    default=4,
    help="--mode square only: sq-face hops per half-leg; loop size = 8*hops "
    "tiles, physical radius grows ~linearly with hops. Ignored for minimal.",
)
parser.add_argument(
    "--snapshot-out",
    type=str,
    default=None,
    help="if set, save full lattice state (u, phi, lam) to "
    "<snapshot-out>_start.npz and <snapshot-out>_end.npz for later "
    "cross-run diffing (e.g. across --hops values). phi is saved in the "
    "adaptive (14, N) leaf-major layout, not the uniform (N, 14) layout.",
)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--lam-ring",
    type=float,
    default=1.0,
    help="lam value on ring tiles only (0=vacuum, matter proxy)",
)
parser.add_argument(
    "--cross-scale",
    type=float,
    default=0.01,
    help="step_cross scale -- the source of a genuine (non-tautological) E",
)
parser.add_argument(
    "--gate",
    choices=("v1", "barrier"),
    default="barrier",
    help="v1: continuous always-fires step_cross. barrier: quantized "
    "all-or-nothing gate (doc/dst_next.md sec 8), now ported to "
    "AdaptiveLattice::step_cross (inc/eq/core/data/lattice/adaptive_lattice"
    ".hpp, _step_cross_local/_step_cross_mirror) -- tests whether a causal"
    "/discrete crossing changes the near-field-vortex -> far-field-radial "
    "outcome vs the inconclusive v1 sweep.",
)
parser.add_argument(
    "--m0",
    type=float,
    default=0.3,
    help="barrier gate threshold (pooled per-cell drive Sum W*max(-phi,0)). "
    "Order-unity, comparable to the ring's own driven phi amplitude, so "
    "crossing fires near the source but not in the decayed far field.",
)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=1500)
parser.add_argument("--n-snapshots", type=int, default=6)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument(
    "--slice-half-width",
    type=float,
    default=1.0,
    help="half-width (cells) of the midplane slice used for the quiver plot",
)
parser.add_argument("--slice-r-max", type=float, default=20.0)
parser.add_argument("--plot-out", type=str, default="poynting_ring_adaptive.png")
parser.add_argument(
    "--threads",
    type=int,
    default=10,
    help="thread count passed to AdaptiveLattice.set_n_threads",
)


def face_vec(face_arr_row):
    """Reduce a 14-component face row to a 3-vector via the sq-face pairs
    (same reduction _occ_vec/A_vec use elsewhere)."""
    return np.array([face_arr_row[fp] - face_arr_row[fm] for fp, fm in _SQ_AXIS_PAIRS])


def gate_field_barrier(phi_T, nb_j, m0):
    """Per-cell (drive > m0) fire decision + raw pooled drive -- same formula
    as CrossV2.gate_field's barrier branch / step_cross(gate="barrier"), but
    inlined here (no adaptive CrossV2 prototype exists) since only the
    diagnostic values are needed, not the step itself (that goes through the
    production AdaptiveLattice.step_cross(gate="barrier") call directly)."""
    valid = nb_j >= 0
    phiv = np.where(valid, phi_T, 0.0)
    drive = (_W_FACE[None, :] * np.maximum(-phiv, 0.0)).sum(axis=1)  # (N,)
    return drive > m0, drive


def main():
    args = parser.parse_args()
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    ring_kwargs = {"hops": args.hops} if args.mode == "square" else {}
    ring_js, ring_faces = find_toroidal_ring(
        cl, axis=args.axis, mode=args.mode, **ring_kwargs
    )
    if ring_js is None:
        print("ERROR: no toroidal ring found -- lattice too small?")
        sys.exit(1)
    w = winding_number(cl, ring_js, axis=args.axis)
    if abs(abs(w) - 1.0) > 1e-6:
        print(f"ERROR: ring winding_number={w:.6f}, expected +-1 -- bad geometry")
        sys.exit(1)
    omega = init_toroidal_winding(
        cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True
    )
    pos = np.array(cl.leaf_pos)
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    # lam only on ring tiles -- never touched again (no step_lam call anywhere
    # in this script), so the radiation zone stays vacuum (lam=0) throughout.
    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    dt = compute_dt(cl)
    period = 2.0 * np.pi / omega if omega > 0 else None
    print(
        f"N={cl.n_leaves}  R={args.R}  mode={args.mode}  hops={args.hops}  "
        f"n_tiles={len(ring_js)}  ring_r={ring_r:.3f}  winding={w:+.3f}  "
        f"omega={omega:.4f}  period={period:.2f}  dt={dt:.4e}",
        flush=True,
    )

    if args.snapshot_out:
        np.savez(
            f"{args.snapshot_out}_start.npz",
            u=np.array(cl.leaf_u),
            phi=np.array(cl.leaf_phi),
            lam=np.array(cl.leaf_lam),
            ring_js=np.array(ring_js),
            ring_faces=np.array(ring_faces),
            ring_centre=ring_centre,
            ring_r=ring_r,
            omega=omega,
            R=args.R,
            mode=args.mode,
            hops=args.hops,
            step=0,
        )
        print(f"[snapshot] {args.snapshot_out}_start.npz", flush=True)
    print(
        f"cross_scale={args.cross_scale}  lam_ring={args.lam_ring}  "
        f"gate={args.gate}" + (f"  m0={args.m0}" if args.gate == "barrier" else ""),
        flush=True,
    )
    cross_kwargs = {"gate": "barrier", "m0": args.m0} if args.gate == "barrier" else {}

    def do_cross():
        if args.cross_scale <= 0.0:
            return
        alat.step_cross(0, dt, args.cross_scale, **cross_kwargs)

    def do_step():
        alat.step(0, dt)
        alat.step_sponge(0)

    # Leaf-local neighbour table -- fixed for the whole run (no AMR
    # split/coarsen calls anywhere in this probe), safe to build once.
    nb_j = leaf_nb_j(cl)

    # Slice cells for the quiver plot: near the axial plane through the ring
    # centre, within slice_r_max of it.
    r_from_centre = np.linalg.norm(pos - ring_centre, axis=1)
    in_plane = np.abs(pos[:, perp[1]] - ring_centre[perp[1]]) < args.slice_half_width
    in_radius = r_from_centre < args.slice_r_max
    slice_idx = np.where(in_plane & in_radius)[0]
    print(f"slice cells: {len(slice_idx)}", flush=True)

    snap_steps = sorted(
        set(
            int(round(k * args.n_steps / (args.n_snapshots - 1)))
            for k in range(args.n_snapshots)
        )
    )
    snapshots = []  # list of dict(step, x, z, Ex, Ez, Bx, Bz, Sx, Sz, Smag)

    for step in range(args.n_steps + 1):
        if step in snap_steps:
            # Bracket step_cross alone: capture phi right before/after it.
            fire_frac_far = None
            drive_by_bin = None
            if args.gate == "barrier":
                phi_T_now = np.array(cl.leaf_phi).T  # (N, 14)
                fire, raw = gate_field_barrier(phi_T_now, nb_j, args.m0)
                fired_cell = fire[slice_idx]
                far_mask = r_from_centre[slice_idx] > 2.0 * ring_r
                if far_mask.any():
                    fire_frac_far = float(fired_cell[far_mask].mean())
                fire_frac_near = (
                    float(fired_cell[~far_mask].mean()) if (~far_mask).any() else 0.0
                )

                # Raw (ungated) pooled drive vs radius -- since nothing fires
                # beyond 2*ring_r at this m0, that region's phi trajectory is
                # pure free-wave dynamics, so this max-drive-per-radius-bin
                # profile tells us directly what m0 WOULD need to be for
                # crossing to reach that radius, without rerunning at every
                # candidate m0.
                raw_cell = raw[slice_idx]
                r_slice = r_from_centre[slice_idx]
                edges = np.linspace(0.0, args.slice_r_max, 9)
                drive_by_bin = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (r_slice >= lo) & (r_slice < hi)
                    drive_by_bin.append(float(raw_cell[m].max()) if m.any() else 0.0)

            phi_before_cross = (
                np.array(cl.leaf_phi).T[slice_idx].copy()
            )  # (n_slice, 14)
            do_cross()
            phi_after_cross = np.array(cl.leaf_phi).T[slice_idx].copy()
            dphi_cross = phi_after_cross - phi_before_cross

            phi_T = np.array(cl.leaf_phi).T  # (N, 14) -- for _b_vec below

            E3 = np.zeros((len(slice_idx), 3))
            B3 = np.zeros((len(slice_idx), 3))
            for m, i in enumerate(slice_idx):
                E3[m] = -face_vec(dphi_cross[m]) / dt
                bv = _b_vec(phi_T, nb_j, int(i))
                if bv is not None:
                    B3[m] = bv
            S3 = np.cross(E3, B3)

            snapshots.append(
                dict(
                    step=step,
                    t=step * dt,
                    x=pos[slice_idx, perp[0]] - ring_centre[perp[0]],
                    z=pos[slice_idx, ax_idx] - ring_centre[ax_idx],
                    Ex=E3[:, perp[0]],
                    Ez=E3[:, ax_idx],
                    Bx=B3[:, perp[0]],
                    Bz=B3[:, ax_idx],
                    Sx=S3[:, perp[0]],
                    Sz=S3[:, ax_idx],
                    Smag=np.linalg.norm(S3, axis=1),
                    Emag=np.linalg.norm(E3, axis=1),
                    Bmag=np.linalg.norm(B3, axis=1),
                )
            )
            fire_str = (
                f"  fire_frac(r<2ring_r)={fire_frac_near:.3f}"
                f" fire_frac(r>2ring_r)={fire_frac_far:.3f}"
                if fire_frac_far is not None
                else ""
            )
            print(
                f"  step={step:6d}  t={step*dt:8.2f}  "
                f"|E|_max={np.max(np.linalg.norm(E3,axis=1)):.3e}  "
                f"|B|_max={np.max(np.linalg.norm(B3,axis=1)):.3e}  "
                f"|S|_max={np.max(np.linalg.norm(S3,axis=1)):.3e}" + fire_str,
                flush=True,
            )
            if drive_by_bin is not None:
                edges = np.linspace(0.0, args.slice_r_max, 9)
                bins_str = "  ".join(
                    f"[{lo:4.1f},{hi:4.1f})={d:.2e}"
                    for lo, hi, d in zip(edges[:-1], edges[1:], drive_by_bin)
                )
                print(f"    max_drive_by_r: {bins_str}", flush=True)

            if step < args.n_steps:
                do_step()
        else:
            do_step()
            do_cross()

    if args.snapshot_out:
        np.savez(
            f"{args.snapshot_out}_end.npz",
            u=np.array(cl.leaf_u),
            phi=np.array(cl.leaf_phi),
            lam=np.array(cl.leaf_lam),
            ring_js=np.array(ring_js),
            ring_faces=np.array(ring_faces),
            ring_centre=ring_centre,
            ring_r=ring_r,
            omega=omega,
            R=args.R,
            mode=args.mode,
            hops=args.hops,
            step=args.n_steps,
        )
        print(f"[snapshot] {args.snapshot_out}_end.npz", flush=True)

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(
            f"[plot] SKIPPED -- matplotlib import failed ({e}); "
            f"npz snapshots/printed diagnostics above are unaffected.",
            flush=True,
        )
        return

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"Poynting flux S=ExB around an oscillating ring (adaptive backend, "
        f"axial slice)  ring_r={ring_r:.2f}  cross_scale={args.cross_scale}",
        fontsize=10,
    )

    for ax, snap in zip(axes, snapshots):
        x, z = snap["x"], snap["z"]
        sx, sz, smag = snap["Sx"], snap["Sz"], snap["Smag"]
        finite = np.isfinite(sx) & np.isfinite(sz) & np.isfinite(smag)
        x, z, sx, sz, smag = x[finite], z[finite], sx[finite], sz[finite], smag[finite]

        # Normalise arrow directions to unit length -- Smag spans many orders
        # of magnitude near vs far from the ring, and quiver's literal
        # (scale_units="xy") length would let a handful of near-ring outliers
        # stretch arrows across the whole plot. Colour (not length) carries
        # magnitude here.
        safe_mag = np.where(smag > 1e-30, smag, 1.0)
        ux, uz = sx / safe_mag, sz / safe_mag
        vmax = np.percentile(smag, 95) if smag.max() > 0 else 1.0
        ax.quiver(
            x,
            z,
            ux,
            uz,
            smag,
            cmap="inferno",
            clim=(0, vmax),
            pivot="mid",
            width=0.004,
            scale=40.0,
            scale_units="width",
        )
        theta = np.linspace(0, 2 * np.pi, 64)
        ax.plot(ring_r * np.cos(theta), np.zeros_like(theta), "c--", lw=0.8, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"t={snap['t']:.1f} (step {snap['step']})", fontsize=9)
        ax.set_xlabel(f"{['x','y','z'][perp[0]]}")
        if snap is snapshots[0]:
            ax.set_ylabel(f"{args.axis}")

    plt.tight_layout()
    fig.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"\n[plot] {args.plot_out}", flush=True)


if __name__ == "__main__":
    main()
