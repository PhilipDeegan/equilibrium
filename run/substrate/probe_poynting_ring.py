"""
probe_poynting_ring.py — Poynting-flux visualisation for a single oscillating
ring "proton" (traveling-wave phi on an 8-tile toroidal ring), checking for
the classic oscillating-dipole Poynting-vector pattern: near-field
circulating loops transitioning to far-field radial outflow
(https://en.wikipedia.org/wiki/Poynting_vector).

E-field construction (important, see project_electron_closure_hall_analogue
memory): defining E := -grad(u) - dphi/dt the standard way gives IDENTICALLY
ZERO under DST's free wave step alone -- verified numerically: dphi[i,k]/dt
under lat.step() matches -W[k]*(u[j]-u[i]) almost exactly (W_HEX=3, W_SQ=1),
so the two terms of E cancel by construction, not approximately. A real,
non-tautological E only appears from a term that modifies phi OUTSIDE the
u-coupled wave equation -- step_cross (already active in
test_spheromak_decay.py's default run) is exactly that. So here:

    E_vec[i] := -(dphi_cross[i]) / dt

where dphi_cross is the phi increment from step_cross ALONE (bracketed
before/after that specific call, isolated from the free wave step), reduced
to a per-cell 3-vector the same way _occ_vec/A_vec are elsewhere. B comes
from the existing curl(phi) reconstruction (_b_vec, reused as-is -- it's
generic, not tied to the occ-field MHD kernels). S = E x B.

lam is a proton/matter-density proxy (project_lam_dual_role memory), not
something the radiated flux itself should carry -- lam is set ONLY on the
ring tiles and never touched again (no step_lam call at all), so the
surrounding radiation zone stays at lam=0 (vacuum) for the whole run.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/probe_poynting_ring.py
    PYTHONPATH=$PWD python3 run/substrate/probe_poynting_ring.py \\
        --R 40 --n-steps 6000 --n-snapshots 6 --cross-scale 0.01
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_resonant_loop,
    winding_number,
)
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.mhd import _SQ_AXIS_PAIRS, _b_vec
from run.substrate.cross_v2_uniform import M0_VOID, CrossV2

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--R", type=float, default=100.0)
parser.add_argument(
    "--mode",
    choices=("minimal", "square"),
    default="minimal",
    help="ring construction: minimal = fixed 8-tile loop (original probe, "
    "~2.8 cells across -- inside the BCC dispersion anisotropy band). "
    "square = scaled loop via --hops (equilibrium.substrate.flux_tube."
    "find_toroidal_ring) -- lets the ring clear that band at larger --hops.",
)
parser.add_argument(
    "--hops",
    type=int,
    default=1,
    help="--mode square only: sq-face hops per half-leg; loop size = 8*hops "
    "tiles, physical radius grows ~linearly with hops. Ignored for minimal.",
)
parser.add_argument(
    "--snapshot-out",
    type=str,
    default=None,
    help="if set, save full lattice state (u, phi, lam) to "
    "<snapshot-out>_start.npz and <snapshot-out>_end.npz for later "
    "cross-run diffing (e.g. across --hops values).",
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
    help="v1: continuous always-fires step_cross (original probe). "
    "barrier: quantized all-or-nothing gate (doc/dst_next.md sec 8/CrossV2) "
    "-- tests whether a causal/discrete crossing changes the near-field-vortex "
    "-> far-field-radial outcome vs the inconclusive v1 sweep.",
)
parser.add_argument(
    "--m0",
    type=float,
    default=0.3,
    help="barrier gate threshold (pooled per-cell drive Sum W*max(-phi,0)). "
    "Order-unity, comparable to the ring's own driven phi amplitude, so "
    "crossing fires near the source but not in the decayed far field. "
    f"M0_VOID placeholder ({M0_VOID:.1e}) is too small to gate anything -- "
    "only meaningful with a real knot-radius calibration (see project_step_cross_v2 "
    "/ doc/dst_next.md sec 8), not used as the default here.",
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
parser.add_argument("--plot-out", type=str, default="poynting_ring.png")


def face_vec(face_arr_row):
    """Reduce a 14-component face row to a 3-vector via the sq-face pairs
    (same reduction _occ_vec/A_vec use elsewhere)."""
    return np.array([face_arr_row[fp] - face_arr_row[fm] for fp, fm in _SQ_AXIS_PAIRS])


def main():
    args = parser.parse_args()
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    ring_kwargs = {"hops": args.hops} if args.mode == "square" else {}
    ring_tiles, ring_faces = find_toroidal_ring(
        lat, axis=args.axis, mode=args.mode, **ring_kwargs
    )
    if ring_tiles is None:
        print("ERROR: no toroidal ring found -- lattice too small?")
        sys.exit(1)
    w = winding_number(lat, ring_tiles, axis=args.axis)
    if abs(abs(w) - 1.0) > 1e-6:
        print(f"ERROR: ring winding_number={w:.6f}, expected +-1 -- bad geometry")
        sys.exit(1)
    omega = init_resonant_loop(
        lat, ring_tiles, ring_faces, amplitude=args.amplitude, phi_only=True
    )
    ring_pos = lat.pos[ring_tiles]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    # lam only on ring tiles -- never touched again (no step_lam call anywhere
    # in this script), so the radiation zone stays vacuum (lam=0) throughout.
    for i in ring_tiles:
        lat.lam[i] = args.lam_ring

    dt = lat.stable_dt()
    period = 2.0 * np.pi / omega if omega > 0 else None
    print(
        f"N={lat.N}  R={args.R}  mode={args.mode}  hops={args.hops}  "
        f"n_tiles={len(ring_tiles)}  ring_r={ring_r:.3f}  winding={w:+.3f}  "
        f"omega={omega:.4f}  period={period:.2f}  dt={dt:.4e}",
        flush=True,
    )

    if args.snapshot_out:
        np.savez(
            f"{args.snapshot_out}_start.npz",
            u=lat.u.copy(),
            phi=lat.phi.copy(),
            lam=lat.lam.copy(),
            ring_tiles=np.array(ring_tiles),
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
    # cross_op is kept only for the gate_field() fire/raw diagnostics below --
    # the actual step now goes through the native SubstrateLattice.step_cross
    # gate="barrier" port (equilibrium/substrate/lattice.py), verified to
    # match this prototype bit-for-bit.
    cross_op = (
        CrossV2(lat, gate=args.gate, m0=args.m0) if args.gate == "barrier" else None
    )
    cross_kwargs = {"gate": "barrier", "m0": args.m0} if args.gate == "barrier" else {}

    def do_cross():
        if args.cross_scale <= 0.0:
            return
        lat.step_cross(dt, scale=args.cross_scale, **cross_kwargs)

    # Slice cells for the quiver plot: near the axial plane through the ring
    # centre, within slice_r_max of it.
    pos = lat.pos
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
            if cross_op is not None:
                fire, raw = cross_op.gate_field(lat.phi)
                fired_cell = fire.any(axis=1)[
                    slice_idx
                ]  # any face firing, per slice cell
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
                raw_cell = raw[:, 0][
                    slice_idx
                ]  # pooled drive is broadcast; col 0 suffices
                r_slice = r_from_centre[slice_idx]
                edges = np.linspace(0.0, args.slice_r_max, 9)
                drive_by_bin = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (r_slice >= lo) & (r_slice < hi)
                    drive_by_bin.append(float(raw_cell[m].max()) if m.any() else 0.0)
            phi_before_cross = lat.phi[slice_idx].copy()
            do_cross()
            phi_after_cross = lat.phi[slice_idx].copy()
            dphi_cross = phi_after_cross - phi_before_cross

            E3 = np.zeros((len(slice_idx), 3))
            B3 = np.zeros((len(slice_idx), 3))
            for m, i in enumerate(slice_idx):
                E3[m] = -face_vec(dphi_cross[m]) / dt
                bv = _b_vec(lat.phi, lat.nb, int(i))
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
                lat.step(dt)
        else:
            lat.step(dt)
            do_cross()

    if args.snapshot_out:
        np.savez(
            f"{args.snapshot_out}_end.npz",
            u=lat.u.copy(),
            phi=lat.phi.copy(),
            lam=lat.lam.copy(),
            ring_tiles=np.array(ring_tiles),
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
        f"Poynting flux S=ExB around an oscillating ring "
        f"(axial slice)  ring_r={ring_r:.2f}  cross_scale={args.cross_scale}",
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
        # Fixed, explicit scale (NOT None/auto) -- quiver's autoscale derives
        # from the MEAN vector length across all points; early snapshots have
        # very few nonzero (unit-length) vectors among thousands of exact
        # zeros, which drags the mean down and makes autoscale wildly
        # overcompensate, stretching the few real vectors into lines spanning
        # the whole plot. A fixed scale keeps all panels comparable too.
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
