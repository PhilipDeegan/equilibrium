"""
Follow-up to probe_poynting_time_evolution.py: that run's coarse
log_every=50 (dt=5.128e-3, so 0.256 physical-time between samples) turned
out to be too coarse to resolve the real late-time oscillation in |S| at
fixed radius -- the detected "peaks" were suspiciously close to small
integer multiples of the logging interval itself (aliasing), not a
trustworthy period estimate.

The mesh itself is a single UNIFORM-resolution build_bcc_sphere -- no AMR,
no coarse/fine mesh regions anywhere. Every "coarse"/"fine" distinction
below is purely about LOGGING CADENCE (how often a trace row gets computed
over time), never about spatial resolution.

This script:
  1. Computes a trace row every --coarse-log-every steps (default 50,
     ~0.26 physical time apart) through the initial wavefront-transit phase
     (matches the earlier run exactly, for cross-checking).
  2. Switches to a FINER logging cadence (--fine-log-every, default 2,
     ~0.0103 physical time apart -- ~25x finer) for a burst starting once
     the wavefront has passed every measurement shell, to actually resolve
     the oscillation period without aliasing.
  3. At each of several target physical times (--plot-times), renders and
     saves an XZ cross-section PNG of |S| directly, inline as the run
     progresses -- so a killed run keeps every frame already written up to
     that point. No raw array is ever persisted for this -- only the
     rendered PNG -- keeping one 43MB (N=5.4M) raw array per plot time,
     ~100 of them over a full t=50 run, would be an unbounded multi-GB disk
     leak with no purpose once it's been plotted (an earlier version of
     this script did exactly that, since fixed).
     trace.npy and the lattice state (--final-state-out) are each written
     ONCE, at the very end -- a killed run loses the in-progress trace, but
     resumes cleanly from the last --final-state-out checkpoint via
     --restart-from, so mid-run trace flushing isn't needed for resilience.

Perf fix (2026-09-09, see doc/mems for the R100/N=5.4M run this was found
on): `local_energy_flux_vector`'s `face_unit_vectors(pos, nb)` term was being
rebuilt (several GB of (N,14,3) temporaries, single-threaded) on every
logged step. The per-step flux is now computed ONLY over the union of the
measurement shells (a tiny fraction of N, independent of R) rather than the
whole domain -- the original slowdown was the full-domain recompute
happening on EVERY fine-logged step, not the occasional render.

Memory fix (2026-09-09, same day, found at R=150/N=18.4M): a full-domain
`face_unit_vectors`/`local_energy_flux_vector` call is NEVER done anymore,
not even transiently at plot time -- its (N,14,3) internal temporaries
(5.75GB at N=18.4M, several times that counting every intermediate) OOM'd
the run even though the RESULT was immediately sliced down and discarded.
Both the shell-trace values and the rendered-frame |S| are now each computed
directly on their own small row subset (`face_unit_vectors(..., pos_own=...)`
keeps every intermediate shaped by the subset, never by N) -- see
equilibrium/substrate/energy_flux.py's pos_own/u_own docs.
"""

import argparse
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import face_unit_vectors, local_energy_flux_vector
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--mode", type=str, default="square", choices=["square", "octagon"])
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=8000)  # matches the earlier full run's duration (t~41)
parser.add_argument("--coarse-log-every", type=int, default=50)
parser.add_argument("--fine-start-t", type=float, default=6.0, help="physical time to switch to fine logging")
parser.add_argument("--fine-log-every", type=int, default=2)
parser.add_argument("--fine-duration-t", type=float, default=12.0, help="physical time span of the fine-logging burst")
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--sector-half-angle", type=float, default=20.0)
parser.add_argument("--shell-width-frac", type=float, default=0.15)
parser.add_argument("--shells", type=str, default="8,12,16,20,24")
parser.add_argument(
    "--plot-every-t",
    type=float,
    default=0.5,
    help="physical-time cadence for rendering an XZ |S| frame -- applies over whatever "
    "[start_step, start_step+n_steps) range this invocation covers, not a fixed absolute "
    "window, so a --restart-from leg keeps the same cadence without needing to know the "
    "global time range in advance. Set to 0 to disable.",
)
parser.add_argument(
    "--plot-times",
    type=str,
    default=None,
    help="optional comma-separated explicit physical times to render a frame at, in addition "
    "to the --plot-every-t cadence; the final step is always plotted too",
)
parser.add_argument("--out-prefix", type=str, default="exp/2026/09/08/12_15_poynting_time_evolution/fine")
parser.add_argument("--slab-half-thickness", type=float, default=1.5, help="|y| < this, cell units, for the XZ cross-section plot slab")
parser.add_argument(
    "--final-state-out",
    type=str,
    default=None,
    help="if set, save the final AdaptiveLattice state -- the exact step is embedded in the "
    "filename automatically (e.g. 'foo.h5' -> 'foo_step0003900.h5'), no sidecar file needed",
)
parser.add_argument(
    "--restart-from", type=str, default=None, help="resume from an HDF5 state saved via --final-state-out"
)
parser.add_argument(
    "--start-step",
    type=int,
    default=None,
    help="global step index the restart state represents (i.e. steps already completed); "
    "auto-parsed from the --restart-from filename's _stepNNNNNNN suffix if omitted "
    "(needed explicitly only for checkpoints saved before this naming existed)",
)


def _apply_sponge_bc(cl, args):
    # AdaptiveLattice.load() restores the mesh + u/phi/lam fields exactly, but
    # NOT the functional quadratic_sponge BC -- .save() only writes a trivial
    # placeholder BC record (verified 2026-09-09: type=0, no per-cell damp
    # array). Same class of gotcha as the AMR "rebuild BC whenever the leaf
    # set changes" fix -- called unconditionally (fresh build or reload) for
    # that reason, not just on the fresh-build path.
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(_lat.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner))


def _find_ring(cl, args):
    ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode=args.mode, hops=args.hops)
    if ring_js is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)
    return ring_js, ring_faces


def _ring_geometry(cl, ring_js):
    pos = np.array(cl.leaf_pos)
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())
    return ring_centre, ring_r


def build_fresh_ic(args):
    """Fresh sphere + seeded toroidal phi winding + lam ring, t=0."""
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)
    _apply_sponge_bc(cl, args)

    ring_js, ring_faces = _find_ring(cl, args)
    omega = init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True)
    ring_centre, ring_r = _ring_geometry(cl, ring_js)

    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    dt = compute_dt(cl)
    print(
        f"N={cl.n_leaves} R={args.R} hops={args.hops} ring_r={ring_r:.3f} omega={omega:.4f} dt={dt:.4e}",
        flush=True,
    )
    return alat, cl, ring_centre, ring_r, dt, 0


def _read_start_step(args):
    if args.start_step is not None:
        return args.start_step
    m = re.search(r"_step(\d+)\.h5$", args.restart_from)
    if not m:
        print(
            f"ERROR: couldn't parse a _stepNNNNNNN.h5 suffix out of {args.restart_from!r} and "
            "--start-step not given -- pass --start-step explicitly for checkpoints saved before "
            "this naming existed",
            file=sys.stderr,
        )
        sys.exit(1)
    return int(m.group(1))


def reload_ic(args):
    """Resume from a --final-state-out checkpoint -- mesh/u/phi/lam restored
    as-is, BC rebuilt (see _apply_sponge_bc), ring geometry re-derived
    (topology-only, doesn't touch fields), start_step parsed straight out of
    the checkpoint's own filename (or --start-step override) -- no sidecar
    file needed."""
    alat = _lat.AdaptiveLattice.load(args.restart_from)
    alat.set_n_threads(args.threads)
    cl = alat[0]
    _apply_sponge_bc(cl, args)

    ring_js, ring_faces = _find_ring(cl, args)
    ring_centre, ring_r = _ring_geometry(cl, ring_js)
    start_step = _read_start_step(args)

    dt = compute_dt(cl)
    print(
        f"RESUMED from {args.restart_from} at start_step={start_step} (t={start_step * dt:.3f})  "
        f"N={cl.n_leaves} R={args.R} hops={args.hops} ring_r={ring_r:.3f} dt={dt:.4e}",
        flush=True,
    )
    return alat, cl, ring_centre, ring_r, dt, start_step


def main():
    args = parser.parse_args()
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]
    shells = [float(s) for s in args.shells.split(",")]
    explicit_plot_times = sorted(float(s) for s in args.plot_times.split(",")) if args.plot_times else []

    if args.restart_from:
        alat, cl, ring_centre, ring_r, dt, start_step = reload_ic(args)
    else:
        alat, cl, ring_centre, ring_r, dt, start_step = build_fresh_ic(args)
    pos = np.array(cl.leaf_pos)

    # cadence-based plot times, computed over THIS invocation's own
    # [start_step, start_step+n_steps) range -- not a fixed absolute window --
    # so a --restart-from leg keeps plotting every --plot-every-t regardless of
    # how far along the global timeline it starts (see 2026-09-10 bug: a fixed
    # 0.5-50.0 default list silently stopped plotting past t=50)
    cadence_plot_times = []
    if args.plot_every_t and args.plot_every_t > 0:
        import math

        t_start = start_step * dt
        t_end = (start_step + args.n_steps) * dt
        first = math.ceil(t_start / args.plot_every_t) * args.plot_every_t
        cadence_plot_times = list(np.arange(first, t_end + 1e-9, args.plot_every_t))
    plot_times = sorted(set(round(t_, 6) for t_ in cadence_plot_times + explicit_plot_times))
    print(f"shells={shells} plot_times={plot_times}", flush=True)

    nb_j = leaf_nb_j(cl)

    rel = pos - ring_centre
    ax_comp = rel[:, ax_idx]
    perp_comp = np.linalg.norm(rel[:, perp], axis=1)
    theta = np.degrees(np.arctan2(perp_comp, np.abs(ax_comp)))
    half = args.sector_half_angle
    equatorial = theta > (90.0 - half)
    r_from_centre = np.linalg.norm(rel, axis=1)

    shell_masks = [
        equatorial & (r_from_centre >= r0 - args.shell_width_frac * r0) & (r_from_centre <= r0 + args.shell_width_frac * r0)
        for r0 in shells
    ]
    # union of the (tiny) shell masks -- the off-plot-step trace only ever needs
    # these cells, not the full N-cell domain
    union_mask = np.zeros(pos.shape[0], dtype=bool)
    for m in shell_masks:
        union_mask |= m
    union_idx = np.where(union_mask)[0]
    pos_u, nb_u = pos[union_idx], nb_j[union_idx]
    # pos_own=pos_u keeps every internal temporary shaped (union_size,14,3) --
    # NEVER (N,14,3). A full-domain face_unit_vectors call, even one whose
    # result is immediately sliced and discarded, OOM'd at N=18.4M (2026-09-09)
    n_hat_u = face_unit_vectors(pos, nb_u, pos_own=pos_u)
    shell_sub_masks = [m[union_idx] for m in shell_masks]
    print(f"shell union: {union_idx.size} / {pos.shape[0]} cells", flush=True)

    # slab (|y|<slab_half_thickness) for the rendered XZ cross-section -- same
    # "only ever touch a small row subset" treatment as the shell union above
    slab = np.abs(pos[:, 1]) < args.slab_half_thickness
    slab_idx = np.where(slab)[0]
    pos_slab, nb_slab = pos[slab_idx], nb_j[slab_idx]
    n_hat_slab = face_unit_vectors(pos, nb_slab, pos_own=pos_slab)
    x_slab, z_slab = pos_slab[:, 0], pos_slab[:, 2]
    print(f"slab |y|<{args.slab_half_thickness}: {slab_idx.size} / {pos.shape[0]} cells", flush=True)

    fine_start_step = int(round(args.fine_start_t / dt))
    fine_end_step = int(round((args.fine_start_t + args.fine_duration_t) / dt))
    plot_steps = sorted(set(int(round(t_ / dt)) for t_ in plot_times))
    print(f"fine logging: steps [{fine_start_step},{fine_end_step}) every {args.fine_log_every}", flush=True)
    print(f"plot steps: {plot_steps}", flush=True)

    np.save(f"{args.out_prefix}_pos.npy", pos)

    from pathlib import Path

    frames_dir = Path(f"{args.out_prefix}_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(t_phys, Smag_slab):
        # Smag_slab is already restricted to the slab -- renders and
        # discards it, only the PNG is kept, no raw-array accumulation
        finite_pos = Smag_slab[np.isfinite(Smag_slab) & (Smag_slab > 0)]
        vmin = np.log10(np.percentile(finite_pos, 1)) if finite_pos.size else -10
        vmax = np.log10(np.percentile(finite_pos, 99)) if finite_pos.size else 0
        c = np.log10(np.clip(Smag_slab, 10.0**vmin, None))

        fig, ax = plt.subplots(figsize=(6, 5.5))
        sc = ax.scatter(x_slab, z_slab, c=c, s=3, cmap="inferno", vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(
            f"Poynting-ring |S|, XZ slice (|y|<{args.slab_half_thickness:.1f})\n"
            f"t={t_phys:.3f}  log10|S| in [{vmin:.1f},{vmax:.1f}]",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = frames_dir / f"t_{t_phys:07.3f}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"saved {out_path}", flush=True)

    trace = []
    global_step = start_step
    for step in range(args.n_steps):
        in_fine = fine_start_step <= global_step < fine_end_step
        log_every = args.fine_log_every if in_fine else args.coarse_log_every
        is_last_step = step == args.n_steps - 1
        want_plot = global_step in plot_steps or is_last_step
        do_log = (global_step % log_every == 0) or is_last_step or want_plot

        # (14,N) view/array from the C++ binding -- sliced by column below,
        # never transposed+copied to a full (N,14) array (see module docstring)
        phi_full = np.asarray(cl.leaf_phi) if do_log else None
        alat.step(0, dt)
        alat.step_sponge(0)

        if do_log:
            u_after = np.array(cl.leaf_u)

            # trace: always the shell union only, same per-cell math regardless
            # of want_plot, just skips the N-2e6 cells nobody's looking at
            phi_u = phi_full[:, union_idx].T
            S3_u = local_energy_flux_vector(pos_u, nb_u, phi_u, u_after, n_hat=n_hat_u, u_own=u_after[union_idx])
            Smag_u = np.linalg.norm(S3_u, axis=1)
            row_vals = [float(np.mean(Smag_u[sm])) if sm.any() else float("nan") for sm in shell_sub_masks]
            t_phys = (global_step + 1) * dt
            row = [t_phys] + row_vals
            trace.append(row)
            if global_step % 500 == 0 or want_plot:
                print(
                    f"step={global_step:6d} t={t_phys:8.3f} "
                    + " ".join(f"S[r={r0:.0f}]={v:.3e}" for r0, v in zip(shells, row[1:])),
                    flush=True,
                )
            if want_plot:
                phi_slab = phi_full[:, slab_idx].T
                S3_slab = local_energy_flux_vector(
                    pos_slab, nb_slab, phi_slab, u_after, n_hat=n_hat_slab, u_own=u_after[slab_idx]
                )
                Smag_slab = np.linalg.norm(S3_slab, axis=1)
                save_frame(t_phys, Smag_slab)

        global_step += 1

    trace = np.array(trace)
    np.save(f"{args.out_prefix}_trace.npy", trace)

    if args.final_state_out:
        from pathlib import Path as _Path

        base = _Path(args.final_state_out)
        state_path = base.with_name(f"{base.stem}_step{global_step:07d}{base.suffix or '.h5'}")
        alat.save(str(state_path))
        print(f"saved final state: {state_path}  (step={global_step}, t={global_step * dt:.3f})", flush=True)

    # period estimate from the fine-logged window at the innermost shell (strongest signal)
    fine_mask = (trace[:, 0] >= args.fine_start_t) & (trace[:, 0] < args.fine_start_t + args.fine_duration_t)
    tt, ss = trace[fine_mask, 0], trace[fine_mask, 1]
    peaks = [tt[i] for i in range(1, len(ss) - 1) if ss[i] > ss[i - 1] and ss[i] > ss[i + 1]]
    print(f"\nfine-window peaks at shell r={shells[0]}: {[round(p,4) for p in peaks]}")
    if len(peaks) > 1:
        diffs = np.diff(peaks)
        print(f"peak-to-peak intervals: {[round(d,4) for d in diffs]}")
        print(f"mean period estimate (fine-resolved): {diffs.mean():.4f}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
