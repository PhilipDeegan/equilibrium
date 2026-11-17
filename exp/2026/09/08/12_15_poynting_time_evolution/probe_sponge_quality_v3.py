"""
v2: the H(t)-based sponge comparison (probe_sponge_quality.py) was the
WRONG diagnostic -- total domain energy is dominated by the ring's own
near-field/reactive content that never reaches the boundary at all, so it
came back flat (99.96-99.97% retained) across every sponge config,
regardless of tuning; also 4000 steps is less than one period of the
ring's own natural oscillation (period~32), nowhere near enough cycles for
radiation loss to register in a global energy sum either way.

This version tracks the shell-averaged |S| signal instead (the same
diagnostic from probe_poynting_time_evolution.py) at shells near the
sponge (r=20, r=24) AND well inside it (r=8), over more than one full
oscillation period, for the same 4 sponge configs -- if a deeper/better-
tuned sponge genuinely reduces reflection, the outer shells should show a
cleaner (lower-amplitude, faster-decaying-after-first-pass) signal there
than the default; if all configs look equally messy, reflection likely
isn't the dominant cause of the late-time irregularity.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import face_unit_vectors, local_energy_flux_vector
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring, init_toroidal_winding, leaf_nb_j

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--mode", type=str, default="octagon", choices=["square", "octagon"])
parser.add_argument("--hops", type=int, default=8)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=6500)  # > 1 period (period~32/dt~6240 steps)
parser.add_argument("--log-every", type=int, default=25)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--sector-half-angle", type=float, default=20.0)
parser.add_argument("--shell-width-frac", type=float, default=0.15)
parser.add_argument("--shells", type=str, default="8,20,24")
parser.add_argument("--damp-values", type=str, default="0.003,0.005,0.007,0.01,0.015,0.02")
parser.add_argument(
    "--gamma-values",
    type=str,
    default=None,
    help="integrated-variant sweep values (a RATE, unlike --damp-values' fixed fraction); "
    "roughly damp_max/dt to be comparable -- defaults to --damp-values if unset, which is "
    "almost certainly the wrong scale (see [[project_poynting_perf_r150_farfield]])",
)
parser.add_argument("--variants", type=str, default="split,integrated,directional", choices=None)
parser.add_argument(
    "--taper-width",
    type=float,
    default=3.0,
    help="outflow_taper variant only: taper band width (same length units as --R), see "
    "exp/2026/09/16/12_30_outflow_taper_probe -- idea 1, [[project_bc_fn_inert_on_u]]",
)
parser.add_argument(
    "--sponge-fraction",
    type=float,
    default=0.15,
    help="fraction of r_max given to the sponge -- widen this at small R to keep the sponge's "
    "absolute width (in cells) comparable across R, since a fixed fraction shrinks the ramp "
    "to near-nothing at small R (e.g. R=10*0.15 ~= 1.5 cells, vs R=65*0.15 ~= 9.8 cells)",
)
parser.add_argument(
    "--plot-times",
    type=str,
    default="",
    help="comma-separated physical times to render an XZ log10|S| cross-section frame at "
    "(e.g. bracketing the predicted round-trip time) -- empty disables plotting. Cheap even at "
    "large N: only the thin |y|<slab-half-thickness row subset is ever read at these steps, "
    "same row-subset trick as the shell-union trace (see probe_poynting_fine_and_snapshots.py).",
)
parser.add_argument("--slab-half-thickness", type=float, default=1.5)
parser.add_argument(
    "--out-prefix", type=str, default="exp/2026/09/08/12_15_poynting_time_evolution/sponge_v3"
)
parser.add_argument(
    "--final-state-out",
    type=str,
    default=None,
    help="if set, save the final AdaptiveLattice state -- the exact global step is embedded in "
    "the filename automatically (e.g. 'foo.h5' -> 'foo_step0003900.h5'), same convention as "
    "probe_poynting_fine_and_snapshots.py. NOTE: PML's u_aux/phi_aux auxiliary state is NOT "
    "persisted (see bc.hpp) -- a resumed PML run restarts that state at zero, a small bounded "
    "transient, not a correctness bug.",
)
parser.add_argument(
    "--restart-from", type=str, default=None, help="resume from an HDF5 state saved via --final-state-out"
)
parser.add_argument(
    "--start-step",
    type=int,
    default=None,
    help="global step index the restart state represents; auto-parsed from the --restart-from "
    "filename's _stepNNNNNNN suffix if omitted",
)


def save_frame(t_phys, Smag_slab, x_slab, z_slab, frames_dir, label):
    finite_pos = Smag_slab[np.isfinite(Smag_slab) & (Smag_slab > 0)]
    vmin = np.log10(np.percentile(finite_pos, 1)) if finite_pos.size else -10
    vmax = np.log10(np.percentile(finite_pos, 99)) if finite_pos.size else 0
    c = np.log10(np.clip(Smag_slab, 10.0**vmin, None))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(x_slab, z_slab, c=c, s=3, cmap="inferno", vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(f"{label}  t={t_phys:.3f}  log10|S| in [{vmin:.1f},{vmax:.1f}]", fontsize=10)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = frames_dir / f"t_{t_phys:07.3f}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


def run_config(args, sponge_fraction, damp_max, label, shells, variant="split"):
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    fresh = args.restart_from is None
    if fresh:
        alat = _lat.AdaptiveLattice()
        alat.add_cluster()
        cl = alat[0]
        _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
        alat.rebuild(0)
        start_step = 0
    else:
        # AdaptiveLattice.load() restores mesh/u/phi/lam exactly, but NOT the
        # BC (rebuilt unconditionally below regardless of fresh/restart) --
        # same gotcha as probe_poynting_fine_and_snapshots.py's _apply_sponge_bc.
        alat = _lat.AdaptiveLattice.load(args.restart_from)
        cl = alat[0]
        start_step = args.start_step
        if start_step is None:
            m = re.search(r"_step(\d+)\.h5$", args.restart_from)
            if not m:
                raise ValueError("--start-step not given and not parseable from --restart-from filename")
            start_step = int(m.group(1))
    alat.set_n_threads(args.threads)

    pos_all = np.array(cl.leaf_pos)
    r_all = np.linalg.norm(pos_all, axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - sponge_fraction)
    if variant == "integrated":
        # gamma is a rate (1/time), folded directly into the leapfrog step
        # (step_damped) instead of a separate post-step multiply -- see
        # [[project_poynting_perf_r150_farfield]]. damp_max is reused as the
        # gamma_max parameter name so the same sweep values are comparable.
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge_integrated(
                target=0.0, gamma_max=damp_max, r_all=r_all, r_inner=r_inner
            )
        )
    elif variant == "directional":
        # PML-lite: only damps the per-face flux component aligned with the
        # radial (outward) direction; tangential faces and u itself are
        # untouched -- see bc.hpp's gamma_face doc comment.
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge_directional(
                target=0.0, gamma_max=damp_max, pos_all=pos_all, r_inner=r_inner
            )
        )
    elif variant == "pml":
        # Genuine split-field PML: u/phi[k] split into a decaying radial
        # auxiliary + implicit tangential remainder, recombined each step --
        # see bc.hpp's proj_r/u_aux/phi_aux doc comment. Supersedes the
        # failed "directional" shortcut above.
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge_pml(
                target=0.0, gamma_max=damp_max, pos_all=pos_all, r_inner=r_inner
            )
        )
    elif variant == "outflow":
        # First-order Mur/Sommerfeld one-way condition, applied right at the
        # literal domain edge (nb[k]<0) -- no annulus/r_inner needed at all,
        # unlike every other variant here. damp_max is reused as c_lattice
        # (the assumed local propagation speed) so the same sweep-value
        # plumbing (--gamma-values) works unchanged. See bc.hpp's
        # outflow_active doc comment / [[project_bc_fn_inert_on_u]].
        cl.set_bc(_lat.BoundaryCondition.outflow(np.array(cl.leaf_nb), c_lattice=damp_max))
        r_inner = r_max  # no annulus; only affects the round-trip-time printout below
    elif variant == "outflow_ledger":
        # Ledger-capped (passive) ghost, 2026-09-16 (bc.hpp's outflow_credit
        # doc comment) -- same ghost as "outflow" but a face can never inject
        # more energy over its own history than it has genuinely absorbed.
        cl.set_bc(_lat.BoundaryCondition.outflow_ledger(np.array(cl.leaf_nb), c_lattice=damp_max))
        r_inner = r_max
    elif variant == "outflow_taper":
        # Same one-way ghost at r_max as "outflow", plus an additive
        # relaxation-sink taper band over the last --taper-width (real
        # interior leaves only, true boundary leaves excluded to avoid
        # double-counting -- see bc.hpp's outflow_taper_weight doc comment).
        cl.set_bc(
            _lat.BoundaryCondition.outflow_tapered(
                np.array(cl.leaf_nb), pos_all, taper_width=args.taper_width, c_lattice=damp_max
            )
        )
        r_inner = r_max
    elif variant == "pml_outflow":
        # Idea 4 ([[project_bc_fn_inert_on_u]]/doc/dst/bc.md): genuine PML
        # sponge annulus (r_inner..r_max) PLUS an outflow one-way Mur ghost
        # at the literal domain edge, instead of the fully-reflective
        # absorbing() wall every other sponge variant sits in front of.
        # damp_max is reused as gamma_max (same as "pml"); c_lattice fixed at
        # the outflow-sweep-confirmed optimum (sqrt(10)) rather than swept.
        cl.set_bc(
            _lat.BoundaryCondition.pml_outflow(
                np.array(cl.leaf_nb), pos_all, gamma_max=damp_max, r_inner=r_inner, c_lattice=np.sqrt(10.0)
            )
        )
    else:
        cl.set_bc(_lat.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=damp_max, r_all=r_all, r_inner=r_inner))

    # find_toroidal_ring is purely geometric (deterministic from mesh + mode/hops),
    # needed either way for the shell/plot geometry below -- only the actual
    # winding/lam SEEDING is skipped on restart (state already loaded).
    ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode=args.mode, hops=args.hops)
    if fresh:
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True)
        lam = np.array(cl.leaf_lam)
        for j in ring_js:
            lam[j] = args.lam_ring
        cl.set_leaf_lam(lam)
    pos = pos_all
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)

    dt = compute_dt(cl)
    r_ring = float(np.median(np.linalg.norm(ring_pos, axis=1)))
    c_void = np.sqrt(10.0)
    t_round_trip = 2.0 * (r_inner - r_ring) / c_void
    print(
        f"[{label}] N={cl.n_leaves} sponge_fraction={sponge_fraction} damp_max={damp_max} "
        f"r_inner={r_inner:.2f} r_ring={r_ring:.2f} dt={dt:.4e} start_step={start_step} "
        f"predicted_round_trip_t={t_round_trip:.2f} "
        f"(t range this leg: {start_step * dt:.2f} -> {(start_step + args.n_steps) * dt:.2f})",
        flush=True,
    )

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

    # Shell-union row subset only -- avoids the documented R150-scale perf trap
    # (full-domain phi copy/transpose + full-row face_unit_vectors rebuild every
    # logged step; see probe_poynting_fine_and_snapshots.py's module docstring
    # and [[project_poynting_perf_r150_farfield]]'s perf-fix history). Mesh is
    # static for this probe, so n_hat_u is precomputed once, not per step.
    union_idx = np.where(np.any(np.stack(shell_masks), axis=0))[0]
    pos_u = pos[union_idx]
    nb_u = nb_j[union_idx]
    n_hat_u = face_unit_vectors(pos, nb_u, pos_own=pos_u)
    shell_sub_masks = [m[union_idx] for m in shell_masks]

    # Optional XZ cross-section frames (row-subset trick again -- only the
    # thin |y|<slab-half-thickness slab is ever read, cheap even at N~18M).
    plot_times = [float(t) for t in args.plot_times.split(",") if t.strip()] if args.plot_times else []
    plot_steps = {int(round(t / dt)) - 1 for t in plot_times} if plot_times else set()
    frames_dir = None
    slab_idx = pos_s = nb_s = n_hat_s = x_slab = z_slab = None
    if plot_times:
        slab_ax = perp[1] if len(perp) > 1 else perp[0]
        plot_ax1 = perp[0]
        slab_mask = np.abs(pos[:, slab_ax]) < args.slab_half_thickness
        slab_idx = np.where(slab_mask)[0]
        pos_s = pos[slab_idx]
        nb_s = nb_j[slab_idx]
        n_hat_s = face_unit_vectors(pos, nb_s, pos_own=pos_s)
        x_slab = pos_s[:, plot_ax1]
        z_slab = pos_s[:, ax_idx]
        frames_dir = Path(f"{args.out_prefix}_{label}_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)

    trace = []
    final_global_step = start_step
    for step in range(args.n_steps):
        global_step = start_step + step
        want_plot = global_step in plot_steps
        do_log = (step % args.log_every == 0) or (step == args.n_steps - 1) or want_plot
        # np.asarray is a VIEW ((14,N), no copy/transpose) -- see the comment above.
        phi_full = np.asarray(cl.leaf_phi) if do_log else None
        if variant == "integrated":
            alat.step_damped(0, dt)
        elif variant == "directional":
            alat.step_dir_damped(0, dt)
        elif variant == "pml" or variant == "pml_outflow":
            alat.step_pml(0, dt)
        elif variant == "outflow" or variant == "outflow_taper" or variant == "outflow_ledger":
            alat.step(0, dt)  # both are built into step()'s own pass1/pass2_bc
        else:
            alat.step(0, dt)
            alat.step_sponge(0)
        final_global_step = global_step + 1
        if do_log:
            u_after = np.array(cl.leaf_u)
            phi_u = phi_full[:, union_idx].T
            S3_u = local_energy_flux_vector(pos_u, nb_u, phi_u, u_after, n_hat=n_hat_u, u_own=u_after[union_idx])
            Smag_u = np.linalg.norm(S3_u, axis=1)
            t_phys = final_global_step * dt
            row = [t_phys] + [float(np.mean(Smag_u[sm])) if sm.any() else float("nan") for sm in shell_sub_masks]
            trace.append(row)
            if want_plot:
                phi_s = phi_full[:, slab_idx].T
                S3_s = local_energy_flux_vector(pos_s, nb_s, phi_s, u_after, n_hat=n_hat_s, u_own=u_after[slab_idx])
                Smag_s = np.linalg.norm(S3_s, axis=1)
                save_frame(t_phys, Smag_s, x_slab, z_slab, frames_dir, label)

    if args.final_state_out:
        out_path = f"{args.final_state_out}_{label}_step{final_global_step:07d}.h5"
        alat.save(out_path)
        print(f"[{label}] saved final state -> {out_path}", flush=True)

    return np.array(trace)


def main():
    args = parser.parse_args()
    shells = [float(s) for s in args.shells.split(",")]
    # (sponge_fraction, damp_max/gamma_max, label, integrated) -- split (existing
    # vol_damp/step_sponge) vs integrated (new gamma/step_damped) at matched
    # damp values, so the two implementations are directly comparable.
    damp_values = [float(d) for d in args.damp_values.split(",")]
    gamma_values = [float(g) for g in (args.gamma_values or args.damp_values).split(",")]
    variants = args.variants.split(",")
    configs = []
    sf = args.sponge_fraction
    if "split" in variants:
        configs += [(sf, d, f"split_d{d}", "split") for d in damp_values]
    if "integrated" in variants:
        configs += [(sf, g, f"integ_g{g}", "integrated") for g in gamma_values]
    if "directional" in variants:
        configs += [(sf, g, f"dir_g{g}", "directional") for g in gamma_values]
    if "pml" in variants:
        configs += [(sf, g, f"pml_g{g}", "pml") for g in gamma_values]
    if "outflow" in variants:
        configs += [(sf, c, f"outflow_c{c}", "outflow") for c in gamma_values]
    if "outflow_ledger" in variants:
        configs += [(sf, c, f"outflow_ledger_c{c}", "outflow_ledger") for c in gamma_values]
    if "outflow_taper" in variants:
        configs += [(sf, c, f"outflow_taper_c{c}", "outflow_taper") for c in gamma_values]
    if "pml_outflow" in variants:
        configs += [(sf, g, f"pml_outflow_g{g}", "pml_outflow") for g in gamma_values]
    results = {}
    for frac, damp, label, variant in configs:
        trace = run_config(args, frac, damp, label, shells, variant=variant)
        np.save(f"{args.out_prefix}_trace_{label}.npy", trace)
        results[label] = trace
        print(f"[{label}] done, {len(trace)} log points", flush=True)

    print("\n=== late-time (last third of run) outer-shell |S| comparison across configs ===")
    for label, trace in results.items():
        n = len(trace)
        late = trace[2 * n // 3 :]
        for i, r0 in enumerate(shells):
            col = late[:, i + 1]
            finite = col[np.isfinite(col) & (col > 0)]
            if finite.size:
                print(f"{label:20s} r={r0:4.0f}: mean={finite.mean():.4e} std={finite.std():.4e} max={finite.max():.4e}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
