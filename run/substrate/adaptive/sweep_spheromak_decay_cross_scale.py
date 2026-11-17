"""
Adaptive spheromak decay — cross_scale calibration sweep.

Re-run of exp/2026/06/18/14_00_spheromak_decay_sweep (cs=0.005 viable,
cs=0.010 "too aggressive") — now stale for two reasons:
  1. That sweep predates the sigma(lam)^2 resistive fix to step_cross — lam
     stayed pinned at 1.0 the whole run with zero resistance, so decay was
     completely lam-blind.
  2. It also predates the adaptive step_cross sign-convention fix (2026-07-04):
     the old adaptive step_cross double-processed every antisymmetric face
     pair (see doc/mems/project_step_cross.md), so even a lam-blind comparison
     would have decayed ~2x faster than the now-fixed single-processing path.

Multiprocessing pattern mirrors run/substrate/validate_step_cross.py: each
cross_scale value runs in its own subprocess via a top-level picklable worker
that builds its own AdaptiveLattice from scratch (deterministic IC — no
randomness in find_toroidal_spheromak/init_toroidal_winding — so every worker
reproduces the identical IC without needing to pickle large arrays across the
process boundary).

Default sweep: 8 log-spaced cross_scale values in [1e-4, 1e-2].

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_spheromak_decay_cross_scale.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_spheromak_decay_cross_scale.py \\
      --cross-scales 1e-4 3e-4 1e-3 3e-3 1e-2 --n-steps 50000
"""

import argparse
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from tst.substrate.adaptive import W2_SUM, GrowingAMRSchedule
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_spheromak,
    init_toroidal_winding,
    phi_ring_circulation,
)


def _replay_mesh(alat, cl, saved_keys, saved_levels, amr_interval):
    """Re-split a fresh L0 lattice until its leaf set matches the snapshot's
    leaf_ancestry keys.

    Ancestry (root L0 gid + child-index path) is the only cross-run-stable
    leaf identity: positions alias (adjacent split parents place children on
    the shared face midpoint — same centre, same level) and child gids depend
    on split order.

    Reconstructs by direct split_batch() calls (deterministic: this cell
    splits, full stop) rather than the criterion_max/adapt() threshold
    machinery — adapt() can also coarsen, and going through it here was the
    root cause of a real bug: an L8 telescope replayed via adapt() silently
    collapsed to L3 on restart (see project_odd27_amr_junction_leak). split()
    has no coarsen path at all, so this cannot regress deep cells. Any
    current leaf whose ancestry is absent from the snapshot must be an
    ancestor of a finer saved leaf, so it gets split (split_batch silently
    skips any cell that fails can_split's max_level cap; those retry next
    round once nothing else remains to split).

    amr_interval is no longer used here: since split_batch never touches
    cl's internal adapt-call counter, that counter is left exactly where it
    was before this function ran (0, for a freshly constructed cluster) —
    the same phase a non-restarted run starts at, so there is nothing left
    to pad.
    """
    del amr_interval
    max_level = int(np.max(saved_levels))
    cl.info = _lat.DSTInfo()
    cl.info.max_level = max_level

    saved_keys = np.asarray(saved_keys)
    for _ in range(4 * (max_level + 2)):
        cur = np.array(cl.leaf_ancestry)
        mism = ~np.isin(cur, saved_keys)
        if not mism.any():
            break
        j_need = np.nonzero(mism)[0].astype(np.int32)
        alat.split_batch(0, j_need)
    else:
        raise RuntimeError(
            f"_replay_mesh: could not reproduce snapshot mesh "
            f"({cl.n_leaves} leaves vs {len(saved_keys)} saved) — snapshot "
            f"from a different lattice build?"
        )

    cur = np.array(cl.leaf_ancestry)
    if set(cur.tolist()) != set(saved_keys.tolist()):
        raise RuntimeError(
            f"_replay_mesh: reconstructed mesh ({len(cur)} leaves) does not "
            f"exactly match the snapshot's leaf set ({len(saved_keys)} "
            f"leaves) — a leaf-count match alone is not sufficient proof of "
            f"an exact reconstruction."
        )

    cl.info = _lat.DSTInfo()


def _compute_dt_l0(cl, w2_sum=W2_SUM, D1=1.0):
    """CFL timestep for an L0 cell, excluding fixed-u cells.

    Deliberately ignores the current max refinement level — TimeStepper.by_level
    (used when AMR is enabled) subcycles finer cells internally, so the caller
    only ever needs the coarse-cell dt, not a dt shrunk by 2**max_level for the
    whole mesh.
    """
    fu_arr = np.array(cl.leaf_fixed_u).astype(bool)
    lam_arr = np.array(cl.leaf_lam, dtype=np.float64)
    free = ~fu_arr
    sigma_max = float(np.max(1.0 / (1.0 + lam_arr[free])))
    return 0.4 * D1 / (sigma_max * w2_sum)


parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--lam-rings", type=float, default=1.0, help="lam on ring tiles (0=vacuum)"
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument("--axis", default="z")
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-steps", type=int, default=50000)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument(
    "--max-level",
    type=int,
    default=0,
    help="AMR max refinement level (0 = AMR disabled, flat BCC lattice as before)",
)
parser.add_argument(
    "--criterion",
    choices=["relative_u", "absolute"],
    default="relative_u",
    help="split criterion — relative_u (phi/u, scale-invariant, recommended) or "
    "absolute (raw |phi| threshold, diagnostic only)",
)
parser.add_argument(
    "--split-hi",
    type=float,
    default=0.0,
    help="split threshold (0 = refinement never fires even if max-level > 0)",
)
parser.add_argument(
    "--split-lo", type=float, default=0.0, help="coarsen threshold (0 = disabled)"
)
parser.add_argument(
    "--split-ref",
    type=float,
    default=1e-6,
    help="u floor for relative_u — this IC starts at u=0 everywhere, so a hard "
    "floor (not an auto-derived percentile) is needed to avoid divide-by-zero",
)
parser.add_argument(
    "--amr-interval",
    type=int,
    default=1000,
    help="DSTInfo.split_interval — adapt() is called every step but the C++ side "
    "only evaluates split/coarsen every this many calls.  With "
    "--amr-growing, this becomes the interval CAP instead of a fixed value.",
)
parser.add_argument(
    "--amr-growing",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="use a geometrically growing adapt cadence (GrowingAMRSchedule) instead "
    "of the fixed --amr-interval: re-mesh often while the IC is still "
    "settling, coarsening the cadence up to --amr-interval as the cap once "
    "refinement has converged.  On by default whenever AMR is enabled "
    "(max-level>0 and split-hi>0).  Automatically disabled (falls back to "
    "the fixed --amr-interval) under --restart-from, since the restart "
    "mesh-replay's adapt_calls padding assumes a fixed cadence — pass "
    "--no-amr-growing to opt out explicitly instead.",
)
parser.add_argument(
    "--amr-growing-start",
    type=int,
    default=10,
    help="first adapt-call interval under --amr-growing (doubles thereafter, "
    "capped at --amr-interval)",
)
parser.add_argument("--n-jobs", type=int, default=8)
parser.add_argument(
    "--n-threads",
    type=int,
    default=1,
    help="C++ threads per job (parallelism is process-level by default; "
    "raise this to also thread within each job, e.g. when running "
    "fewer concurrent jobs than cores)",
)
parser.add_argument("--lo", type=float, default=1e-4)
parser.add_argument("--hi", type=float, default=1e-2)
parser.add_argument(
    "--cross-scales",
    type=float,
    nargs="+",
    default=None,
    help="explicit cross_scale values (overrides --n-jobs log-spacing between --lo/--hi)",
)
parser.add_argument(
    "--out", type=str, default="tmp/spheromak_decay_cross_scale_sweep.png"
)
parser.add_argument(
    "--save-snapshot-dir",
    type=str,
    default=None,
    help="if set, each job dumps its final phi/u/lam/pos state to "
    "<dir>/decay_cs<cross_scale>_step<final_step>.npz for later field-level "
    "inspection or --restart-from",
)
parser.add_argument(
    "--plot-dir",
    type=str,
    default=None,
    help="if set, each job renders a probe_decay_compare field plot every "
    "--plot-every steps (plus t=0) into <dir>/cs<cross_scale>/",
)
parser.add_argument(
    "--plot-every",
    type=int,
    default=1000,
    help="step cadence for --plot-dir field snapshots",
)
parser.add_argument(
    "--restart-from",
    type=str,
    default=None,
    help="directory of decay_R<R>_cs<cross_scale>_step<n>.npz snapshots (as "
    "written by --save-snapshot-dir) to resume field state from and run "
    "--n-steps further steps; the start step is parsed from each filename; "
    "cross_scales with no matching snapshot start from zero",
)


def _build_and_run(
    cross_scale,
    R,
    amplitude,
    lam_rings,
    lam_scale,
    lam_mode,
    axis,
    sponge,
    damp_max,
    n_steps,
    report,
    max_level=0,
    criterion="relative_u",
    split_hi=0.0,
    split_lo=0.0,
    split_ref=1e-6,
    amr_interval=200,
    amr_growing=False,
    amr_growing_start=10,
    save_snapshot_path=None,
    plot_dir=None,
    plot_every=1000,
    restart_path=None,
    start_step=0,
    n_threads=1,
):
    """Build a fresh Hopf-ring spheromak IC and run one cross_scale to n_steps.

    Deterministic — every call with the same kwargs (aside from cross_scale)
    produces a bit-identical IC, so this is safe to call independently in
    each worker process without sharing state.
    """
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(n_threads)
    cl = alat[0]

    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    def _rebuild_sponge():
        # quadratic_sponge's vol_damp is indexed by compact leaf index, so a
        # BC built from the L0 mesh scrambles (and overruns) once the leaf
        # set changes — rebuild from current leaf radii after every mesh
        # change. r_inner is held fixed from the L0 mesh; refined boundary
        # children can sit marginally beyond the L0 r_max, shifting the
        # damping span by < one fine-cell width (irrelevant for a sponge).
        r_now = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge(
                target=0.0, damp_max=damp_max, r_all=r_now, r_inner=r_inner
            )
        )

    A_js, A_faces, B_js, B_faces = find_toroidal_spheromak(cl, axis=axis)
    if A_js is None:
        raise RuntimeError("no toroidal spheromak found — lattice too small?")

    # Ring global ids — captured on the L0 mesh, stable across split/coarsen
    # (compact leaf indices are not).
    gid0 = np.array(cl.leaf_global_indices)
    A_gs = gid0[A_js]
    B_gs = gid0[B_js]

    def _seed_pristine_winding():
        """(Re-)write the winding IC from scratch onto the CURRENT leaf set.

        Idempotent — zeroes u/phi first, so it's safe to call again after
        pre-adapting the mesh (see the pre-adapt block below), producing the
        same pristine IC the run would have started from had the mesh
        always been at its converged refined structure, instead of the
        field state left behind by pre-adapt's own warm-up steps.
        """
        cl.set_leaf_u(np.zeros(cl.n_leaves, dtype=np.float64))
        cl.set_leaf_phi(np.zeros((14, cl.n_leaves), dtype=np.float64))
        init_toroidal_winding(cl, A_js, A_faces, amplitude=amplitude, phi_only=True)
        init_toroidal_winding(cl, B_js, B_faces, amplitude=amplitude, phi_only=True)
        if lam_rings > 0.0:
            lam = np.zeros(cl.n_leaves, dtype=np.float64)
            for i in set(A_js) | set(B_js):
                lam[i] = lam_rings
            cl.set_leaf_lam(lam)
        else:
            cl.set_leaf_lam(np.zeros(cl.n_leaves, dtype=np.float64))

    if restart_path is None:
        _seed_pristine_winding()
    else:
        data = np.load(restart_path)
        if "R" in data.files and abs(float(data["R"]) - R) > 1e-9:
            raise ValueError(
                f"restart snapshot {restart_path} was built at R={float(data['R'])}, "
                f"but this run requested R={R} — refusing to load mismatched topology"
            )
        n_l0 = cl.n_leaves
        n_saved = int(data["pos"].shape[0])
        if n_saved != n_l0 and "key" not in data.files:
            raise RuntimeError(
                f"restart snapshot {restart_path} was taken on an AMR-adapted "
                f"mesh ({n_saved} leaves vs {n_l0} at L0) but predates the "
                f"leaf_ancestry snapshot format — the mesh cannot be safely "
                f"reconstructed (leaf positions alias between adjacent split "
                f"parents). Re-save it with the updated script."
            )
        if n_saved != n_l0:
            # Snapshot was taken on an AMR-adapted mesh — replay the
            # refinement so the leaf sets match before loading fields.
            _replay_mesh(alat, cl, data["key"], data["level"], amr_interval)
        # Permute saved leaf columns into the current leaf order. Ancestry
        # keys when available; leaf order is deterministic for flat legacy
        # snapshots (same build_bcc_sphere), so identity is safe there.
        if "key" in data.files:
            saved_keys = np.asarray(data["key"])
            order = np.argsort(saved_keys)
            cur = np.array(cl.leaf_ancestry)
            perm = order[np.searchsorted(saved_keys[order], cur)]
            if not np.array_equal(saved_keys[perm], cur):
                raise RuntimeError(
                    f"restart snapshot {restart_path}: leaf ancestry does not "
                    f"match the rebuilt mesh — snapshot from a different "
                    f"lattice build?"
                )
        else:
            perm = np.arange(n_saved)
        cl.set_leaf_phi(np.ascontiguousarray(data["phi"][:, perm]))
        cl.set_leaf_u(np.ascontiguousarray(data["u"][perm]))
        cl.set_leaf_lam(np.ascontiguousarray(data["lam"][perm]))
        if n_saved != n_l0:
            # Compact leaf indices shifted during replay — re-derive the ring
            # tiles from their stable global ids, and rebuild the sponge BC
            # for the new leaf ordering. Ring tiles are masked from splitting
            # during the run, so their gids are always leaves in the snapshot.
            g_to_j = {int(g): j for j, g in enumerate(np.array(cl.leaf_global_indices))}
            A_js = [g_to_j[int(g)] for g in A_gs]
            B_js = [g_to_j[int(g)] for g in B_gs]
            _rebuild_sponge()

    centre_A = np.array(cl.leaf_pos)[A_js].mean(axis=0)
    centre_B = np.array(cl.leaf_pos)[B_js].mean(axis=0)
    dt = _compute_dt_l0(cl)  # L0 dt; TimeStepper.by_level subcycles finer cells

    def _remap_ring():
        nonlocal A_js, B_js, centre_A, centre_B
        gid_now = np.array(cl.leaf_global_indices)
        g_to_j = {int(g): j for j, g in enumerate(gid_now)}
        A_js = [g_to_j[int(g)] for g in A_gs]
        B_js = [g_to_j[int(g)] for g in B_gs]
        pos_now = np.array(cl.leaf_pos)
        centre_A = pos_now[A_js].mean(axis=0)
        centre_B = pos_now[B_js].mean(axis=0)

    # ------------------------------------------------------------------
    # AMR — no geometric restriction: split_criterion decides purely from
    # local phi/u dynamics, so refinement (or lack of it) at any location,
    # including the sphere centre, is an emergent outcome, not imposed.
    # A_js/B_js are re-derived from stable global leaf ids (not the compact
    # leaf index, which shifts on every split/coarsen) whenever the leaf
    # count changes, so they never point at the wrong cells.
    # ------------------------------------------------------------------
    amr_enabled = max_level > 0 and split_hi > 0.0
    # Growing cadence assumes it owns adapt_calls from step 0 — the restart
    # mesh-replay path pads that counter to align with a FIXED split_interval
    # (see _replay_mesh), so fall back silently rather than fight it.
    amr_growing = amr_growing and restart_path is None
    amr_schedule = None
    sc = None
    if amr_enabled:
        sc = (
            _lat.SplitCriterion.relative_u(hi=split_hi, lo=split_lo, ref=split_ref)
            if criterion == "relative_u"
            else _lat.SplitCriterion.absolute(hi=split_hi, lo=split_lo)
        )

    # by_level whenever fine leaves exist (e.g. a replayed AMR restart run
    # with refinement disabled) — fixed(dt) at the L0 CFL is unstable there.
    has_fine = int(np.max(np.array(cl.leaf_level))) > 0
    ts = (
        _lat.TimeStepper.by_level(dt)
        if (amr_enabled or has_fine)
        else _lat.TimeStepper.fixed(dt)
    )
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    if amr_enabled and restart_path is None:
        # Bring the mesh to its organically-converged refinement structure
        # BEFORE the recorded run begins, instead of growing into it
        # reactively over the first several amr_growing_start/amr_interval
        # steps — which would otherwise run the ring's high-gradient
        # neighbourhood at an under-resolved mesh for a while (see
        # [[project_subcycle_amr_reflux]] memory / TestByLevelStability for
        # the AMR unit-test analogue of this, refine_region_to_convergence
        # in tst/substrate/adaptive/__init__.py). The relative_u/absolute
        # SplitCriterion reads live criterion_max data, populated only by
        # _step_pass3 — i.e. by actually stepping — so this interleaves one
        # step with each adapt() round rather than refining a static
        # geometric region up front the way the unit-test helper does.
        # split_interval is forced to 1 here so every round evaluates.
        # Round cap: depth is bounded by max_level, but the SPATIAL extent
        # of the region needing refinement grows outward roughly one hop
        # per round (each round is one step), so it's not bounded by
        # max_level alone — this is a generous head-start budget, not a
        # correctness requirement. Not fully converging here isn't fatal:
        # the main loop's own organic regrow (amr_schedule/amr_interval)
        # keeps extending refinement throughout the run regardless: this
        # phase only exists to avoid the WORST of the initial under-
        # resolved transient right at the ring, not to fully pre-solve the
        # mesh's eventual steady-state extent.
        PRE_ADAPT_MAX_ROUNDS = 50
        pre_info = _lat.DSTInfo()
        pre_info.max_level = max_level
        pre_info.split_interval = 1
        cl.info = pre_info
        cl.set_split_criterion(sc)
        n_pre_adapts = 0
        for _ in range(PRE_ADAPT_MAX_ROUNDS):
            evolver.step(ts)
            cmax = np.array(cl.criterion_max)
            cmax[A_js] = np.nan
            cmax[B_js] = np.nan
            cl.set_criterion_max(cmax)
            n_before = cl.n_leaves
            alat.adapt(0)
            n_pre_adapts += 1
            if cl.n_leaves == n_before:
                break
            _remap_ring()
            _rebuild_sponge()
        else:
            print(
                f"pre-adapt IC: mesh still growing after {PRE_ADAPT_MAX_ROUNDS} rounds "
                f"({cl.n_leaves} leaves) — proceeding; organic regrow in the main "
                f"loop will continue extending it"
            )
        # The warm-up steps above advanced the field state — re-seed the
        # pristine winding IC on the now-converged (and otherwise
        # untouched) mesh, so the recorded run starts from exactly the
        # intended t=0 state, just already at its natural resolution.
        _seed_pristine_winding()
        dt = _compute_dt_l0(cl)

    if amr_enabled:
        final_split_interval = 1 if amr_growing else max(1, amr_interval)
        if restart_path is None and not amr_growing:
            # cl.adapt_calls (the C++ cadence counter driving split_interval)
            # is a cluster field, not part of cl.info and not exposed to
            # Python — reassigning cl.info above does NOT reset it, so the
            # pre-adapt loop's own n_pre_adapts adapt() calls left it out of
            # sync with the main loop's assumption that adapt_calls==0 at the
            # start of the recorded run. Pad it back to a multiple of
            # final_split_interval — same pattern _replay_mesh uses for
            # restarts — so the main loop's Python-side (rel_step based)
            # ring-mask refresh and the C++-side call_idx cadence gate stay
            # aligned; otherwise a ring tile can slip through unprotected on
            # a misaligned evaluation round (see [[project_subcycle_amr_reflux]]
            # memory — this exact failure mode was caught by a smoke test,
            # not just reasoned about). Only relevant for the fixed
            # (non-growing) cadence — growing always uses split_interval=1,
            # where every call evaluates regardless of the counter's value.
            cl.clear_split_criterion()
            cl.info = (
                _lat.DSTInfo()
            )  # split_hi=0/split_lo=0/split_interval=1 -> genuine no-op
            for _ in range((-n_pre_adapts) % final_split_interval):
                alat.adapt(0)
        info = _lat.DSTInfo()
        info.max_level = max_level
        info.split_interval = final_split_interval
        cl.info = info
        cl.set_split_criterion(sc)
        if amr_growing:
            amr_schedule = GrowingAMRSchedule(
                start_interval=amr_growing_start, max_interval=max(1, amr_interval)
            )

    ring_idx = set(A_js) | set(B_js)
    ring_r = float(
        np.mean(np.linalg.norm(np.array(cl.leaf_pos)[list(ring_idx)], axis=1))
    )
    cs_plot_dir = (
        f"{plot_dir}/R{int(round(R))}_cs{cross_scale:.4e}" if plot_dir else None
    )

    def plot_field_snapshot(step):
        from run.substrate.adaptive.probe_decay_compare import plot_snapshots

        snap = dict(
            label=f"R={int(round(R))} cs={cross_scale:.2e} step={step}",
            phi=np.array(
                cl.leaf_phi
            ).T.copy(),  # leaf_phi is (14, N); plot_snapshots wants (N, 14)
            u=np.array(cl.leaf_u).copy(),
            lam=np.array(cl.leaf_lam).copy(),
            pos=np.array(cl.leaf_pos).copy(),
            ring_r=ring_r,
        )
        gridsize = max(20, int(r_max * 1.3))
        plot_snapshots(
            [snap], plot_dir=cs_plot_dir, tag=f"step{step:06d}", gridsize=gridsize
        )

    n_rep = n_steps // report + 1
    t_hist = np.zeros(n_rep)
    H_hist = np.zeros(n_rep)
    phi_rms_h = np.zeros(n_rep)
    lam_A_h = np.zeros(n_rep)
    lam_B_h = np.zeros(n_rep)
    circ_A_h = np.zeros(n_rep)
    circ_B_h = np.zeros(n_rep)

    def record(si, step):
        phi_a = np.array(cl.leaf_phi)
        u_a = np.array(cl.leaf_u)
        lam_a = np.array(cl.leaf_lam)
        t_hist[si] = step * dt
        H_hist[si] = float(np.sum(u_a**2) + 0.5 * np.sum(phi_a**2))
        phi_rms_h[si] = float(np.sqrt(np.mean(phi_a**2)))
        lam_A_h[si] = float(np.mean(lam_a[A_js]))
        lam_B_h[si] = float(np.mean(lam_a[B_js]))
        circ_A_h[si] = phi_ring_circulation(cl, A_js, centre_A)
        circ_B_h[si] = phi_ring_circulation(cl, B_js, centre_B)

    si = 0
    record(0, start_step)
    if cs_plot_dir and plot_every > 0:
        plot_field_snapshot(start_step)
    for rel_step in range(1, n_steps + 1):
        step = start_step + rel_step
        evolver.step(ts)
        alat.step_sponge(0)
        if cross_scale > 0.0:
            alat.step_cross(0, dt, cross_scale)
        if lam_mode == "dynamic":
            alat.step_lam(0, dt, lam_scale)
        # Every cell's split/coarsen eligibility is decided purely by the
        # dynamics-driven criterion — except the ring tiles themselves, which
        # must stay single leaves or A_js/B_js/find_toroidal_spheromak's
        # assumed 8/4-tile connectivity breaks. Masked to NaN (never acted on)
        # right before the interval where adapt() actually evaluates.
        #
        # Growing cadence: info.split_interval=1, so adapt() would evaluate
        # (and re-mesh) on EVERY call — only call it at all on schedule.due()
        # steps.  Fixed cadence: adapt() is called every step as before,
        # relying on the C++ split_interval gate to no-op between intervals
        # (unchanged, so restart's adapt_calls padding still lines up).
        if amr_schedule is not None:
            if amr_schedule.due(rel_step):
                cmax = np.array(cl.criterion_max)
                cmax[A_js] = np.nan
                cmax[B_js] = np.nan
                cl.set_criterion_max(cmax)
                n_before = cl.n_leaves
                alat.adapt(0)
                if cl.n_leaves != n_before:
                    _remap_ring()
                    _rebuild_sponge()
        elif amr_enabled:
            if (rel_step - 1) % info.split_interval == 0:
                cmax = np.array(cl.criterion_max)
                cmax[A_js] = np.nan
                cmax[B_js] = np.nan
                cl.set_criterion_max(cmax)
            n_before = cl.n_leaves
            alat.adapt(0)
            if cl.n_leaves != n_before:
                _remap_ring()
                _rebuild_sponge()
        if rel_step % report == 0:
            si += 1
            record(si, step)
        if cs_plot_dir and plot_every > 0 and rel_step % plot_every == 0:
            plot_field_snapshot(step)

    if save_snapshot_path:
        np.savez(
            save_snapshot_path,
            phi=np.array(cl.leaf_phi).copy(),
            u=np.array(cl.leaf_u).copy(),
            lam=np.array(cl.leaf_lam).copy(),
            pos=np.array(cl.leaf_pos).copy(),
            level=np.array(cl.leaf_level).copy(),
            key=np.array(cl.leaf_ancestry).copy(),
            A_tiles=np.array(A_js, dtype=int),
            B_tiles=np.array(B_js, dtype=int),
            R=np.asarray(R, dtype=float),
            cross_scale=np.asarray(cross_scale, dtype=float),
        )

    return dict(
        t=t_hist[: si + 1],
        H=H_hist[: si + 1],
        phi_rms=phi_rms_h[: si + 1],
        lam_A=lam_A_h[: si + 1],
        lam_B=lam_B_h[: si + 1],
        circ_A=circ_A_h[: si + 1],
        circ_B=circ_B_h[: si + 1],
        n_A=len(A_js),
        n_B=len(B_js),
        dt=dt,
        n_leaves=cl.n_leaves,
        level_max=int(np.max(np.array(cl.leaf_level))),
    )


def _worker(task):
    """Top-level, picklable — runs one cross_scale value in its own subprocess."""
    cross_scale, kw = task
    return cross_scale, _build_and_run(cross_scale, **kw)


def main():
    args = parser.parse_args()

    if args.cross_scales:
        cross_scales = sorted(args.cross_scales)
    else:
        cross_scales = list(
            np.logspace(np.log10(args.lo), np.log10(args.hi), args.n_jobs)
        )

    kw = dict(
        R=args.R,
        amplitude=args.amplitude,
        lam_rings=args.lam_rings,
        lam_scale=args.lam_scale,
        lam_mode=args.lam_mode,
        axis=args.axis,
        sponge=args.sponge,
        damp_max=args.damp_max,
        n_steps=args.n_steps,
        report=args.report,
        max_level=args.max_level,
        criterion=args.criterion,
        split_hi=args.split_hi,
        split_lo=args.split_lo,
        split_ref=args.split_ref,
        amr_interval=args.amr_interval,
        amr_growing=args.amr_growing,
        amr_growing_start=args.amr_growing_start,
        plot_dir=args.plot_dir,
        plot_every=args.plot_every,
        n_threads=args.n_threads,
    )

    print(
        f"R={args.R}  n_steps={args.n_steps}  lam_rings={args.lam_rings}  "
        f"lam_scale={args.lam_scale}  lam_mode={args.lam_mode}"
    )
    if args.max_level > 0:
        cadence = (
            f"growing {args.amr_growing_start}->{args.amr_interval}"
            if args.amr_growing and not args.restart_from
            else f"fixed {args.amr_interval}"
        )
        print(
            f"AMR: max_level={args.max_level}  criterion={args.criterion}  "
            f"split_hi={args.split_hi:.3e}  split_lo={args.split_lo:.3e}  "
            f"split_ref={args.split_ref:.3e}  amr_interval={cadence}"
        )
    print(
        f"cross_scales ({len(cross_scales)}): "
        + ", ".join(f"{c:.4e}" for c in cross_scales)
    )

    if args.save_snapshot_dir:
        from pathlib import Path

        Path(args.save_snapshot_dir).mkdir(parents=True, exist_ok=True)

    R_tag = int(round(args.R))

    restart_paths = {}
    if args.restart_from:
        import glob
        import re

        for cs in cross_scales:
            matches = sorted(
                glob.glob(f"{args.restart_from}/decay_R{R_tag}_cs{cs:.4e}_step*.npz")
            )
            if not matches:
                print(
                    f"no restart snapshot for cs={cs:.4e} in {args.restart_from} "
                    f"(expected decay_R{R_tag}_cs{cs:.4e}_step<n>.npz) — starting from zero"
                )
                continue
            path = matches[-1]
            m = re.search(r"_step(\d+)\.npz$", path)
            if not m:
                raise ValueError(f"cannot deduce start step from filename: {path}")
            restart_paths[cs] = (path, int(m.group(1)))
            print(f"restart cs={cs:.4e}: {path} (start_step={m.group(1)})")

    tasks = []
    for cs in cross_scales:
        task_kw = dict(kw)
        start_step = 0
        if cs in restart_paths:
            restart_path, start_step = restart_paths[cs]
            task_kw["restart_path"] = restart_path
            task_kw["start_step"] = start_step
        if args.save_snapshot_dir:
            final_step = start_step + args.n_steps
            task_kw["save_snapshot_path"] = (
                f"{args.save_snapshot_dir}/decay_R{R_tag}_cs{cs:.4e}_step{final_step:06d}.npz"
            )
        tasks.append((cs, task_kw))
    with mp.Pool(len(tasks)) as pool:
        results = dict(pool.map(_worker, tasks))

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    any_cs = cross_scales[0]
    print(
        f"\nN_A={results[any_cs]['n_A']}  N_B={results[any_cs]['n_B']}  "
        f"dt={results[any_cs]['dt']:.4e}"
    )
    amr_cols = args.max_level > 0
    hdr = (
        f"{'cross_scale':>12}  {'phi_rms_0':>10}  {'phi_rms_f':>10}  "
        f"{'H_0':>10}  {'H_f':>10}  {'lam_A_f':>8}  {'lam_B_f':>8}  "
        f"{'circ_B_0':>10}  {'circ_B_f':>10}"
        + (f"  {'n_leaves':>9}  {'lvl_max':>7}" if amr_cols else "")
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for cs in cross_scales:
        h = results[cs]
        line = (
            f"{cs:12.4e}  {h['phi_rms'][0]:10.4e}  {h['phi_rms'][-1]:10.4e}  "
            f"{h['H'][0]:10.4e}  {h['H'][-1]:10.4e}  {h['lam_A'][-1]:8.4f}  "
            f"{h['lam_B'][-1]:8.4f}  {h['circ_B'][0]:+10.4e}  {h['circ_B'][-1]:+10.4e}"
        )
        if amr_cols:
            line += f"  {h['n_leaves']:9d}  {h['level_max']:7d}"
        print(line)

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, cs in enumerate(cross_scales):
        h = results[cs]
        color = cmap(i / max(len(cross_scales) - 1, 1))
        axes[0].semilogy(h["t"], h["H"] + 1e-30, color=color, label=f"cs={cs:.2e}")
        axes[1].semilogy(
            h["t"], np.abs(h["circ_B"]) + 1e-30, color=color, label=f"cs={cs:.2e}"
        )
        axes[2].plot(h["t"], h["lam_A"], color=color, label=f"cs={cs:.2e}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("H (log)")
    axes[0].set_title("Total energy")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("|circ_B| (log)")
    axes[1].set_title("Ring B circulation")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("lam_A")
    axes[2].set_title("Ring A lam (resistance check)")
    for ax in axes:
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nplot saved -> {args.out}")


if __name__ == "__main__":
    main()
