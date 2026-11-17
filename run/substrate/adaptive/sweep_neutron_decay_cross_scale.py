"""
Adaptive neutron (Taylor spheromak) decay — cross_scale calibration sweep.

Same purpose as sweep_spheromak_decay_cross_scale.py (that script's minimal
8-tile Hopf ring, calibrated for the electron/proton knot model) but for the
much larger, smoothly-tapered Taylor spheromak IC used in
test_neutron_decay.py — the neutron model. Sequential (no multiprocessing —
each run is already large; unlike the ring IC there's no "protected tile"
bookkeeping to make workers independent, so a plain loop is simpler and the
memory footprint per run is bigger, making concurrent workers less
attractive anyway).

No ring-tile protection is needed anywhere here (AMR pre-adapt, organic
regrow, restart replay): the Taylor spheromak IC (init_spheromak_ic) sets
u/lam/phi purely from cell position, and phi_toroidal_circulation likewise
re-derives everything from the current leaf set on every call — there is no
fixed-index/discrete-loop-connectivity assumption to break, unlike
find_toroidal_ring/find_toroidal_spheromak's 8/4-tile pattern. That also
means re-seeding after AMR pre-adapt is trivial: just call
init_spheromak_ic() again on the refined mesh.

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_neutron_decay_cross_scale.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/sweep_neutron_decay_cross_scale.py \\
      --cross-scales 1e-4 3e-4 1e-3 --n-steps 20000 --max-level 1 --split-hi 0.02
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from run.substrate.adaptive.probe_decay_compare import plot_snapshots
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import (
    _compute_dt_l0,
    _replay_mesh,
)
from tst.substrate.adaptive import GrowingAMRSchedule
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_spheromak_ic,
    phi_toroidal_scan,
)

R_SCAN = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=80.0, help="domain sphere radius")
parser.add_argument(
    "--sph-frac",
    type=float,
    default=0.06,
    help="spheromak radius as fraction of R (scale = R * sph_frac)",
)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument(
    "--amplitude", type=float, default=1.0, help="phi pre-load amplitude"
)
parser.add_argument(
    "--sigma",
    type=float,
    default=None,
    help="Gaussian taper width in spheromak units (None=hard cutoff at R_sph)",
)
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument(
    "--lam-relax-rate",
    type=float,
    default=0.0,
    help="rate for step_lam_relax: dlam/dt = rate*(alpha*|phi| - lam)  (0=off, "
    "can be combined with --lam-mode dynamic)",
)
parser.add_argument("--lam-alpha", type=float, default=1.0)
parser.add_argument(
    "--gate",
    choices=("v1", "barrier"),
    default="barrier",
    help="v1: continuous always-fires step_cross. barrier: quantized "
    "all-or-nothing gate (doc/dst_next.md sec 8), ported to "
    "AdaptiveLattice::step_cross. Default m0 (1e-4) is the value calibrated "
    "on the adaptive Poynting-ring probe (see "
    "doc/mems/project_poynting_dipole_calibration.md) to actually let the "
    "gate fire beyond the immediate source.",
)
parser.add_argument(
    "--m0",
    type=float,
    default=1e-2,
    help="barrier gate threshold (pooled per-cell drive Sum W*max(-phi,0)); "
    "ignored for gate=v1",
)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument(
    "--max-level",
    type=int,
    default=0,
    help="AMR max refinement level (0 = AMR disabled, flat BCC lattice)",
)
parser.add_argument(
    "--criterion", choices=["relative_u", "absolute"], default="relative_u"
)
parser.add_argument("--split-hi", type=float, default=0.0)
parser.add_argument("--split-lo", type=float, default=0.0)
parser.add_argument("--split-ref", type=float, default=1e-6)
parser.add_argument("--amr-interval", type=int, default=1000)
parser.add_argument(
    "--amr-growing", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--amr-growing-start", type=int, default=10)
parser.add_argument("--lo", type=float, default=1e-4)
parser.add_argument("--hi", type=float, default=1e-2)
parser.add_argument("--n-values", type=int, default=8)
parser.add_argument(
    "--cross-scales",
    type=float,
    nargs="+",
    default=None,
    help="explicit cross_scale values (overrides --n-values log-spacing between --lo/--hi)",
)
parser.add_argument(
    "--out", type=str, default="tmp/neutron_decay_cross_scale_sweep.png"
)
parser.add_argument("--save-snapshot-dir", type=str, default=None)
parser.add_argument(
    "--plot-dir",
    type=str,
    default=None,
    help="if set, each cross_scale renders a field snapshot plot every "
    "--plot-every steps (plus t=0) into <dir>/R<R>_cs<cross_scale>/",
)
parser.add_argument("--plot-every", type=int, default=1000)
parser.add_argument(
    "--restart-from",
    type=str,
    default=None,
    help="directory of decay_neutron_R<R>_cs<cross_scale>_step<n>.npz "
    "snapshots (as written by --save-snapshot-dir) to resume from",
)


def _build_and_run(
    cross_scale,
    R,
    sph_frac,
    B0,
    lam_target,
    u_max,
    amplitude,
    sigma,
    lam_scale,
    lam_mode,
    lam_relax_rate,
    lam_alpha,
    sponge,
    damp_max,
    n_steps,
    report,
    gate="barrier",
    m0=1e-4,
    max_level=0,
    criterion="relative_u",
    split_hi=0.0,
    split_lo=0.0,
    split_ref=1e-6,
    amr_interval=1000,
    amr_growing=True,
    amr_growing_start=10,
    save_snapshot_path=None,
    plot_dir=None,
    plot_every=1000,
    restart_path=None,
    start_step=0,
    n_threads=10,
):
    """Build a fresh Taylor-spheromak neutron IC and run one cross_scale to n_steps.

    Deterministic given (R, sph_frac, B0, lam_target, u_max, amplitude,
    sigma) — init_spheromak_ic sets fields purely from cell position, so
    this is safe to call repeatedly (e.g. to re-seed after AMR pre-adapt)
    without any leaf-index bookkeeping.
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
        r_now = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge(
                target=0.0, damp_max=damp_max, r_all=r_now, r_inner=r_inner
            )
        )

    scale = R * sph_frac
    centre = np.zeros(3)

    def _seed_ic():
        """(Re-)write the Taylor spheromak IC onto the CURRENT leaf set.

        Purely position-derived (init_spheromak_ic reads cluster.leaf_pos
        fresh) — idempotent and safe to call again after the mesh changes,
        unlike the ring IC's tile-index-based winding.
        """
        return init_spheromak_ic(
            cl,
            R_sph=1.0,
            scale=scale,
            B0=B0,
            lam_target=lam_target,
            u_max=u_max,
            amplitude=amplitude,
            sigma=sigma,
        )

    if restart_path is None:
        b_max = _seed_ic()
    else:
        data = np.load(restart_path)
        if "R" in data.files and abs(float(data["R"]) - R) > 1e-9:
            raise ValueError(
                f"restart snapshot {restart_path} was built at R={float(data['R'])}, "
                f"but this run requested R={R} — refusing to load mismatched topology"
            )
        n_l0 = cl.n_leaves
        n_saved = int(data["pos"].shape[0])
        if n_saved != n_l0:
            _replay_mesh(alat, cl, data["key"], data["level"], amr_interval)
            _rebuild_sponge()
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
        b_max = float(np.abs(data["phi"]).max())

    dt = _compute_dt_l0(cl)

    cs_plot_dir = (
        f"{plot_dir}/R{int(round(R))}_cs{cross_scale:.4e}" if plot_dir else None
    )

    # ------------------------------------------------------------------
    # AMR — no geometric or tile restriction: split_criterion decides purely
    # from local phi/u dynamics; refinement anywhere, including the sphere
    # centre, is an emergent outcome.
    # ------------------------------------------------------------------
    amr_enabled = max_level > 0 and split_hi > 0.0
    amr_growing = amr_growing and restart_path is None
    amr_schedule = None
    sc = None
    if amr_enabled:
        sc = (
            _lat.SplitCriterion.relative_u(hi=split_hi, lo=split_lo, ref=split_ref)
            if criterion == "relative_u"
            else _lat.SplitCriterion.absolute(hi=split_hi, lo=split_lo)
        )

    has_fine = int(np.max(np.array(cl.leaf_level))) > 0
    ts = (
        _lat.TimeStepper.by_level(dt)
        if (amr_enabled or has_fine)
        else _lat.TimeStepper.fixed(dt)
    )
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    if amr_enabled and restart_path is None:
        # Bring the mesh to its organically-converged refinement structure
        # BEFORE the recorded run begins — see
        # sweep_spheromak_decay_cross_scale.py's identical block for the
        # full rationale (relative_u/absolute SplitCriterion needs live
        # criterion_max, populated only by stepping). No ring-tile masking
        # needed here — nothing is protected from splitting.
        PRE_ADAPT_MAX_ROUNDS = 50
        pre_info = _lat.DSTInfo()
        pre_info.max_level = max_level
        pre_info.split_interval = 1
        cl.info = pre_info
        cl.set_split_criterion(sc)
        n_pre_adapts = 0
        for _ in range(PRE_ADAPT_MAX_ROUNDS):
            evolver.step(ts)
            n_before = cl.n_leaves
            alat.adapt(0)
            n_pre_adapts += 1
            if cl.n_leaves == n_before:
                break
            _rebuild_sponge()
        else:
            print(
                f"pre-adapt IC: mesh still growing after {PRE_ADAPT_MAX_ROUNDS} rounds "
                f"({cl.n_leaves} leaves) — proceeding; organic regrow in the main "
                f"loop will continue extending it"
            )
        # Warm-up steps advanced the field state — re-seed the pristine IC
        # on the now-converged mesh.
        b_max = _seed_ic()
        dt = _compute_dt_l0(cl)

    if amr_enabled:
        final_split_interval = 1 if amr_growing else max(1, amr_interval)
        if restart_path is None and not amr_growing:
            # See sweep_spheromak_decay_cross_scale.py's identical block:
            # cl.adapt_calls (not exposed to Python, not reset by cl.info
            # reassignment) drifts out of sync with the main loop's
            # assumption that it starts at 0 after the pre-adapt loop's own
            # adapt() calls — pad it back to a multiple of
            # final_split_interval so the C++ split_interval cadence gate
            # stays aligned with when the main loop actually intends to
            # evaluate.
            cl.clear_split_criterion()
            cl.info = _lat.DSTInfo()
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

    cross_kwargs = {"gate": "barrier", "m0": m0} if gate == "barrier" else {}

    pos_now = np.array(cl.leaf_pos)
    r_cell = np.linalg.norm((pos_now - centre) / scale, axis=1)
    n_inner = int((r_cell <= 1.0).sum())
    phi_rms0 = float(np.sqrt(np.mean(np.array(cl.leaf_phi) ** 2)))
    lvl_now = int(np.max(np.array(cl.leaf_level)))
    print(
        f"  IC: cross_scale={cross_scale:.4e}  gate={gate}"
        + (f"  m0={m0:.2e}" if gate == "barrier" else "")
        + f"  N_total={cl.n_leaves}  "
        f"N_spheromak={n_inner} ({100 * n_inner / cl.n_leaves:.1f}%)  "
        f"R={R:.0f}  scale={scale:.1f}  B0_max={b_max:.3f}  "
        f"phi_rms={phi_rms0:.4e}  dt={dt:.4e}"
        + (f"  level_max={lvl_now}" if amr_enabled else ""),
        flush=True,
    )

    def plot_field_snapshot(step):
        snap = dict(
            label=f"R={int(round(R))} cs={cross_scale:.2e} step={step}",
            phi=np.array(cl.leaf_phi).T.copy(),
            u=np.array(cl.leaf_u).copy(),
            lam=np.array(cl.leaf_lam).copy(),
            pos=np.array(cl.leaf_pos).copy(),
            ring_r=scale,
        )
        gridsize = max(20, int(r_max * 1.3))
        plot_snapshots(
            [snap], plot_dir=cs_plot_dir, tag=f"step{step:06d}", gridsize=gridsize
        )

    n_rep = n_steps // report + 1
    t_hist = np.zeros(n_rep)
    H_hist = np.zeros(n_rep)
    phi_rms_h = np.zeros(n_rep)
    lam_mean_h = np.zeros(n_rep)
    lam_max_h = np.zeros(n_rep)
    lam_std_h = np.zeros(n_rep)
    circ_hist = np.zeros((n_rep, len(R_SCAN)))

    def record(si, step):
        phi_a = np.array(cl.leaf_phi)
        u_a = np.array(cl.leaf_u)
        lam_a = np.array(cl.leaf_lam)
        t_hist[si] = step * dt
        H_hist[si] = float(np.sum(u_a**2) + 0.5 * np.sum(phi_a**2))
        phi_rms_h[si] = float(np.sqrt(np.mean(phi_a**2)))
        lam_mean_h[si] = float(lam_a.mean())
        lam_max_h[si] = float(lam_a.max())
        lam_std_h[si] = float(lam_a.std())
        # Vectorized unique-face circulation scan — same math as the legacy
        # per-shell phi_toroidal_circulation loop (agrees to ~4e-13 on evolved
        # AMR meshes) at ~1/100th the cost; geometry is recomputed per record
        # so AMR mesh changes are picked up.
        mid, n_hat, phi_f = face_flux_geometry(cl, centre)
        circ_hist[si] = phi_toroidal_scan(mid, n_hat, phi_f, R_SCAN)

    si = 0
    record(0, start_step)
    if cs_plot_dir and plot_every > 0:
        plot_field_snapshot(start_step)
    for rel_step in range(1, n_steps + 1):
        step = start_step + rel_step
        evolver.step(ts)
        alat.step_sponge(0)
        if cross_scale > 0.0:
            alat.step_cross(0, dt, cross_scale, **cross_kwargs)
        if lam_mode == "dynamic":
            alat.step_lam(0, dt, lam_scale)
        if lam_relax_rate > 0.0:
            step_lam_relax(cl, dt, lam_relax_rate, lam_alpha)

        if amr_schedule is not None:
            if amr_schedule.due(rel_step):
                n_before = cl.n_leaves
                alat.adapt(0)
                if cl.n_leaves != n_before:
                    _rebuild_sponge()
        elif amr_enabled:
            n_before = cl.n_leaves
            alat.adapt(0)
            if cl.n_leaves != n_before:
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
            R=np.asarray(R, dtype=float),
            cross_scale=np.asarray(cross_scale, dtype=float),
        )

    return dict(
        t=t_hist[: si + 1],
        H=H_hist[: si + 1],
        phi_rms=phi_rms_h[: si + 1],
        lam_mean=lam_mean_h[: si + 1],
        lam_max=lam_max_h[: si + 1],
        lam_std=lam_std_h[: si + 1],
        circ=circ_hist[: si + 1],
        b_max=b_max,
        dt=dt,
        n_leaves=cl.n_leaves,
        level_max=int(np.max(np.array(cl.leaf_level))),
    )


def main():
    args = parser.parse_args()

    if args.cross_scales is not None:
        cross_scales = sorted(args.cross_scales)
    else:
        cross_scales = list(
            np.logspace(np.log10(args.lo), np.log10(args.hi), args.n_values)
        )

    kw = dict(
        R=args.R,
        sph_frac=args.sph_frac,
        B0=args.B0,
        lam_target=args.lam_target,
        u_max=args.u_max,
        amplitude=args.amplitude,
        sigma=args.sigma,
        lam_scale=args.lam_scale,
        lam_mode=args.lam_mode,
        lam_relax_rate=args.lam_relax_rate,
        lam_alpha=args.lam_alpha,
        sponge=args.sponge,
        damp_max=args.damp_max,
        n_steps=args.n_steps,
        report=args.report,
        gate=args.gate,
        m0=args.m0,
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
        f"R={args.R}  sph_frac={args.sph_frac}  scale={args.R * args.sph_frac:.1f}  "
        f"lam_target={args.lam_target}  lam_scale={args.lam_scale}  "
        f"lam_mode={args.lam_mode}  n_steps={args.n_steps}"
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
                glob.glob(
                    f"{args.restart_from}/decay_neutron_R{R_tag}_cs{cs:.4e}_step*.npz"
                )
            )
            if not matches:
                print(
                    f"no restart snapshot for cs={cs:.4e} in {args.restart_from} — "
                    f"starting from zero"
                )
                continue
            path = matches[-1]
            m = re.search(r"_step(\d+)\.npz$", path)
            if not m:
                raise ValueError(f"cannot deduce start step from filename: {path}")
            restart_paths[cs] = (path, int(m.group(1)))
            print(f"restart cs={cs:.4e}: {path} (start_step={m.group(1)})")

    results = {}
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
                f"{args.save_snapshot_dir}/decay_neutron_R{R_tag}_cs{cs:.4e}_step{final_step:06d}.npz"
            )
        print(f"\n=== cross_scale={cs:.4e} ===", flush=True)
        results[cs] = _build_and_run(cs, **task_kw)

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    amr_cols = args.max_level > 0
    r_ref_idx = len(R_SCAN) // 2
    r_ref = R_SCAN[r_ref_idx]
    hdr = (
        f"{'cross_scale':>12}  {'phi_rms_0':>10}  {'phi_rms_f':>10}  "
        f"{'H_0':>10}  {'H_f':>10}  {'lam_mean_f':>10}  {'lam_max_f':>10}  "
        f"{'circ_r' + str(r_ref) + '_0':>12}  {'circ_r' + str(r_ref) + '_f':>12}"
        + (f"  {'n_leaves':>9}  {'lvl_max':>7}" if amr_cols else "")
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for cs in cross_scales:
        h = results[cs]
        line = (
            f"{cs:12.4e}  {h['phi_rms'][0]:10.4e}  {h['phi_rms'][-1]:10.4e}  "
            f"{h['H'][0]:10.4e}  {h['H'][-1]:10.4e}  {h['lam_mean'][-1]:10.4f}  "
            f"{h['lam_max'][-1]:10.4f}  {h['circ'][0, r_ref_idx]:+12.4e}  "
            f"{h['circ'][-1, r_ref_idx]:+12.4e}"
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
            h["t"],
            np.abs(h["circ"][:, r_ref_idx]) + 1e-30,
            color=color,
            label=f"cs={cs:.2e}",
        )
        axes[2].plot(h["t"], h["lam_mean"], color=color, label=f"cs={cs:.2e}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("H (log)")
    axes[0].set_title("Total energy")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(f"|circ r={r_ref}| (log)")
    axes[1].set_title("Toroidal circulation")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("lam_mean")
    axes[2].set_title("Mean lam (resistance check)")
    for ax in axes:
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nplot saved -> {args.out}")


if __name__ == "__main__":
    main()
