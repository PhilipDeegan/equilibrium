"""
Refined neutron/proton decay — coarse base grid sized to contain the real
Bohr radius, with a deep, narrow telescope of AMR refinement concentrated at
the origin to resolve the fm-scale neutron/proton knot.

This is sweep_neutron_decay_cross_scale.py with the IC-construction section
replaced: instead of sizing the spheromak as scale = R * sph_frac (uniform
resolution everywhere), the knot's real physical scale is fixed by a
real-world unit calibration, and only a small region around the origin is
refined down to that resolution (refine_region_to_convergence, a fixed
deterministic telescope built once at IC time -- NOT the organic/criterion-
driven growing AMR the original script uses). Everything else (cross_scale
handling, gate/m0, lam dynamics, sponge BC, snapshot restart, the report
table, the H/circ/lam summary plot, and the optional --plot-dir per-step
field snapshots) is unchanged from the original.

Real-world calibration (2026-07-18, recalibrated 2026-07-19 for the ratio-3
Odd27 split -- see SPLIT_RATIO):
  Fit 2*a0 (Bohr-orbit diameter, 1.05835e-10 m) inside 90% of the R domain's
  diameter:
    D_domain = 2*a0 / 0.90 = 1.17595e-10 m  ->  R_domain = 5.8797e-11 m
    1 sim-unit (level-0 cell, at R=100) = 5.8797e-13 m
  Neutron diameter ~1.6e-15 m needs >=8 cells across it to clear the BCC
  dispersion anisotropy threshold (project_bcc_dispersion_anisotropy /
  project_step_cross_v2's minor-width confound). Under Odd27 (ratio-3,
  cell size shrinks 3x per level instead of 2x), max_level=8 already gives
  ~17.9 cells across the neutron diameter (L7 only gives ~6.0 -- not quite
  enough; L8 clears the threshold with a comfortable margin, similar in
  spirit to the old ratio-2 calibration's L12~11.2). L12 is no longer
  needed at all under this ratio -- it would now be a ~54x-oversized margin.

The refined region's radius shrinks by SPLIT_RATIO at every successive
level so cell COUNT added per level stays roughly constant (density grows
~27x per level from finer packing under Odd27, radius^3 shrinks ~27x from
the SPLIT_RATIO-per-level shrink -- they cancel).

Run:
  PYTHONPATH=$PWD python3 -u tst/substrate/adaptive/refined_neutron_decay.py \\
      --R 100 --max-level 8 --n-steps 50 --cross-scales 1e-3 --m0 2.49
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from tst.substrate.adaptive.probe_decay_compare import plot_snapshots
from tst.substrate.adaptive.sweep_spheromak_decay_cross_scale import (
    _compute_dt_l0,
    _replay_mesh,
)

from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from tst.substrate.adaptive import refine_region_to_convergence
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    face_flux_geometry,
    phi_toroidal_scan,
)


def init_spheromak_ic_refinable(
    cluster,
    *,
    R_sph=1.0,
    scale=1.5,
    B0=1.0,
    lam_target=0.1,
    u_max=1.0,
    amplitude=0.0,
    cx=0.0,
    cy=0.0,
    cz=0.0,
    sigma=None,
):
    """Level-aware variant of adaptive_flux_tube.init_spheromak_ic.

    The original scales phi's face-midpoint sample by a FIXED, level-0-sized
    _FACE_HOP offset regardless of the cell's actual size -- fine for a
    uniform mesh, but on a genuinely multi-level refined mesh (leaf_level
    varying a lot) that offset wildly overshoots the true local neighbourhood
    for deep cells, silently zeroing phi everywhere (found 2026-07-18: a
    560-cell inner region at level 14 seeded phi_rms=0.0 exactly). Fix: scale
    the hop offset by each cell's own actual size (3**-level, since cell size
    shrinks by ratio-3 per level under the Odd27 split model). u/lam are
    unaffected by this bug (cell-centre evaluation only).
    """
    from scipy.special import spherical_jn

    from equilibrium.substrate.discrete import LAM_MAX

    centre = np.array([cx, cy, cz])
    k_ = 4.493 / R_sph

    def _B(pos):
        p = (pos - centre) / scale
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        B = np.zeros_like(p)
        mask = (r <= R_sph) & (r > 0)
        rm = r[mask]
        cf = B0 * spherical_jn(1, k_ * rm) / rm
        if sigma is not None:
            cf *= np.exp(-0.5 * (rm / sigma) ** 2)
        B[mask, 0] = cf * (z[mask] * x[mask] / rm - y[mask])
        B[mask, 1] = cf * (z[mask] * y[mask] / rm + x[mask])
        B[mask, 2] = cf * (-(x[mask] ** 2 + y[mask] ** 2) / rm)
        return B

    pos = np.array(cluster.leaf_pos)
    N = pos.shape[0]
    levels = np.array(cluster.leaf_level)
    cell_size = 1.0 / (
        SPLIT_RATIO**levels
    )  # d1=1 at level 0, shrinks by SPLIT_RATIO per level

    B_c = _B(pos)
    b_mag = np.linalg.norm(B_c, axis=1)
    b_max = float(b_mag.max()) or 1.0
    lam = np.minimum(lam_target * b_mag / b_max, LAM_MAX)
    cluster.set_leaf_lam(lam)
    cluster.set_leaf_u(u_max * b_mag / b_max)

    r_cell = np.linalg.norm((pos - centre) / scale, axis=1)
    n_inner = int((r_cell <= R_sph).sum())
    print(
        f"  init_spheromak_ic_refinable: scale={scale:.4e}  R_sph={R_sph}  sigma={sigma}"
        f"  inner_cells={n_inner}/{N}  ({100 * n_inner / N:.1f}%)",
        flush=True,
    )

    if amplitude > 0.0:
        hop = (
            _FACE_HOP[None, :, :] * cell_size[:, None, None]
        )  # (N, 14, 3), per-cell scaled
        midpoints = pos[:, None, :] + 0.5 * hop
        B_mid = _B(midpoints.reshape(-1, 3)).reshape(N, 14, 3)
        phi = np.einsum("jkd,kd->kj", B_mid, _FACE_NORM)
        phi_max = float(np.abs(phi).max()) or 1.0
        phi *= amplitude / phi_max
        nbr_r = np.linalg.norm((pos[:, None, :] + hop - centre) / scale, axis=2)
        phi[(nbr_r > R_sph).T] = 0.0
        phi[:, r_cell > R_sph] = 0.0
        cluster.set_leaf_phi(np.ascontiguousarray(phi))
        phi_rb = np.array(cluster.leaf_phi)
        print(
            f"  [post-store] phi_rms={float(np.sqrt(np.mean(phi_rb**2))):.4e}  "
            f"phi_nonzero_cells={int(np.count_nonzero(np.abs(phi_rb).max(axis=0) > 1e-15))}",
            flush=True,
        )

    return b_max


# ---------------------------------------------------------------------------
# Real-world calibration (see module docstring)
# ---------------------------------------------------------------------------
SPLIT_RATIO = 3.0  # Odd27 refinement ratio: cell size shrinks by this factor per level
A0_M = 5.29177210903e-11  # Bohr radius, metres
BC_MARGIN = 0.90  # usable fraction of domain diameter after BC/sponge margin
R_DOMAIN_M = A0_M / BC_MARGIN  # = D_domain/2, since D_domain = 2*a0/margin
NEUTRON_RADIUS_M = 0.8e-15  # ~half of the 1.6 fm neutron diameter

R_SCAN = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--R", type=float, default=100.0, help="coarse (level-0) domain radius"
)
parser.add_argument(
    "--max-level",
    type=int,
    default=8,
    help="deepest refinement level for the origin telescope "
    "(ratio-3/Odd27: L8 gives ~17.9 cells across the neutron diameter, clearing the >=8-cell "
    "anisotropy threshold; L12 is no longer needed at this ratio, see module docstring)",
)
parser.add_argument(
    "--region-r0",
    type=float,
    default=5.0,
    help="level-1 refinement radius, in level-0 (coarse) sim-units; shrinks by SPLIT_RATIO each "
    "level after (default gives margin over the neutron's own scale at R=100/L8)",
)
parser.add_argument(
    "--scale-override",
    type=float,
    default=None,
    help="bypass the real-world neutron/Bohr-radius calibration and use a synthetic "
    "spheromak radius (in sim-units), interpreted as the scale AT --scale-ref-level "
    "(see below) and automatically rescaled by 3**(scale_ref_level - max_level) for "
    "whatever --max-level this run actually uses (ratio-3/Odd27 telescope). This holds "
    "BOTH the telescope-reach margin (finest_region_r / scale) AND the cells-across-"
    "diameter ratio invariant across --max-level, independent of R -- so e.g. testing "
    "at L2 with --scale-ref-level 8 automatically uses an object 3**6=729x bigger "
    "than at L8, giving the exact same relative resolution/detail, just cheaper to "
    "run. Ignored (real calibration used) if unset.",
)
parser.add_argument(
    "--scale-ref-level",
    type=int,
    default=8,
    help="the --max-level that --scale-override's value is calibrated at (default 8, "
    "matching this script's real max_level default); ignored if --scale-override unset",
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
    "all-or-nothing gate (doc/dst_next.md sec 8).",
)
parser.add_argument(
    "--m0",
    type=float,
    default=2.49,
    help="barrier gate threshold (pooled per-cell drive Sum W*max(-phi,0)); "
    "ignored for gate=v1",
)
parser.add_argument(
    "--eta-h",
    type=float,
    default=0.0,
    help="hyper-resistivity coefficient (0=off, exact no-op): k^4-selective damping "
    "applied only at AMR coarse-fine boundary cells via "
    "step_boundary_hyper_resistivity_all, once per coarse step after evolver.step(ts). "
    "See project_odd27_amr_junction_leak memory / CLAWD Part 5.",
)
parser.add_argument(
    "--eta-h-imex",
    action="store_true",
    help="use the diagonal-implicit (IMEX Euler) treatment of the hyper-resistivity "
    "term instead of the pure-explicit one -- see CLAWD Part 8. No effect if --eta-h 0.",
)
parser.add_argument(
    "--eta-h-per-level",
    type=float,
    nargs="+",
    default=None,
    help="direct per-level hyper-resistivity coefficients (eta_h_by_level[0], [1], ...), "
    "no base+exponent formula -- overrides --eta-h/--eta-h-imex's ratio scaling entirely "
    "(uses step_boundary_hyper_resistivity_per_level_all, always IMEX). A single flat "
    "--eta-h cannot serve shallow and deep boundary levels at once (useful ranges differ "
    "by ~ratio^(4*max_level), see project_odd27_amr_junction_leak Part 10) -- this lets "
    "each level be tuned independently. Levels beyond the list get eta_h=0.",
)
parser.add_argument(
    "--eta-h-ratio",
    type=float,
    default=None,
    help="dimensionless target ratio s = dt*eta_h_level*D_i (D_i = IMEX diagonal "
    "coefficient, ~1/L^4) applied uniformly at every boundary level via "
    "step_boundary_hyper_resistivity_ratio_all -- s directly controls how much of a "
    "boundary cell's own value gets replaced by its neighbour-geometric-extrapolation "
    "each step (s=0 no-op, s->inf full replacement), independent of dt/level since D_i "
    "already carries the 1/L^4 scaling. Takes priority over --eta-h-per-level/--eta-h if "
    "given. See project_odd27_amr_junction_leak Part 10.",
)
parser.add_argument(
    "--n-steps", type=int, default=50, help="coarse (level-0) timesteps"
)
parser.add_argument("--report", type=int, default=5)
parser.add_argument("--sponge", type=float, default=0.10)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument(
    "--bytes-per-cell",
    type=float,
    default=229.0,
    help="measured SoA footprint per leaf cell (cluster.hpp CellSlab layout) -- reporting only",
)
parser.add_argument(
    "--cross-scales",
    type=float,
    nargs="+",
    default=[1e-3],
    help="explicit cross_scale values",
)
parser.add_argument("--out", type=str, default="tmp/refined_neutron_decay_sweep.png")
parser.add_argument("--save-snapshot-dir", type=str, default=None)
parser.add_argument(
    "--plot-dir",
    type=str,
    default=None,
    help="if set, each cross_scale renders a field snapshot plot every "
    "--plot-every steps (plus t=0) into <dir>/R<R>_cs<cross_scale>/",
)
parser.add_argument("--plot-every", type=int, default=10)
parser.add_argument(
    "--restart-from",
    type=str,
    default=None,
    help="directory of decay_refined_R<R>_cs<cross_scale>_step<n>.npz "
    "snapshots (as written by --save-snapshot-dir) to resume from",
)


def _build_telescope(alat, cl, region_r0, max_level):
    """Deterministic, fixed refinement telescope at the origin -- NOT the
    organic/criterion-driven growing AMR the original sweep script uses.
    Radius halves each level so added cell count stays roughly constant
    per level (empirically confirmed, see module docstring)."""
    cl.info.split_hi = 0.5
    cl.info.split_lo = 0.0
    for lvl in range(1, max_level + 1):
        radius = region_r0 / (SPLIT_RATIO ** (lvl - 1))

        def _region(cl, radius=radius):
            r = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
            return r < radius

        refine_region_to_convergence(alat, cl, _region, stage_max_level=lvl)
    cl.set_criterion_max(np.full(cl.n_leaves, np.nan))  # clear telescope criterion
    cl.clear_split_criterion()
    cl.info = _lat.DSTInfo()  # no organic regrowth during the run


def _build_and_run(
    cross_scale,
    R,
    region_r0,
    max_level,
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
    m0=2.49,
    eta_h=0.0,
    eta_h_imex=False,
    eta_h_per_level=None,
    eta_h_ratio=None,
    scale_override=None,
    scale_ref_level=8,
    save_snapshot_path=None,
    plot_dir=None,
    plot_every=10,
    restart_path=None,
    start_step=0,
    n_threads=10,
):
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

    centre = np.zeros(3)
    sim_unit_m = R_DOMAIN_M / R
    finest_region_r = (
        region_r0 / (SPLIT_RATIO ** (max_level - 1)) if max_level > 0 else r_max
    )
    if scale_override is not None:
        scale = scale_override * SPLIT_RATIO ** (scale_ref_level - max_level)
        print(
            f"calibration: SCALE OVERRIDE active -- scale_ref={scale_override:.4e} "
            f"@ L{scale_ref_level}  ->  scale={scale:.4e} sim-units @ L{max_level} "
            f"(real-world neutron/Bohr-radius calibration bypassed; this run is a pure "
            f"numerical stability probe, not a calibrated physical result)",
            flush=True,
        )
    else:
        scale = NEUTRON_RADIUS_M / sim_unit_m
        print(
            f"calibration: 1 sim-unit={sim_unit_m:.4e} m  neutron scale={scale:.4e} sim-units  "
            f"(a0={A0_M:.4e} m, margin={BC_MARGIN})",
            flush=True,
        )
    if max_level > 0 and finest_region_r < scale:
        print(
            f"WARNING: finest refined region ({finest_region_r:.4e}) is SMALLER than "
            f"the neutron scale ({scale:.4e}) -- IC will not be fully resolved. "
            f"Increase --region-r0.",
            flush=True,
        )

    def _seed_ic():
        return init_spheromak_ic_refinable(
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
        if max_level > 0:
            _build_telescope(alat, cl, region_r0, max_level)
            _rebuild_sponge()
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
            _replay_mesh(alat, cl, data["key"], data["level"], 1)
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

    has_fine = int(np.max(np.array(cl.leaf_level))) > 0
    ts = _lat.TimeStepper.by_level(dt) if has_fine else _lat.TimeStepper.fixed(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

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
        f"R={R:.0f}  scale={scale:.4e}  B0_max={b_max:.3f}  "
        f"phi_rms={phi_rms0:.4e}  dt={dt:.4e}  level_max={lvl_now}  "
        f"est_mem={cl.n_leaves * 229 / 1e9:.4f} GB",
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
        gridsize = max(20, int(scale * 6))
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
        mid, n_hat, phi_f = face_flux_geometry(cl, centre)
        circ_hist[si] = phi_toroidal_scan(mid, n_hat, phi_f, R_SCAN)

    si = 0
    record(0, start_step)
    if cs_plot_dir and plot_every > 0:
        plot_field_snapshot(start_step)
    for rel_step in range(1, n_steps + 1):
        step = start_step + rel_step
        evolver.step(ts)
        if eta_h_ratio is not None:
            alat.step_boundary_hyper_resistivity_ratio_all([eta_h_ratio] * 16)
        elif eta_h_per_level is not None:
            alat.step_boundary_hyper_resistivity_per_level_all(
                dt, eta_h_per_level, imex=True
            )
        elif eta_h != 0.0:
            alat.step_boundary_hyper_resistivity_all(dt, eta_h, imex=eta_h_imex)
        alat.step_sponge(0)
        if cross_scale > 0.0:
            alat.step_cross(0, dt, cross_scale, **cross_kwargs)
        if lam_mode == "dynamic":
            alat.step_lam(0, dt, lam_scale)
        if lam_relax_rate > 0.0:
            step_lam_relax(cl, dt, lam_relax_rate, lam_alpha)

        if rel_step % report == 0:
            si += 1
            record(si, step)
            print(
                f"  step={step:6d}  H={H_hist[si]:.6e}  H/H0={H_hist[si] / H_hist[0]:.4f}",
                flush=True,
            )
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

    cross_scales = sorted(args.cross_scales)

    kw = dict(
        R=args.R,
        region_r0=args.region_r0,
        max_level=args.max_level,
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
        eta_h=args.eta_h,
        eta_h_imex=args.eta_h_imex,
        eta_h_per_level=args.eta_h_per_level,
        eta_h_ratio=args.eta_h_ratio,
        scale_override=args.scale_override,
        scale_ref_level=args.scale_ref_level,
        plot_dir=args.plot_dir,
        plot_every=args.plot_every,
        n_threads=args.n_threads,
    )

    print(
        f"R={args.R}  max_level={args.max_level}  region_r0={args.region_r0}  "
        f"lam_target={args.lam_target}  lam_scale={args.lam_scale}  "
        f"lam_mode={args.lam_mode}  n_steps={args.n_steps}",
        flush=True,
    )
    print(
        f"cross_scales ({len(cross_scales)}): "
        + ", ".join(f"{c:.4e}" for c in cross_scales),
        flush=True,
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
                    f"{args.restart_from}/decay_refined_R{R_tag}_cs{cs:.4e}_step*.npz"
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
                f"{args.save_snapshot_dir}/decay_refined_R{R_tag}_cs{cs:.4e}_step{final_step:06d}.npz"
            )
        print(f"\n=== cross_scale={cs:.4e} ===", flush=True)
        results[cs] = _build_and_run(cs, **task_kw)

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    r_ref_idx = len(R_SCAN) // 2
    r_ref = R_SCAN[r_ref_idx]
    hdr = (
        f"{'cross_scale':>12}  {'phi_rms_0':>10}  {'phi_rms_f':>10}  "
        f"{'H_0':>10}  {'H_f':>10}  {'lam_mean_f':>10}  {'lam_max_f':>10}  "
        f"{'circ_r' + str(r_ref) + '_0':>12}  {'circ_r' + str(r_ref) + '_f':>12}  "
        f"{'n_leaves':>9}  {'lvl_max':>7}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for cs in cross_scales:
        h = results[cs]
        line = (
            f"{cs:12.4e}  {h['phi_rms'][0]:10.4e}  {h['phi_rms'][-1]:10.4e}  "
            f"{h['H'][0]:10.4e}  {h['H'][-1]:10.4e}  {h['lam_mean'][-1]:10.4f}  "
            f"{h['lam_max'][-1]:10.4f}  {h['circ'][0, r_ref_idx]:+12.4e}  "
            f"{h['circ'][-1, r_ref_idx]:+12.4e}  {h['n_leaves']:9d}  {h['level_max']:7d}"
        )
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
