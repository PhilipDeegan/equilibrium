"""
DST mirror instability — single-realisation run for direct comparison against
run/substrate/mirror_instability_phare.py.

Uses the same occupation-field MHD kernels as the Brazil-plot pipeline
(equilibrium/substrate/mhd.py) but instead of many random (beta_para, R)
draws pooled into a scatter plot, initialises every B-cell to the SAME
(beta_para, R) — the DST analogue of PHARE's homogeneous anisotropic IC —
and evolves it, dumping snapshots at IC / early / late checkpoints.

Domain: same solenoid-in-a-sphere setup as plot_brazil_instabilities.py
(SubstrateLattice + _solenoid_cells/_set_solenoid). There's no periodic-box
primitive at this layer, so "homogeneous" here means "same (beta_para, R)
everywhere among the B-cells", not a literal uniform field — B magnitude and
direction still vary with position inside the solenoid, same as any run
built on this lattice.

A small random perturbation seeds the instability (this model has no
intrinsic particle noise the way PHARE's kinetic protons do, so something
has to break the initial symmetry for growth to occur — PHARE gets this for
free from finite-N counting statistics).

Checkpoints are picked in DST's own natural step units, not matched 1:1 to
PHARE's final_time — DST's wave speed is ~280x PHARE's Alfven speed at the
Harris-sheet calibration (see project_harris_phare memory), so "IC / early /
late" is a structural correspondence (unrelaxed -> growing -> saturated),
not a literal time match.

--use-hall enables the electron-closure (Hall analogue) correction
(equilibrium/substrate/mhd.py step_hall), referenced against a ground-state
background field captured once via capture_ground_state right after
_set_solenoid -- see project_electron_closure_hall_analogue memory. The
ground state is the substrate's own base/rest configuration (the solenoid's
own field, present everywhere in the domain from the IC), not something
imposed from an exterior boundary -- no boundary-condition machinery is
involved. Off by default, since it changes the comparison's physics (a new
source term) rather than being silently blended into the existing
raw-kernel-set baseline.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/mirror_instability_dst.py
    PYTHONPATH=$PWD python3 run/substrate/mirror_instability_dst.py \\
        --beta-para 5 --aniso 0.4   # firehose-unstable instead of mirror
    PYTHONPATH=$PWD python3 run/substrate/mirror_instability_dst.py \\
        --use-hall --lam 2.0 --hall-amplitude 0.2
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.ab import _set_solenoid, _solenoid_cells
from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.mhd import (
    _SQ_AXIS_PAIRS,
    _b_vec,
    _b_vec_safe,
    _occ_vec,
    capture_ground_state,
    step_frozen_in,
    step_hall,
    step_occ_lorentz,
    step_occ_n,
    step_occ_stream,
)

# ---------------------------------------------------------------------------
# Homogeneous (beta_para, R) init, same construction as plot_brazil's
# _init_occ but with a fixed (beta_para, R) instead of a per-cell random draw.
# ---------------------------------------------------------------------------


def _init_occ_homogeneous(lat, b_cell_list, beta_para, aniso, noise, rng):
    """
    Every B-cell starts at the same (beta_para, R=aniso) point.
    occ_mag = 0.5*|B| (sub-Alfvenic, matches plot_brazil's convention).
    noise: fractional random perturbation on |occ| per cell (symmetry seed).
    """
    occ = np.zeros((lat.N, 14))
    occ_n = np.zeros(lat.N)
    para_frac = 1.0 / (1.0 + 2.0 * aniso)

    for i in b_cell_list:
        bv = _b_vec(lat.phi, lat.nb, i)
        if bv is None:
            continue
        B_sq = float(np.dot(bv, bv))
        B_mag = float(np.sqrt(B_sq))
        if B_mag < 1e-10:
            continue
        b_hat = bv / B_mag

        occ_mag = 0.5 * B_mag * (1.0 + noise * float(rng.uniform(-1.0, 1.0)))
        v_para_mag = occ_mag * float(np.sqrt(para_frac))
        v_perp_mag = occ_mag * float(np.sqrt(1.0 - para_frac))

        v_para = v_para_mag * b_hat

        e = rng.standard_normal(3)
        e -= np.dot(e, b_hat) * b_hat
        en = float(np.linalg.norm(e))
        if en < 1e-10:
            continue
        v_perp = (v_perp_mag / en) * e

        v = v_para + v_perp
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            occ[i, fp] = max(v[ax], 0.0)
            occ[i, fm] = max(-v[ax], 0.0)

        # back-compute n so beta_para = T_para/(n*|B|^2) matches the target
        occ_n[i] = max(0.25 * para_frac / beta_para, 1e-6)

    return occ, occ_n


def _local_beta_R(lat, occ, occ_n, b_cell_list):
    """Per-cell (beta_para, beta_perp) arrays, aligned with b_cell_list order."""
    bp = np.full(len(b_cell_list), np.nan)
    bq = np.full(len(b_cell_list), np.nan)
    for idx, i in enumerate(b_cell_list):
        curl_a = _b_vec_safe(lat, i)
        if curl_a is None:
            continue
        curl_a_sq = float(np.dot(curl_a, curl_a))
        if curl_a_sq < 1e-30:
            continue
        a_hat = curl_a / np.sqrt(curl_a_sq)
        v = _occ_vec(occ[i])
        if not np.all(np.isfinite(v)):
            continue
        n = float(occ_n[i])
        if n < 1e-30:
            continue
        T_para = float(np.dot(v, a_hat)) ** 2
        T_perp = float(np.dot(v, v)) - T_para
        denom = n * curl_a_sq
        bp[idx] = T_para / denom
        bq[idx] = 0.5 * T_perp / denom
    return bp, bq


def _snapshot(lat, occ, occ_n, b_cell_list, step):
    bp, bq = _local_beta_R(lat, occ, occ_n, b_cell_list)
    return dict(
        step=step,
        phi=lat.phi.copy(),
        occ=occ.copy(),
        occ_n=occ_n.copy(),
        beta_para=bp,
        beta_perp=bq,
    )


# ---------------------------------------------------------------------------


def run(args):
    lat = SubstrateLattice(radius_cells=args.radius)
    _set_solenoid(lat, _solenoid_cells(lat, args.sol_radius), flux_quanta=args.flux)
    dt = lat.stable_dt()

    b_cell_list = [
        i
        for i in range(lat.N)
        if (bv := _b_vec(lat.phi, lat.nb, i)) is not None and np.dot(bv, bv) > 1e-24
    ]
    print(f"N={lat.N}  B-cells={len(b_cell_list)}  dt={dt:.6f}")

    if args.use_hall:
        # capture_ground_state must run before any noise/evolution -- it's
        # the substrate's own base/rest state (see
        # equilibrium/substrate/mhd.py step_hall docstring), not a snapshot
        # of the perturbed IC below.
        capture_ground_state(lat)
        lat.lam[b_cell_list] = args.lam
        print(
            f"electron-closure (Hall analogue) ON: lam={args.lam} "
            f"amplitude={args.hall_amplitude}"
        )

    rng = np.random.default_rng(args.seed)
    occ, occ_n = _init_occ_homogeneous(
        lat, b_cell_list, args.beta_para, args.aniso, args.noise, rng
    )

    R_mirror = 1.0 + 1.0 / args.beta_para
    R_fh = (1.0 - 1.0 / args.beta_para) if args.beta_para > 1.0 else float("nan")
    print(
        f"beta_para={args.beta_para}  R={args.aniso}  "
        f"R_mirror={R_mirror:.3f}  R_firehose={R_fh:.3f}"
    )

    checkpoints = {}
    checkpoints["ic"] = _snapshot(lat, occ, occ_n, b_cell_list, 0)

    step = 0
    for target, label in ((args.n_early, "early"), (args.n_late, "late")):
        while step < target:
            step_frozen_in(lat, occ, dt, b_cell_list)
            step_occ_lorentz(lat, occ, dt, b_cell_list)
            step_occ_stream(lat, occ, dt, b_cell_list)
            step_occ_n(lat, occ, occ_n, dt, b_cell_list)
            if args.use_hall:
                step_hall(lat, dt, amplitude=args.hall_amplitude, cells=b_cell_list)
            step += 1
        checkpoints[label] = _snapshot(lat, occ, occ_n, b_cell_list, step)
        bp, bq = checkpoints[label]["beta_para"], checkpoints[label]["beta_perp"]
        R = bq / bp
        finite = np.isfinite(R)
        print(
            f"  {label:5s} step={step:6d}  "
            f"beta_para=[{np.nanmin(bp):.3e},{np.nanmax(bp):.3e}]  "
            f"R=[{np.nanmin(R[finite]):.3e},{np.nanmax(R[finite]):.3e}]  "
            f"mean(R)={np.nanmean(R[finite]):.3f}"
        )
        if np.nanmin(bp) < 1e-3:
            print(
                "    NOTE: some cells have beta_para -> 0 (T_para crashing "
                "toward zero as they mirror-trap) — R=beta_perp/beta_para "
                "diverges there as a ratio artifact, not unbounded physical "
                "growth. Prefer --n-late below ~180 for this IC to stay in "
                "the smooth growth regime, or compare beta_para/beta_perp "
                "separately instead of R past this point."
            )

    out = dict(
        pos=lat.pos[b_cell_list],
        b_cell_list=np.array(b_cell_list),
        beta_para_ic=np.float32(args.beta_para),
        aniso_ic=np.float32(args.aniso),
        radius=np.float32(args.radius),
        sol_radius=np.float32(args.sol_radius),
        dt=np.float32(dt),
    )
    for label, snap in checkpoints.items():
        out[f"{label}_step"] = snap["step"]
        out[f"{label}_phi"] = snap["phi"][b_cell_list]
        out[f"{label}_occ"] = snap["occ"][b_cell_list]
        out[f"{label}_occ_n"] = snap["occ_n"][b_cell_list]
        out[f"{label}_beta_para"] = snap["beta_para"]
        out[f"{label}_beta_perp"] = snap["beta_perp"]

    np.savez_compressed(args.out, **out)
    print(f"Saved -> {args.out}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--beta-para", type=float, default=3.0)
    parser.add_argument(
        "--aniso",
        type=float,
        default=2.5,
        help="R = T_perp/T_para (2.5 -> mirror-unstable, "
        "matches mirror_instability_phare.py default)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="fractional per-cell |occ| perturbation (symmetry seed)",
    )
    parser.add_argument("--radius", type=float, default=30.0)
    parser.add_argument("--sol-radius", type=float, default=6.0)
    parser.add_argument("--flux", type=int, default=1)
    parser.add_argument(
        "--n-early",
        type=int,
        default=20,
        help="step count for the 'a few steps' checkpoint",
    )
    parser.add_argument(
        "--n-late",
        type=int,
        default=150,
        help="step count for the final checkpoint (default 150: "
        "smooth mirror growth phase, before per-cell "
        "beta_para->0 makes R a ratio artifact past ~180)",
    )
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--out", default="mirror_instability_dst.npz")
    parser.add_argument(
        "--use-hall",
        action="store_true",
        help="enable the electron-closure (Hall analogue) "
        "correction -- step_hall, referenced against "
        "the ground-state background field captured "
        "via capture_ground_state. Off by default: "
        "this changes the comparison's physics (a new "
        "source term), so it's opt-in rather than "
        "silently blended into the existing "
        "raw-kernel-set baseline (see "
        "feedback_no_instab_limit_hack memory for why "
        "new mechanisms stay opt-in here).",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="uniform lat.lam value set on B-cells when "
        "--use-hall is on (chi=lam/(1+lam) weights "
        "step_hall; lam=1.0 matches the floor value "
        "beta_para/beta_perp already assume elsewhere "
        "when lam is otherwise 0)",
    )
    parser.add_argument(
        "--hall-amplitude",
        type=float,
        default=0.1,
        help="step_hall coupling strength (empirical, "
        "DST-native -- not a physical 1/(n*e) Hall "
        "coefficient)",
    )
    args = parser.parse_args()
    run(args)
