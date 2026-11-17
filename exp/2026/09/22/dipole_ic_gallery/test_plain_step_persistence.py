"""Persistence test under PLAIN reversible wave dynamics ONLY -- step_cross
NEVER called, not even once.

Philip's correction (2026-09-22, in conversation): step_cross is
specifically the translation/inertia mechanism (doc/dst.md's "Through-cell
crossing" section: it exists so a winding can physically translate through
the lattice -- nothing more general), and it's explicitly unconfirmed/
provisional ontology (feedback_model_structure_provisional.md), not a
general-purpose decay test. The general "does this structure persist"
question should be asked with the base reversible wave equation alone,
matching the convention exp/2026/08/17/12_25_curl_full_dipole_ic_rebuild/
probe_pol_tor_lockstep_convergence.py already used ("evolver.step(ts) --
pure wave dynamics, NO step_cross").

project_spheromak_decay.md's own decay demonstration (circ_B: -1.447 -> ~0)
was produced BY applying step_cross/cross_scale ("via step_cross topology
opening", its own stated goal) -- never shown to decay under plain step()
alone. This script checks whether that's also true (or not) for today's
candidate constructions.

Five constructions, same R_DOMAIN=80/R_sph=1.0/scale=50.0 convention as the
rest of exp/2026/09/22/dipole_ic_gallery/:
  1. init_electron_dipole_ic_vecpot  -- re-confirm/refute "exact fixed point"
  2. init_electron_dipole_ic (non-vecpot) -- circulation-free, open-tail candidate
  3. seed_dipole_ic (dipole_mode_ic.py) -- two-component (u,phi) standing wave
  4. plain 8-tile ring (find_toroidal_ring mode="minimal" + init_toroidal_winding)
  5. init_spheromak_ic -- known-decaying reference (decays under step_cross;
     does it also decay under plain step() alone?)

Per construction, tracked over N_STEPS of plain evolver.step(ts):
  - total field energy H(t) = sum(u^2) + 0.5*sum(phi^2) (conservation check)
  - macro phi_ring_circulation on a fixed ring (own centreline for the
    spherical constructions, the object's own ring for the plain 8-tile case)
  - radial energy-density profile shape at t=0, mid-run, and end-of-run
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/deegan/git/eq")
sys.path.insert(0, _HERE)

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_all_dipole_ics as gallery
from plot_per_ic_diagnostics import radial_profile
from equilibrium import lattice as _lat
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_electron_dipole_ic,
    init_electron_dipole_ic_vecpot,
    init_spheromak_ic,
    init_toroidal_winding,
    phi_ring_circulation,
)

N_STEPS = 300
REPORT_EVERY = 50
RING_HOPS = 27  # matches measure_ring_circulation.py's convention


def total_energy(cl):
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2))


def _seed_vecpot(cl, pos, cell_size):
    init_electron_dipole_ic_vecpot(cl, R_sph=gallery.R_SPH, scale=gallery.SCALE, amplitude=1.0)
    return None


def _seed_nonvecpot(cl, pos, cell_size):
    init_electron_dipole_ic(cl, R_sph=gallery.R_SPH, scale=gallery.SCALE, amplitude=1.0)
    return None


def _seed_spheromak(cl, pos, cell_size):
    init_spheromak_ic(cl, R_sph=gallery.R_SPH, scale=gallery.SCALE, amplitude=1.0)
    return None


def _seed_dipole_mode(cl, pos, cell_size):
    seed_dipole_ic(
        cl, pos, cell_size, gallery.SCALE, gallery._FACE_HOP, gallery._FACE_NORM,
        r_sph=gallery.R_SPH, amplitude=1.0,
    )
    return None


def _seed_plain_ring(cl, pos, cell_size):
    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="minimal")
    if ring_js is None:
        raise RuntimeError("plain 8-tile ring not found")
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0)
    return (ring_js, ring_faces)


CONSTRUCTIONS = [
    ("init_electron_dipole_ic_vecpot", _seed_vecpot),
    ("init_electron_dipole_ic (non-vecpot)", _seed_nonvecpot),
    ("seed_dipole_ic (dipole_mode_ic.py)", _seed_dipole_mode),
    ("plain 8-tile ring (find_toroidal_ring+init_toroidal_winding)", _seed_plain_ring),
    ("init_spheromak_ic", _seed_spheromak),
]


def get_measurement_ring(cl):
    return find_toroidal_ring(cl, axis="z", mode="square", hops=RING_HOPS)


def shape_drift(r0, p0, r1, p1, r_max):
    """Max relative deviation of the radial profile at bins where the t=0
    profile is at least 1% of its own peak (ignores empty/noise-floor bins)."""
    valid = (~np.isnan(p0)) & (~np.isnan(p1)) & (r0 <= r_max)
    if not valid.any():
        return float("nan")
    peak = np.nanmax(p0[valid])
    sig = valid & (p0 > 0.01 * peak)
    if not sig.any():
        return float("nan")
    return float(np.max(np.abs(p1[sig] / p0[sig] - 1.0)))


def run_one(name, seed_fn):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    alat, cl = gallery.build_cluster()
    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, 1.0)
    ring = seed_fn(cl, pos, cell_size)
    ring_js, ring_faces = ring if ring is not None else get_measurement_ring(cl)
    if ring_js is None:
        print(f"    RING-FIND FAILED -- circulation not tracked", flush=True)
        centre = None
    else:
        centre = pos[ring_js].mean(axis=0)

    E0 = total_energy(cl)
    circ0 = phi_ring_circulation(cl, ring_js, centre) if ring_js is not None else float("nan")
    energy0 = gallery.energy_density(cl)
    r0, prof0, counts0 = radial_profile(cl, energy0, gallery.PLOT_R)
    print(
        f"    n_leaves={cl.n_leaves}  build_t={time.time()-t0:.1f}s"
        f"  E0={E0:.6e}  circ0={circ0:.6e}",
        flush=True,
    )

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    steps_hist = [0]
    E_hist = [1.0]
    circ_hist = [circ0]
    prof_mid = None
    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)  # plain wave dynamics ONLY -- step_cross never called
        if step_i == N_STEPS // 2:
            energy_mid = gallery.energy_density(cl)
            r_mid, prof_mid, counts_mid = radial_profile(cl, energy_mid, gallery.PLOT_R)
        if step_i % REPORT_EVERY == 0:
            E = total_energy(cl)
            circ = phi_ring_circulation(cl, ring_js, centre) if ring_js is not None else float("nan")
            steps_hist.append(step_i)
            E_hist.append(E / E0 if E0 else float("nan"))
            circ_hist.append(circ)
            print(f"    step {step_i:4d}  E/E0={E/E0:.8f}  circ={circ:.6e}", flush=True)

    energy_end = gallery.energy_density(cl)
    r_end, prof_end, counts_end = radial_profile(cl, energy_end, gallery.PLOT_R)

    drift_mid = shape_drift(r0, prof0, r_mid, prof_mid, gallery.PLOT_R) if prof_mid is not None else float("nan")
    drift_end = shape_drift(r0, prof0, r_end, prof_end, gallery.PLOT_R)

    if not np.isnan(circ0) and abs(circ0) > 1e-6:
        circ_erosion_frac = float((circ0 - circ_hist[-1]) / circ0)
    elif not np.isnan(circ0):
        circ_erosion_frac = float(abs(circ_hist[-1]))  # started ~0: report raw drift, not a ratio
    else:
        circ_erosion_frac = float("nan")

    print(
        f"    SUMMARY: E/E0 final={E_hist[-1]:.8f}  circ0={circ0:.4e} circ_final={circ_hist[-1]:.4e}"
        f"  shape_drift(mid)={drift_mid:.4f}  shape_drift(end)={drift_end:.4f}"
        f"  total_t={time.time()-t0:.1f}s",
        flush=True,
    )
    return {
        "name": name,
        "E0": E0,
        "circ0": circ0,
        "steps": steps_hist,
        "E_ratio": E_hist,
        "circ": circ_hist,
        "shape_drift_mid": drift_mid,
        "shape_drift_end": drift_end,
        "circ_erosion_frac": circ_erosion_frac,
    }


def main():
    results = []
    for name, seed_fn in CONSTRUCTIONS:
        try:
            results.append(run_one(name, seed_fn))
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            results.append({"name": name, "FAILED": str(exc)})

    print("\n=== FINAL SUMMARY (plain step(), NO step_cross, N_STEPS={}) ===".format(N_STEPS))
    for r in results:
        if "FAILED" in r:
            print(f"{r['name']:60s} FAILED: {r['FAILED']}")
            continue
        print(
            f"{r['name']:60s} E/E0={r['E_ratio'][-1]:.8f}  circ0={r['circ0']:.4e}"
            f"  circ_final={r['circ'][-1]:.4e}  shape_drift_end={r['shape_drift_end']:.4f}"
        )

    # Overlay plots
    fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        if "FAILED" in r:
            continue
        ax_e.plot(r["steps"], r["E_ratio"], marker="o", ms=3, label=r["name"])
        ax_c.plot(r["steps"], r["circ"], marker="o", ms=3, label=r["name"])
    ax_e.set_xlabel("step")
    ax_e.set_ylabel("E(t)/E0")
    ax_e.set_title("Total field energy, plain step() only (no step_cross)")
    ax_e.legend(fontsize=7)
    ax_c.set_xlabel("step")
    ax_c.set_ylabel("phi_ring_circulation")
    ax_c.set_title("Macro ring circulation, plain step() only (no step_cross)")
    ax_c.legend(fontsize=7)
    fig.tight_layout()
    out_path = os.path.join(_HERE, "plain_step_persistence.png")
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
