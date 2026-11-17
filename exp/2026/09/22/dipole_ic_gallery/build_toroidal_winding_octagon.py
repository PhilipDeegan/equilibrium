"""Full characterization of a real candidate electron IC: init_toroidal_winding
seeded on a properly-resolved octagon ring (find_toroidal_ring mode="octagon"),
at this session's production resolution (R_DOMAIN=80, matching every other
construction in this gallery) -- NOT another scoping/prototype exercise.

Combines, in one script (one cluster build, reused throughout):
  1. Static 4-panel diagnostic (same style as plot_per_ic_diagnostics.py):
     linear meridional slice, log meridional slice, log equatorial slice,
     radial energy-density profile with the ring's own radius marked.
  2. t=0 macro ring circulation (phi_ring_circulation), for the comparison
     table against every other construction measured today.
  3. Persistence run under plain step() ONLY (step_cross never called),
     same methodology as test_plain_step_persistence.py: energy, circulation,
     and radial-profile shape tracked over 300 steps.

hops=12 chosen empirically (calibration run, not guessed): gives
ring_radius=29.58 DST units, centroid at the origin to float precision --
matches this project's established electron centreline-radius convention
(r_axis ~ 0.63*scale = 31.5 for the Taylor-state constructions, LOOP_A=30
for the bare current loop).
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/deegan/git/eq")
sys.path.insert(0, _HERE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import plot_all_dipole_ics as gallery
from plot_per_ic_diagnostics import equatorial_slice, radial_profile, _log_floor
from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    phi_ring_circulation,
)

OUT_DIR = os.path.join(_HERE, "per_ic")
os.makedirs(OUT_DIR, exist_ok=True)

HOPS = 12
N_STEPS = 300
REPORT_EVERY = 50


def total_energy(cl):
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2))


def shape_drift(r0, p0, r1, p1, r_max):
    valid = (~np.isnan(p0)) & (~np.isnan(p1)) & (r0 <= r_max)
    if not valid.any():
        return float("nan")
    peak = np.nanmax(p0[valid])
    sig = valid & (p0 > 0.01 * peak)
    if not sig.any():
        return float("nan")
    return float(np.max(np.abs(p1[sig] / p0[sig] - 1.0)))


def main():
    t0 = time.time()
    print("Building cluster (R_DOMAIN=80, same convention as rest of gallery)...", flush=True)
    alat, cl = gallery.build_cluster()
    print(f"  n_leaves={cl.n_leaves}  build_t={time.time()-t0:.1f}s", flush=True)

    pos = np.array(cl.leaf_pos)
    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="octagon", hops=HOPS)
    assert ring_js is not None, f"octagon ring not found at hops={HOPS}"
    ring_pos = pos[ring_js]
    rho = np.sqrt(ring_pos[:, 0] ** 2 + ring_pos[:, 1] ** 2)
    ring_radius = float(rho.mean())
    centre = ring_pos.mean(axis=0)
    print(
        f"  octagon ring: hops={HOPS}  n_tiles={len(ring_js)}"
        f"  ring_radius={ring_radius:.2f}  centre={centre}",
        flush=True,
    )

    omega = init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0)
    print(f"  seeded init_toroidal_winding, amplitude=1.0, resonant omega={omega:.6f}", flush=True)

    # --- t=0 static diagnostics -------------------------------------------------
    R_BOUNDARY = ring_radius  # natural marker for THIS construction, not R_sph*scale
    energy0 = gallery.energy_density(cl)
    e_max = float(energy0.max())
    E0 = total_energy(cl)
    circ0 = phi_ring_circulation(cl, ring_js, centre)
    print(f"  E0={E0:.6e}  circ0={circ0:.6e}  e_max={e_max:.6e}", flush=True)

    x_m, z_m, e_m = gallery.meridional_slice(cl, energy0)
    x_e, y_e, e_e = equatorial_slice(cl, energy0)
    r_c, prof0, counts0 = radial_profile(cl, energy0, gallery.PLOT_R)

    floor = _log_floor(e_max)
    tail_floor = max(e_max * 1e-6, 1e-13)
    valid = counts0 > 0
    beyond = r_c > R_BOUNDARY * 1.1
    beyond_valid = valid & beyond & ~np.isnan(prof0)
    if beyond_valid.any():
        tail_frac = float(np.mean(prof0[beyond_valid] > tail_floor))
        tail_max = float(np.nanmax(prof0[beyond_valid]))
    else:
        tail_frac, tail_max = float("nan"), float("nan")
    verdict = (
        "TAIL past ring radius (open-lineage signature)"
        if tail_frac > 0.5
        else "HARD FLOOR at ring radius (closed-lineage signature)"
    )
    print(f"  tail_frac_above_floor={tail_frac:.3f}  tail_max_beyond={tail_max:.3e}  -> {verdict}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    ax_lin, ax_log, ax_eq, ax_rad = axes.ravel()

    sc = ax_lin.scatter(x_m, z_m, c=e_m, s=3, cmap="inferno", vmin=0.0, vmax=max(e_max, 1e-12))
    ax_lin.set_title("meridional (x-z), linear scale")
    ax_lin.set_xlim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_lin.set_ylim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_lin.set_aspect("equal")
    fig.colorbar(sc, ax=ax_lin, fraction=0.046, pad=0.04)

    e_m_clipped = np.clip(e_m, floor, None)
    sc2 = ax_log.scatter(
        x_m, z_m, c=e_m_clipped, s=3, cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=max(e_max, floor * 10)),
    )
    ax_log.set_title(f"meridional (x-z), LOG scale (floor={floor:.1e})")
    ax_log.set_xlim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_log.set_ylim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_log.set_aspect("equal")
    ax_log.add_patch(plt.Circle((0, 0), R_BOUNDARY, fill=False, ec="cyan", lw=1.0, ls="--"))
    fig.colorbar(sc2, ax=ax_log, fraction=0.046, pad=0.04)

    e_e_clipped = np.clip(e_e, floor, None)
    sc3 = ax_eq.scatter(
        x_e, y_e, c=e_e_clipped, s=3, cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=max(e_max, floor * 10)),
    )
    ax_eq.set_title("equatorial (x-y), LOG scale")
    ax_eq.set_xlim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_eq.set_ylim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_eq.set_aspect("equal")
    ax_eq.add_patch(plt.Circle((0, 0), R_BOUNDARY, fill=False, ec="cyan", lw=1.0, ls="--"))
    fig.colorbar(sc3, ax=ax_eq, fraction=0.046, pad=0.04)

    prof_plot = np.where(np.isnan(prof0), floor, np.clip(prof0, floor, None))
    ax_rad.semilogy(r_c, prof_plot, marker=".", ms=3, lw=1.0, color="darkorange")
    ax_rad.axvline(R_BOUNDARY, color="cyan", ls="--", lw=1.2, label=f"ring radius={R_BOUNDARY:.1f}")
    ax_rad.axhline(tail_floor, color="grey", ls=":", lw=0.8, label="tail-floor threshold")
    ax_rad.set_ylim(floor * 0.5, max(e_max, floor * 10) * 2)
    ax_rad.set_xlim(0, gallery.PLOT_R)
    ax_rad.set_title(f"radial mean energy density -> {verdict}", fontsize=9.5)
    ax_rad.set_xlabel("r (DST units)")
    ax_rad.set_ylabel("mean energy density (per-shell)")
    ax_rad.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"init_toroidal_winding on octagon ring (hops={HOPS}, n_tiles={len(ring_js)},"
        f" ring_radius={ring_radius:.1f})\n(adaptive_flux_tube.py -- t=0 static diagnostics)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    static_path = os.path.join(OUT_DIR, "toroidal_winding_octagon.png")
    fig.savefig(static_path, dpi=140)
    plt.close(fig)
    print(f"  saved {static_path}", flush=True)

    # --- persistence run: plain step() ONLY, step_cross never called -----------
    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    steps_hist = [0]
    E_hist = [1.0]
    circ_hist = [circ0]
    prof_mid = None
    r_mid = None
    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)  # plain wave dynamics ONLY
        if step_i == N_STEPS // 2:
            energy_mid = gallery.energy_density(cl)
            r_mid, prof_mid, _ = radial_profile(cl, energy_mid, gallery.PLOT_R)
        if step_i % REPORT_EVERY == 0:
            E = total_energy(cl)
            circ = phi_ring_circulation(cl, ring_js, centre)
            steps_hist.append(step_i)
            E_hist.append(E / E0 if E0 else float("nan"))
            circ_hist.append(circ)
            print(f"    step {step_i:4d}  E/E0={E/E0:.8f}  circ={circ:.6e}", flush=True)

    energy_end = gallery.energy_density(cl)
    r_end, prof_end, _ = radial_profile(cl, energy_end, gallery.PLOT_R)

    drift_mid = shape_drift(r_c, prof0, r_mid, prof_mid, gallery.PLOT_R) if prof_mid is not None else float("nan")
    drift_end = shape_drift(r_c, prof0, r_end, prof_end, gallery.PLOT_R)

    print(
        f"\n  PERSISTENCE SUMMARY: E/E0 final={E_hist[-1]:.8f}  circ0={circ0:.4e}"
        f"  circ_final={circ_hist[-1]:.4e}  shape_drift(mid)={drift_mid:.4f}"
        f"  shape_drift(end)={drift_end:.4f}  total_t={time.time()-t0:.1f}s",
        flush=True,
    )

    fig2, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(12, 5))
    ax_e.plot(steps_hist, E_hist, marker="o", ms=4, color="C0")
    ax_e.set_xlabel("step")
    ax_e.set_ylabel("E(t)/E0")
    ax_e.set_title("Total field energy, plain step() only (no step_cross)")
    ax_c.plot(steps_hist, circ_hist, marker="o", ms=4, color="C1")
    ax_c.set_xlabel("step")
    ax_c.set_ylabel("phi_ring_circulation")
    ax_c.set_title("Macro ring circulation, plain step() only (no step_cross)")
    fig2.suptitle(
        f"init_toroidal_winding, octagon ring (hops={HOPS}) -- persistence, {N_STEPS} steps",
        fontsize=11,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    persist_path = os.path.join(_HERE, "toroidal_winding_octagon_persistence.png")
    fig2.savefig(persist_path, dpi=140)
    print(f"  saved {persist_path}", flush=True)

    print("\n=== FINAL ===")
    print(f"ring_radius={ring_radius:.2f}  n_tiles={len(ring_js)}  omega={omega:.6f}")
    print(f"static verdict: {verdict}  (tail_frac={tail_frac:.3f})")
    print(f"circ0={circ0:.6e}")
    print(f"E/E0 final={E_hist[-1]:.8f}  circ_final={circ_hist[-1]:.6e}")
    print(f"shape_drift mid={drift_mid:.4f}  end={drift_end:.4f}")


if __name__ == "__main__":
    main()
