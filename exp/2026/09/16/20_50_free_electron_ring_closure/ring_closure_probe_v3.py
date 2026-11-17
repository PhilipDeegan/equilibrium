"""
v3: same ring as v2, but lam is no longer zero everywhere (void) -- it is
SOURCED, every step, by the winding's own local field energy density:
    lam(pos, t) = alpha * (u(pos,t)^2 + 0.5*sum_k phi(pos,t,k)^2)
recomputed fresh each step from the CURRENT field (not a static profile
frozen at the IC's shape) and fed straight into sigma=1/(1+lam) before the
next step() call.

Why this, not v2's void test: if mass=lam (Philip, 2026-09-16 -- an
electron's own presence, having real mass/energy, REQUIRES nonzero lam
sourced by itself, per this project's own mass=depletion definition,
[[project_lam_dual_role]]), testing lam=0 everywhere was never a valid
free-electron test at all -- it was a generic wave in genuinely empty
space. This also supplies the missing nonlinearity v2 was probing for:
sigma depending on the LOCAL field intensity is structurally the same
mechanism as the Kerr effect in nonlinear optics (intensity-dependent
refractive index -> self-focusing that can balance diffraction) -- the
"self-trapping" idea flagged and never attempted in
project_dst_electron_topology's 2026-07-12 section.

alpha is an unconstrained free knob (no validated DST mass-energy-unit
conversion exists yet) -- sweep it and look for whether increasing alpha
suppresses the v2 spread-growth, not for a specific "correct" value.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import dispersion_omega, face_offsets
from ring_closure_probe_v2 import (
    build_ring_ic_companion,
    field_energy,
    radial_spread,
    ring_amplitude,
)


def local_energy_density(lat):
    return lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)


def run_one(n, r, width, R_domain, n_periods_target, report_every, alpha):
    lat = SubstrateLattice(radius_cells=R_domain)
    offsets, w = face_offsets(lat)
    K = n / r
    omega_pred = dispersion_omega(offsets, w, axis=1, K=K)
    theta, env, rho = build_ring_ic_companion(lat, offsets, w, n, r, width, omega_pred)
    mask = env > 0.05 * env.max()

    dt = lat.stable_dt()
    T_pred = 2 * np.pi / omega_pred if omega_pred > 0 else 1.0
    n_steps = max(int(round(n_periods_target * T_pred / dt)), report_every * 8)

    ts, energies, std_rhos, lam_maxs = [], [], [], []
    for s in range(n_steps + 1):
        if alpha > 0.0:
            lat.lam = alpha * local_energy_density(lat)  # nb_idx cache is
            # topology-only (unaffected by lam) -- no _dirty flag needed
        if s % report_every == 0:
            ts.append(s * dt)
            energies.append(field_energy(lat, mask))
            _, sr = radial_spread(lat, mask, rho)
            std_rhos.append(sr)
            lam_maxs.append(float(lat.lam.max()) if alpha > 0.0 else 0.0)
        if s < n_steps:
            lat.step(dt)

    energies = np.array(energies)
    std_rhos = np.array(std_rhos)
    spread0 = std_rhos[len(std_rhos) // 10] if len(std_rhos) > 10 else std_rhos[0]
    spread_end = std_rhos[-1]
    return {
        "alpha": alpha,
        "n_steps": n_steps,
        "H0": energies[0],
        "H_end": energies[-1],
        "H_decay_frac": (energies[0] - energies[-1]) / energies[0],
        "spread0": spread0,
        "spread_end": spread_end,
        "spread_growth": (spread_end - spread0) / spread0,
        "lam_max": max(lam_maxs) if lam_maxs else 0.0,
    }


def main():
    n, r, width, R_domain = 2, 10.0, 2.0, 34.0
    n_periods_target, report_every = 6.0, 10

    print(
        f"{'alpha':>8} {'lam_max':>9} {'decay%':>7} {'spread0':>8}"
        f" {'spread_end':>10} {'spread_grow%':>12}"
    )
    for alpha in [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]:
        res = run_one(n, r, width, R_domain, n_periods_target, report_every, alpha)
        print(
            f"{alpha:8.2f} {res['lam_max']:9.4f} {100*res['H_decay_frac']:7.2f}"
            f" {res['spread0']:8.3f} {res['spread_end']:10.3f}"
            f" {100*res['spread_growth']:12.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
