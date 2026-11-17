"""
Pin down whether the local-circulation "magnetic energy" growth seen in
probe_energy_split_v2.py / resolution_sweep is a genuine source-intrinsic
effect, or reflection pile-up off an unset (default = reflecting/Neumann)
boundary in a domain whose margin was only ~1.64x the object's own radius.

Three configs at fixed cells_across=44 (N~250k, matches the sweep's middle
point), 60 steps (matches the sweep's own step count at this resolution):
  A) original: no BC set (reflecting default), tight margin (domain_ratio as before)
  B) same tight margin, but WITH a sponge (same convention as the Poynting probes)
  C) no BC set (reflecting default), but a much larger margin (domain 4x
     bigger relative to the object) -- pushes reflection arrival much later
     without needing to guess the sponge's own absorption quality
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from equilibrium.substrate.discrete_curl import circulation, leaf_nb_rows
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM

CELLS_ACROSS = (
    22  # cheap -- N~32k at tight margin, keeps the wide-margin case affordable too
)
R_SPH = 1.0
N_STEPS = 30  # matches the resolution sweep's own step count at cells_across=22
REPORT_EVERY = 5
SCALE = CELLS_ACROSS / 2.0
TIGHT_DOMAIN_RATIO = 18.0 / 22.0  # original fixture margin (~1.64x object radius)
WIDE_DOMAIN_RATIO = TIGHT_DOMAIN_RATIO * 3.0  # ~4.9x object radius, ~27x the volume


def run(label, r_domain, use_sponge):
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, r_domain * CELLS_ACROSS, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(8)
    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, 1.0)
    nb = leaf_nb_rows(cl)

    if use_sponge:
        r_all = np.linalg.norm(pos, axis=1)
        r_max = float(r_all.max())
        r_inner = r_max * (1.0 - 0.15)
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge(
                target=0.0, damp_max=0.003, r_all=r_all, r_inner=r_inner
            )
        )

    seed_curl_full_dipole_ic(
        cl,
        pos,
        cell_size,
        SCALE,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        amplitude_pol=1.0,
        amplitude_tor=0.0,
    )

    dt = 0.4 / 6.0
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    def energies():
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        circ = circulation(phi, nb)
        return 0.5 * float(np.sum(u**2)), 0.5 * float(np.sum(circ**2))

    e0, m0 = energies()
    print(
        f"[{label}] N={cl.n_leaves}  r_domain={r_domain*CELLS_ACROSS:.1f}  "
        f"margin_ratio={r_domain*CELLS_ACROSS/(R_SPH*SCALE):.2f}  sponge={use_sponge}  "
        f"e0={e0:.4e} m0={m0:.4e}",
        flush=True,
    )
    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)
        if use_sponge:
            alat.step_sponge(0)
        if step_i % REPORT_EVERY == 0:
            e, m = energies()
            print(
                f"[{label}] step {step_i:3d}/{N_STEPS}  electric_ratio={e/e0:.4f}  "
                f"magnetic_ratio={m/max(m0,1e-300):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    run("A tight+no-sponge", TIGHT_DOMAIN_RATIO, use_sponge=False)
    print(flush=True)
    run("B tight+sponge   ", TIGHT_DOMAIN_RATIO, use_sponge=True)
    print(flush=True)
    run("C wide+no-sponge ", WIDE_DOMAIN_RATIO, use_sponge=False)
