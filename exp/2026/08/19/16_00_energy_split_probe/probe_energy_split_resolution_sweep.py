"""
Is the ~40x local-circulation growth (probe_energy_split_v2.py) a
resolution artifact -- same pattern already seen repeatedly in this project
(BCC dispersion anisotropy, Casimir-Polder under-resolution, spheromak
cells_across scaling)? Sweep CELLS_ACROSS while keeping R_DOMAIN in the
same ratio (so the object occupies more actual lattice cells at fixed
underlying cell size d1=1.0 -- genuine resolution, not just "growing the
object"), track the peak magnetic-energy growth ratio at each resolution.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from equilibrium.substrate.discrete_curl import circulation, leaf_nb_rows
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM

R_SPH = 1.0
REPORT_EVERY_FRAC = 1.0 / 6.0  # 6 reports per run regardless of scale
DOMAIN_RATIO = 18.0 / 22.0  # preserve original fixture's object/domain margin
BASE_CELLS_ACROSS = 22
BASE_N_STEPS = 30  # scaled linearly with cells_across below -- matched phase, not matched step count


def run(cells_across, amplitude_tor):
    n_steps = int(round(BASE_N_STEPS * cells_across / BASE_CELLS_ACROSS))
    report_every = max(1, int(round(n_steps * REPORT_EVERY_FRAC)))
    scale = cells_across / 2.0
    r_domain = cells_across * DOMAIN_RATIO

    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, r_domain, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(8)
    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, 1.0)
    nb = leaf_nb_rows(cl)

    seed_curl_full_dipole_ic(
        cl,
        pos,
        cell_size,
        scale,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        amplitude_pol=1.0,
        amplitude_tor=amplitude_tor,
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
    peak_ratio = 1.0
    for step_i in range(1, n_steps + 1):
        evolver.step(ts)
        if step_i % report_every == 0:
            e, m = energies()
            ratio = m / max(m0, 1e-300)
            peak_ratio = max(peak_ratio, ratio)
            print(
                f"  cells_across={cells_across:4d} step={step_i:3d}/{n_steps}  "
                f"electric_ratio={e/e0:.4f}  magnetic_ratio={ratio:.4f}",
                flush=True,
            )
    print(
        f"cells_across={cells_across:4d}  N={cl.n_leaves:8d}  m0={m0:.4e}  "
        f"peak_magnetic_ratio(<=step{n_steps})={peak_ratio:.4f}",
        flush=True,
    )
    return cl.n_leaves, m0, peak_ratio


if __name__ == "__main__":
    results = []
    for ca in (22, 44, 88):
        print(f"=== cells_across={ca} ===", flush=True)
        n, m0, peak = run(ca, amplitude_tor=0.0)  # poloidal-only, isolates the artifact
        results.append((ca, n, m0, peak))
        print(flush=True)

    print("=== summary (poloidal-only, amplitude_tor=0) ===")
    for ca, n, m0, peak in results:
        print(f"cells_across={ca:4d}  N={n:8d}  peak_magnetic_ratio={peak:.4f}")
