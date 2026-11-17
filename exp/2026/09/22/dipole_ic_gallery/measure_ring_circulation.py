"""Quantitative check: does each "poloidal-only" electron/dipole IC actually
carry zero macro ring circulation (per phi_ring_circulation, the tool
test_curl_full_dipole_ic.py already validated as the correct discriminator --
NOT the per-triangle circulation/RMS ratio, which does not distinguish
poloidal from toroidal fields), or does it secretly carry real circulation
like the spheromak's known-decaying toroidal piece (circ_B: -1.447 -> ~0,
doc/mems/project_spheromak_decay.md)?

Same R_DOMAIN=80/R_sph=1.0/scale=50.0 convention as
exp/2026/09/22/dipole_ic_gallery/plot_all_dipole_ics.py -- reuses its
build_cluster() and seeding functions directly.
"""

import sys

sys.path.insert(0, "/home/deegan/git/eq")

import numpy as np

from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    phi_ring_circulation,
    init_electron_dipole_ic,
    init_electron_dipole_ic_vecpot,
    init_spheromak_ic,
)
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic
from equilibrium import lattice as _lat

R_DOMAIN = 80.0
SCALE = 50.0
R_SPH = 1.0
N_THREADS = 8
_FACE_HOP = __import__(
    "tst.substrate.adaptive.adaptive_flux_tube", fromlist=["_FACE_HOP"]
)._FACE_HOP
_FACE_NORM = __import__(
    "tst.substrate.adaptive.adaptive_flux_tube", fromlist=["_FACE_NORM"]
)._FACE_NORM


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)
    return alat, cl


CONSTRUCTIONS = {
    "init_electron_dipole_ic": lambda cl: init_electron_dipole_ic(
        cl, R_sph=R_SPH, scale=SCALE, amplitude=1.0
    ),
    "init_electron_dipole_ic_vecpot": lambda cl: init_electron_dipole_ic_vecpot(
        cl, R_sph=R_SPH, scale=SCALE, amplitude=1.0
    ),
    "init_spheromak_ic": lambda cl: init_spheromak_ic(
        cl, R_sph=R_SPH, scale=SCALE, amplitude=1.0
    ),
    "seed_dipole_ic": lambda cl: seed_dipole_ic(
        cl,
        np.array(cl.leaf_pos),
        np.full(cl.n_leaves, 1.0),
        SCALE,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        amplitude=1.0,
    ),
}

# hops chosen so the macro ring's own radius lands near this project's
# established centreline radius r_axis ~ 0.63*SCALE = 31.5 DST units
# (cell_size=1.0 here, so hops in DST units ~= hops in cells).
HOPS = 27


def main():
    print(f"R_DOMAIN={R_DOMAIN} SCALE={SCALE} R_SPH={R_SPH} HOPS={HOPS}\n")
    results = {}
    for name, seed_fn in CONSTRUCTIONS.items():
        print(f"--- {name} ---", flush=True)
        alat, cl = build_cluster()
        seed_fn(cl)

        ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="square", hops=HOPS)
        if ring_js is None:
            print(f"    RING-FIND FAILED at hops={HOPS}")
            results[name] = None
            continue
        pos = np.array(cl.leaf_pos)
        centre = pos[ring_js].mean(axis=0)
        # report ring's own radius (distance of its tiles from z-axis, mean)
        rho = np.sqrt(pos[ring_js, 0] ** 2 + pos[ring_js, 1] ** 2)
        ring_radius = float(rho.mean())

        circ = phi_ring_circulation(cl, ring_js, centre)
        print(
            f"    n_ring_tiles={len(ring_js)}  ring_radius={ring_radius:.2f}"
            f"  centre={centre}\n    circulation={circ:.6e}",
            flush=True,
        )
        results[name] = (ring_radius, circ)

    print("\n=== SUMMARY ===")
    for name, r in results.items():
        if r is None:
            print(f"{name}: RING-FIND FAILED")
            continue
        ring_radius, circ = r
        verdict = "CIRCULATION-FREE (poloidal only)" if abs(circ) < 1e-6 else "REAL CIRCULATION (carries decay-enabling free energy)"
        print(f"{name}: ring_radius={ring_radius:.2f}  circ={circ:.6e}  -> {verdict}")


if __name__ == "__main__":
    main()
