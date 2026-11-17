"""Does the curl-full dipole IC's poloidal (u, phi_pol) and toroidal
(phi_tor) content stay locked at a common rate under pure wave dynamics, or
drift apart? Philip's hope: if the spheromak IC is built correctly, it
would produce a genuine dipole with BOTH fluxes moving at c together (the
toroidal x poloidal ~ c^2 / E=mc^2 idea) -- but the current construction
only independently peak-normalizes phi_pol and phi_tor at t=0; nothing
enforces they stay proportionally locked as the system evolves.

Diagnostic (pure evolver.step(), NO step_cross -- isolates wave dynamics
from the resistive/decay operator entirely):
- Macro ring circulation (phi_ring_circulation) is EXACTLY conserved under
  pure step() by the Kelvin-circulation theorem already proven in
  test_ring_circulation_conservation.py -- this is the toroidal-content
  tracer (phi_pol alone gives exactly 0 circulation, confirmed in
  test_curl_full_dipole_ic.py).
- u_rms is the poloidal-content tracer (nothing else seeds u).

If u_rms(t)/u_rms(0) stays close to circ(t)/circ(0)=1.0 (trivially exactly
1 by the theorem), the two components are moving together. If u_rms decays
measurably while circulation stays frozen, they are NOT locked -- the
current IC does not automatically produce Philip's hoped-for structure,
that would need something beyond independent amplitude-1 seeding.

Small test-fixture scale (CELLS_ACROSS=22, R_DOMAIN=18), matching
test_curl_full_dipole_ic.py's own fixture -- fast (~seconds), not the
production checkpoints.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    find_toroidal_ring,
    phi_ring_circulation,
)

CELLS_ACROSS = 22
R_SPH = 1.0
R_DOMAIN = 18.0
SCALE = CELLS_ACROSS / 2.0
N_STEPS = 60
REPORT_EVERY = 10


def build():
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def main():
    alat, cl = build()
    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, 1.0)

    seed_curl_full_dipole_ic(
        cl,
        pos,
        cell_size,
        SCALE,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        amplitude_pol=1.0,
        amplitude_tor=1.0,
    )

    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="square", hops=3)
    assert (
        ring_js is not None
    ), "ring-finding failed -- check R_DOMAIN/hops fit the domain"
    centre = pos[ring_js].mean(axis=0)

    u0 = float(np.sqrt(np.mean(np.array(cl.leaf_u) ** 2)))
    circ0 = phi_ring_circulation(cl, ring_js, centre)
    print(f"t=0  u_rms={u0:.6e}  circ={circ0:.6e}", flush=True)

    dt = (
        0.4 / 6.0
    )  # matches test_ring_circulation_conservation.py's conservative fixed dt
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    for step_i in range(1, N_STEPS + 1):
        evolver.step(ts)  # pure wave dynamics, NO step_cross
        if step_i % REPORT_EVERY == 0:
            u_rms = float(np.sqrt(np.mean(np.array(cl.leaf_u) ** 2)))
            circ = phi_ring_circulation(cl, ring_js, centre)
            print(
                f"step {step_i:3d}  u_rms={u_rms:.6e} (ratio {u_rms/u0:.6f})  "
                f"circ={circ:.6e} (ratio {circ/circ0:.6f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
