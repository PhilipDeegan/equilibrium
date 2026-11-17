"""
"A dipole is just flux traveling at c in a loop -- both electric and
magnetic properties should emerge from this on its own" (Philip,
2026-09-21). Tests this directly, avoiding two things the rest of this
session found were dead ends:

1. NOT the double-ring dipole IC (init_electron_dipole_ic_vecpot) -- that
   construction is a uniform-u + exactly-divergence-free-phi fixed point
   BY DESIGN, so du/dt = -sigma*div(phi) = 0 everywhere, forever, from t=0.
   It is structurally incapable of ever self-sourcing a u/charge signature
   -- confirmed today, not a guess. Using seed_constant_current instead: a
   genuinely non-divergence-free ("curl-full") circulating current on a
   single ring, starting from u=0 everywhere (no artificial background, no
   separately-imposed deficit) -- if charge is going to emerge on its own,
   THIS is the construction that could actually show it.

2. NOT b_vec_vectorized/curl(phi)=B (project_discrete_curl_directional_bias
   -- a real, isolated bug in that specific reconstruction, plus "we are
   not modelling B generally" -- Philip). All three diagnostics below stay
   inside the phi-circulation/u-scalar ontology.

3. Octagon ring (mode="octagon"), not square -- avoids the separate,
   older, already-documented "diamond artifact" source-shape anisotropy
   (project_dst_electron_topology / project_ring_scale_convergence), kept
   in mind per Philip's request to watch for anisotropy when scoping this.

Three diagnostics tracked together, all curl-free:
  - phi circulation (phi_ring_circulation) -- the magnetic-moment-like
    signature, using the already-validated (all-14-direction) machinery.
  - u profile near the ring -- does a persistent, self-sourced charge-like
    signature appear (unlike today's earlier free-deficit test, which
    started WITH a deficit and watched it disperse; this starts from
    u=0 and watches whether the circulating current sources one).
  - a kinematic angular-momentum proxy: L ~ H_ring * r_ring / c (ring's
    own field energy, physical radius, c=sqrt(10) from the validated
    dispersion relation) -- NOT a field r x S integral (already ruled out
    this session, S is a scalar here, not E x B) -- tracked over time to
    see if it behaves like a conserved quantity.
"""

import sys
import time

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.lattice import W_HEX, W_SQ
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    phi_ring_circulation,
    _MIRROR,
)

R_DOMAIN = 40.0
HOPS = 6
SPONGE_FRACTION = 0.15
DAMP_MAX = 0.003
N_THREADS = 8
AMPLITUDE = 1.0
N_STEPS = 4000
REPORT_EVERY = 100
C_SPEED = np.sqrt(10.0)  # validated dispersion-relation wave speed, cell units

_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def seed_constant_current(cl, ring_js, ring_faces, amplitude=1.0):
    """Genuine persistent current: same phi all the way around the ring,
    sum != 0 -- curl-full, NOT div-free like the double-ring dipole."""
    N_ring = len(ring_js)
    phi = cl.leaf_phi.copy()
    for idx in range(N_ring):
        j_i = ring_js[idx]
        j_nxt = ring_js[(idx + 1) % N_ring]
        k = ring_faces[idx]
        mk = int(_MIRROR[k])
        a = amplitude / _W[k]
        phi[k, j_i] = a
        phi[mk, j_nxt] = -a
    cl.set_leaf_phi(phi)


def main():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = float(r_all.max()) * (1.0 - SPONGE_FRACTION)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=DAMP_MAX, r_all=r_all, r_inner=r_inner
        )
    )

    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="octagon", hops=HOPS)
    if ring_js is None:
        raise RuntimeError(f"ring-finding failed at R={R_DOMAIN}, hops={HOPS}")
    pos = np.array(cl.leaf_pos)
    centre = pos[ring_js].mean(axis=0)
    r_ring = float(np.mean(np.linalg.norm(pos[ring_js, :2] - centre[:2], axis=1)))
    print(f"N={cl.n_leaves}  ring tiles={len(ring_js)}  r_ring={r_ring:.3f}  centre={centre}", flush=True)

    seed_constant_current(cl, ring_js, ring_faces, amplitude=AMPLITUDE)
    # u stays at its build-time default (0.0 everywhere) -- no artificial
    # background, no separately-imposed deficit.

    dt = compute_dt(cl)
    r_mask = np.linalg.norm(pos - centre, axis=1) <= 2.0 * r_ring  # local "near-ring" window

    def diagnostics(step):
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        circ = phi_ring_circulation(cl, ring_js, centre)
        u_near = u[r_mask]
        u_l1 = float(np.abs(u_near).sum())  # total |u| sourced near the ring -- charge-signature proxy
        H_ring = float(np.sum(u_near**2) + 0.5 * np.sum(phi[:, r_mask] ** 2))
        L_proxy = H_ring * r_ring / C_SPEED
        return circ, u_l1, H_ring, L_proxy

    print(f"\n{'step':>6} {'t':>8} {'circ':>10} {'u_L1(near)':>12} {'H_ring':>10} {'L_proxy':>10}", flush=True)
    t0 = time.time()
    for step in range(N_STEPS + 1):
        if step % REPORT_EVERY == 0:
            circ, u_l1, H_ring, L_proxy = diagnostics(step)
            print(f"{step:6d} {step*dt:8.2f} {circ:10.4e} {u_l1:12.4e} {H_ring:10.4e} {L_proxy:10.4e}", flush=True)
        if step < N_STEPS:
            alat.step(0, dt)
            alat.step_sponge(0)
    print(f"\ndone in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
