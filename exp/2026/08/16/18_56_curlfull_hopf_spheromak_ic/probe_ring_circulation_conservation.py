"""Validate the existing coordinate-free curl-full construction
(tst.substrate.adaptive.adaptive_flux_tube.init_toroidal_winding +
find_toroidal_ring) and test the Kelvin-circulation-conservation hypothesis:
does plain wave dynamics (evolver.step(), no step_cross) conserve the ring's
circulation EXACTLY (pass2's dphi/dt=-sigma*grad(u) always has zero
circulation around any closed loop of DISCRETE differences -- same
telescoping fact as equilibrium.substrate.discrete_curl, just on the ring's
own larger loop instead of an elementary 3-cell triangle), while step_cross
(the resistive/topology-opening term) erodes it, enabling reconnection --
matching PHARE's flux-tube merging mechanism and this session's
project_dst_discrete_curl finding.

NOTE: init_toroidal_winding is coordinate-free (phi_val = amplitude*cos(K*
(idx+0.5)) assigned by RING INDEX only, no x/y/z anywhere in the generative
rule) -- unlike init_spheromak_ic (a DIFFERENT, older function in the same
file), which is the same buggy Cartesian Taylor-state construction
dipole_mode_ic.py replaced (u=|B|, phi=B.n_hat from two different
equations) -- do not use init_spheromak_ic for this investigation.
winding_number/phi_ring_circulation DO use coordinates, but only to measure/
interpret circulation as a signed physical number for humans -- that's a
diagnostic concern, not a generative one, and doesn't reintroduce the
Cartesian-dependence problem.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    winding_number,
    phi_ring_circulation,
    _MIRROR,
)

R_DOMAIN = 18.0


def seed_constant_current(cl, ring_js, ring_faces, amplitude=1.0):
    """UNLIKE init_toroidal_winding's amplitude*cos(K*(idx+0.5)) standing
    wave (sums to exactly zero over one period -- confirmed analytically,
    -3.8e-15 numerically -- so it IS expressible as a scalar's discrete
    derivative around the ring, i.e. curl-free in disguise, matching its
    measured phi_ring_circulation~1e-15), this assigns the SAME phi value
    all the way around the ring -- a genuine persistent current, sum =
    amplitude*N_ring != 0, which CANNOT be a scalar's discrete derivative
    around a closed loop (the necessary zero-sum condition fails by
    construction). This is the actual curl-full counter-example.
    """
    N_ring = len(ring_js)
    phi = cl.leaf_phi.copy()
    for idx in range(N_ring):
        j_i = ring_js[idx]
        j_nxt = ring_js[(idx + 1) % N_ring]
        k = ring_faces[idx]
        mk = int(_MIRROR[k])
        phi[k, j_i] = amplitude
        phi[mk, j_nxt] = -amplitude
    cl.set_leaf_phi(phi)


def build():
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    return alat, cl


def report(label, cl, ring_js, centre):
    wn = winding_number(cl, ring_js)
    circ = phi_ring_circulation(cl, ring_js, centre)
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    H = float(np.sum(u**2) + 0.5 * np.sum(phi**2))
    print(
        f"[{label}] winding_number={wn:.6f}  phi_ring_circulation={circ:.6e}  H={H:.6e}",
        flush=True,
    )
    return wn, circ, H


def main():
    alat, cl = build()
    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="square", hops=3)
    if ring_js is None:
        print("no ring found -- try a bigger domain or fewer hops")
        return
    print(f"ring found: {len(ring_js)} tiles", flush=True)
    centre = np.array(cl.leaf_pos)[ring_js].mean(axis=0)

    seed_constant_current(cl, ring_js, ring_faces, amplitude=1.0)
    wn0, circ0, H0 = report("t=0", cl, ring_js, centre)

    dt = 0.4 / (
        2.0 * 3.0
    )  # conservative CFL guess: 0.4 * L_min / (sigma_max * W2_SUM), W2_SUM~ up to ~30
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    print("\n--- plain wave dynamics only (no step_cross) ---", flush=True)
    for i in range(1, 11):
        evolver.step(ts)
        if i % 2 == 0:
            report(f"pure step {i}", cl, ring_js, centre)

    print("\n--- rebuilding fresh IC, now WITH step_cross ---", flush=True)
    alat2, cl2 = build()
    seed_constant_current(cl2, ring_js, ring_faces, amplitude=1.0)
    evolver2 = _lat.AdaptiveLatticeEvolver(alat2)
    report("t=0 (fresh copy)", cl2, ring_js, centre)
    cross_scale = 1e-2
    for i in range(1, 11):
        evolver2.step(ts)
        alat2.step_cross_all(dt, cross_scale)
        if i % 2 == 0:
            report(f"step_cross step {i}", cl2, ring_js, centre)


if __name__ == "__main__":
    main()
