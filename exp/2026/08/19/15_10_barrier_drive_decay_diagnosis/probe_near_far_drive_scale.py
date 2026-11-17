"""
What's the natural (undrained, cross_scale=0) pooled drive scale near the
ring vs far from it? Checking whether a "derived" m0 (doc/dst_next.md #8:
m0_proton~0.16, m0_neutron~2.49, order-unity) would EVER fire anywhere in
this ring construction, vs the picked m0=1e-4 used so far.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    winding_number,
)

R = 30.0
HOPS = 4
N_STEPS = 3000
LOG_EVERY = 250
_W = np.array([3.0] * 8 + [1.0] * 6)

alat = _lat.AdaptiveLattice()
alat.add_cluster()
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
alat.set_n_threads(8)

r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
r_max = float(r_all.max())
r_inner = r_max * (1.0 - 0.15)
cl.set_bc(
    _lat.BoundaryCondition.quadratic_sponge(
        target=0.0, damp_max=0.003, r_all=r_all, r_inner=r_inner
    )
)

ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="square", hops=HOPS)
w = winding_number(cl, ring_js, axis="z")
omega = init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0, phi_only=True)
pos = np.array(cl.leaf_pos)
ring_pos = pos[ring_js]
ring_centre = ring_pos.mean(axis=0)
ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

dt = compute_dt(cl)
r_from_centre = np.linalg.norm(pos - ring_centre, axis=1)

near = (r_from_centre >= 1.0 * ring_r) & (r_from_centre <= 3.0 * ring_r)
far = (r_from_centre >= 0.4 * R) & (r_from_centre <= 0.8 * R)
on_ring = np.zeros(cl.n_leaves, dtype=bool)
on_ring[ring_js] = True

print(
    f"N={cl.n_leaves} ring_r={ring_r:.3f} near_band=[{1.0*ring_r:.2f},{3.0*ring_r:.2f}] "
    f"n_near={int(near.sum())} n_far={int(far.sum())} n_on_ring={int(on_ring.sum())} dt={dt:.4e}",
    flush=True,
)
print(
    "candidate m0 values: picked=1e-4, derived_proton~0.16, derived_neutron~2.49",
    flush=True,
)


def stats(mask, phi):
    inward = np.where(phi < 0.0, -phi, 0.0)
    drive = inward @ _W
    d = drive[mask]
    if d.size == 0:
        return 0.0, 0.0, 0
    return float(d.max()), float(np.percentile(d, 99)), int(d.size)


for step in range(N_STEPS + 1):
    if step % LOG_EVERY == 0:
        phi = np.array(cl.leaf_phi).T
        max_on, p99_on, n_on = stats(on_ring, phi)
        max_near, p99_near, n_near = stats(near, phi)
        max_far, p99_far, n_far = stats(far, phi)
        print(
            f"step={step:5d}  ring: max={max_on:.4e} p99={p99_on:.4e}  |  "
            f"near: max={max_near:.4e} p99={p99_near:.4e}  |  "
            f"far: max={max_far:.4e} p99={p99_far:.4e}",
            flush=True,
        )
    alat.step(0, dt)
    alat.step_sponge(0)

print("done", flush=True)
