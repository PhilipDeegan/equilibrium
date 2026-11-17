"""
Diagnostic: does the barrier gate's drive in a fixed outer radius band
decay below m0 over long settle times, independent of the sponge?

Hypothesis being tested: probe_poynting_dipole_compare.py's "n=0" result at
n_steps=11200 (vs nonzero at 8000) happens because gate="barrier" drains
phi continuously EVERY step during the main loop (not just at the final
diagnostic step) -- so running longer under continuous barrier drainage
depletes the drive in a fixed outer band, eventually pushing it below m0
everywhere there, regardless of the true underlying wave amplitude.

Two configs, same cheap small-R setup:
  A) cross_scale=0.01, gate=barrier every step (matches the real script)
  B) cross_scale=0.0 (pure wave dynamics, no draining) -- control

Track, every N steps, in the SAME fixed radius band (as a fraction of R):
  - max drive, count of cells with drive>m0, in-band
  - max |phi| in-band (raw amplitude, gate-independent)
sponge starts well outside the fit band (r_inner far beyond r_max_fit) so
this isolates the barrier-drainage-over-time effect from sponge absorption.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)

R = 30.0
HOPS = 4
M0 = 1e-4
CROSS_SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
LOG_EVERY = 500

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

lam = np.array(cl.leaf_lam)
for j in ring_js:
    lam[j] = 1.0
cl.set_leaf_lam(lam)

dt = compute_dt(cl)
period = 2.0 * np.pi / omega if omega > 0 else None
r_from_centre = np.linalg.norm(pos - ring_centre, axis=1)

r_min_fit, r_max_fit = 0.4 * R, 0.8 * R
in_band = (r_from_centre >= r_min_fit) & (r_from_centre <= r_max_fit)
n_band = int(in_band.sum())

print(
    f"N={cl.n_leaves} R={R} hops={HOPS} winding={w:+.3f} ring_r={ring_r:.3f} "
    f"omega={omega:.4f} period={period:.2f} dt={dt:.4e} periods_at_n_steps={N_STEPS*dt/period:.3f} "
    f"cross_scale={CROSS_SCALE} band=[{r_min_fit:.1f},{r_max_fit:.1f}] n_band={n_band} "
    f"r_inner={r_inner:.1f} r_max={r_max:.1f}",
    flush=True,
)


def drive_stats():
    phi = np.array(cl.leaf_phi).T  # (N,14)
    inward = np.where(phi < 0.0, -phi, 0.0)
    drive = inward @ _W
    d_band = drive[in_band]
    n_above = int((d_band > M0).sum())
    max_drive = float(d_band.max()) if d_band.size else 0.0
    p99_drive = float(np.percentile(d_band, 99)) if d_band.size else 0.0
    max_phi = float(np.abs(phi[in_band]).max()) if in_band.sum() else 0.0
    return max_drive, p99_drive, n_above, max_phi


for step in range(N_STEPS + 1):
    if step % LOG_EVERY == 0:
        max_d, p99_d, n_above, max_phi = drive_stats()
        print(
            f"step={step:6d}  t={step*dt:8.3f}  periods={step*dt/period:6.3f}  "
            f"max_drive={max_d:.4e}  p99_drive={p99_d:.4e}  n_above_m0={n_above:6d}/{n_band}  "
            f"max|phi|={max_phi:.4e}",
            flush=True,
        )
    alat.step(0, dt)
    alat.step_sponge(0)
    if CROSS_SCALE > 0.0:
        alat.step_cross(0, dt, CROSS_SCALE, gate="barrier", m0=M0)

print("done", flush=True)
