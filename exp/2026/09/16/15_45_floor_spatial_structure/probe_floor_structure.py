"""
Visualize the spatial structure of the outflow dipole floor found
2026-09-16 (~1.9e-4 of IC, non-decaying) -- the ledger-instrumentation
session found the boundary goes essentially inert by round trip ~4 while
the domain's *own* u-energy keeps falling for many more round trips before
settling into the floor, and that surviving floor energy is mostly stored
in phi, not u (whole-domain u^2+phi^2 floor is flat at 1.86e-4 while pure
u^2 alone keeps dropping past round trip 4). This script dumps per-leaf
energy density (u^2 + sum_k phi[k]^2) at several times and renders XZ
slices to see WHERE that residual lives -- concentrated near the two
Gaussian source poles, spread over the outer shell, or something else.

Same dipole IC/outflow BC/geometry as
exp/2026/09/16/12_30_outflow_taper_probe/probe_outflow_taper.py's --ic
dipole, R=10 default.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt

R = 10.0
AMP = 5.0
SIGMA = 2.0
DIPOLE_SEP = 6.0
C_LATTICE = np.sqrt(10.0)
N_ROUND_TRIPS = 15.0
SLAB_HALF_THICKNESS = 0.75
OUT_PREFIX = "exp/2026/09/16/15_45_floor_spatial_structure/floor"

alat = _lat.AdaptiveLattice()
alat.add_cluster()
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, lam0=0.0)
alat.rebuild(0)
alat.set_n_threads(8)

pos = np.array(cl.leaf_pos)
r = np.linalg.norm(pos, axis=1)
d = 0.5 * DIPOLE_SEP
cp = np.array([0.0, 0.0, d])
cm = np.array([0.0, 0.0, -d])
rp = np.linalg.norm(pos - cp, axis=1)
rm = np.linalg.norm(pos - cm, axis=1)
u_ic = AMP * np.exp(-(rp**2) / (2.0 * SIGMA**2)) - AMP * np.exp(-(rm**2) / (2.0 * SIGMA**2))
cl.set_leaf_u(u_ic.copy())

nb = np.array(cl.leaf_nb)
cl.set_bc(_lat.BoundaryCondition.outflow(nb, c_lattice=C_LATTICE))

dt = compute_dt(cl)
t_round_trip = 2.0 * R / C_LATTICE  # r_ring=0 for a point-like dipole IC
print(f"N={cl.n_leaves} dt={dt:.4e} t_round_trip={t_round_trip:.3f}", flush=True)

energy_ic = float(np.sum(u_ic**2))

slab_mask = np.abs(pos[:, 1]) < SLAB_HALF_THICKNESS
x_slab = pos[slab_mask, 0]
z_slab = pos[slab_mask, 2]


def energy_density():
    u = np.array(cl.leaf_u)
    phi = np.asarray(cl.leaf_phi)  # (14, N)
    return u**2 + np.sum(phi**2, axis=0)


def save_frame(round_trip, t_phys):
    e = energy_density()
    e_slab = e[slab_mask]
    finite_pos = e_slab[e_slab > 0]
    vmin = np.log10(np.percentile(finite_pos, 1)) if finite_pos.size else -12
    vmax = np.log10(np.percentile(finite_pos, 99)) if finite_pos.size else 0
    c = np.log10(np.clip(e_slab, 10.0**vmin, None))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(x_slab, z_slab, c=c, s=14, cmap="inferno", vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(
        f"round_trip={round_trip} t={t_phys:.2f}  log10(u^2+phi^2) in [{vmin:.1f},{vmax:.1f}]\n"
        f"whole-domain u^2 sum / IC = {float(np.sum(np.array(cl.leaf_u)**2))/energy_ic:.3e}",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z (dipole axis)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log10 energy density")
    fig.tight_layout()
    out_path = f"{OUT_PREFIX}_rt{round_trip:02d}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


round_trips_to_save = {0, 1, 2, 4, 6, 9, 12, 15}
n_steps = int(round(N_ROUND_TRIPS * t_round_trip / dt))

save_frame(0, 0.0)
next_rt = 1
for step in range(n_steps):
    alat.step(0, dt)
    t_phys = (step + 1) * dt
    if t_phys >= next_rt * t_round_trip - 1e-9:
        if next_rt in round_trips_to_save:
            save_frame(next_rt, t_phys)
        next_rt += 1
        if next_rt > N_ROUND_TRIPS:
            break

print("done", flush=True)
