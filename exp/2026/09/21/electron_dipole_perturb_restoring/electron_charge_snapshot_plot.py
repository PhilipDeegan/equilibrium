"""
Same R=80/electron-scale=50 setup as v4, but instead of the scalar
deficit-window summary, saves an XZ-plane slice (|y|<slice_half_width) of u
at step=0 and step=N_STEPS for a few configs, and plots start-vs-end
side by side. Philip asked to see what's actually happening spatially,
not just summary numbers.

Configs plotted: vacuum (pure dispersal baseline) and dipole_lam2 (phi
structure + strongest static lam tested so far) -- the two ends of what's
been compared all session.
"""

import sys

sys.path.insert(0, "/home/p/git/eq")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import init_electron_dipole_ic_vecpot

R_DOMAIN = 80.0
ELECTRON_SCALE = 50.0
SPONGE_CELLS = 10.0
SPONGE_FRACTION = SPONGE_CELLS / R_DOMAIN
DAMP_MAX = 0.003
N_THREADS = 8

DEFICIT_AMP = 0.2
DEFICIT_WIDTH_FRAC = 0.3
N_STEPS = 400
SLICE_HALF_WIDTH = 1.0  # |y| < this, an XZ-plane slice
PLOT_R = 20.0  # zoom window for the plot, in cells, around the origin

CONFIGS = ["vacuum", "dipole_lam2"]


def build_cluster():
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
    return alat, cl


def run_config(alat, cl, name, pos, deficit, dt, slice_mask):
    if name == "vacuum":
        phi0 = np.zeros((14, cl.n_leaves))
        lam0 = np.zeros(cl.n_leaves)
    else:
        lam_target = {"dipole_lam0": 0.0, "dipole_lam1": 1.0, "dipole_lam2": 2.0}[name]
        init_electron_dipole_ic_vecpot(
            cl, R_sph=1.0, scale=ELECTRON_SCALE, B0=1.0, lam_target=lam_target, u_bg=0.0,
            amplitude=1.0, div_clean=True, clean_rtol=1e-8,
        )
        phi0 = np.array(cl.leaf_phi)
        lam0 = np.array(cl.leaf_lam)

    u0 = np.zeros(cl.n_leaves) - deficit
    phi, u, lam = phi0.copy(), u0.copy(), lam0.copy()

    e0 = u[slice_mask] ** 2 + 0.5 * np.sum(phi[:, slice_mask] ** 2, axis=0)
    snap_start = {"u": u[slice_mask].copy(), "e": e0.copy()}

    for step in range(N_STEPS):
        cl.set_leaf_phi(phi.copy())
        cl.set_leaf_u(u.copy())
        cl.set_leaf_lam(lam.copy())
        alat.step(0, dt)
        alat.step_sponge(0)
        phi, u = np.array(cl.leaf_phi), np.array(cl.leaf_u)

    e1 = u[slice_mask] ** 2 + 0.5 * np.sum(phi[:, slice_mask] ** 2, axis=0)
    snap_end = {"u": u[slice_mask].copy(), "e": e1.copy()}
    print(f"[{name}] done", flush=True)
    return snap_start, snap_end


def main():
    alat, cl = build_cluster()
    dt = compute_dt(cl)
    pos = np.array(cl.leaf_pos)
    r_from_center = np.linalg.norm(pos, axis=1)
    width = DEFICIT_WIDTH_FRAC * ELECTRON_SCALE
    deficit = DEFICIT_AMP * np.exp(-0.5 * (r_from_center / width) ** 2)

    slice_mask = (np.abs(pos[:, 1]) < SLICE_HALF_WIDTH) & (r_from_center < PLOT_R)
    x_s, z_s = pos[slice_mask, 0], pos[slice_mask, 2]
    print(f"N={cl.n_leaves}  slice cells={int(slice_mask.sum())}  t_final={N_STEPS*dt:.2f}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for col, name in enumerate(CONFIGS):
        snap_start, snap_end = run_config(alat, cl, name, pos, deficit, dt, slice_mask)
        vmax = max(np.abs(snap_start["u"]).max(), np.abs(snap_end["u"]).max(), 1e-12)

        ax = axes[0, col]
        sc = ax.scatter(x_s, z_s, c=snap_start["u"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=6)
        ax.set_title(f"{name}: u at t=0")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="u")

        ax = axes[1, col]
        sc = ax.scatter(x_s, z_s, c=snap_end["u"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=6)
        ax.set_title(f"{name}: u at t={N_STEPS*dt:.2f}")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="u")

    fig.suptitle(
        f"XZ slice (|y|<{SLICE_HALF_WIDTH}), R_domain={R_DOMAIN}, electron_scale={ELECTRON_SCALE}, "
        f"deficit_amp={DEFICIT_AMP}"
    )
    fig.tight_layout()
    out_path = "exp/2026/09/21/electron_dipole_perturb_restoring/charge_snapshot_start_end.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
