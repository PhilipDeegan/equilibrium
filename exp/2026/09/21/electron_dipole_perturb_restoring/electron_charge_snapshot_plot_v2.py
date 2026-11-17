"""
v2: fixes v1's uninformative plot. The deficit (width=15) visually swamps
the whole frame on a fixed +-0.2 color scale, and the actual dynamics
(1-3% change over t=2.05, per v4's frac_rate) are invisible against that.

Fix: plot u at t=0 next to the DIFFERENCE delta_u = u(t_final) - u(t=0),
with its own auto-scaled (much smaller-range) color map -- that's where
the actual signal lives. Also saves the raw slice arrays to npz so this
doesn't need re-simulating to iterate on the plot again.
"""

import sys

sys.path.insert(0, "/home/p/git/eq")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

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
SLICE_HALF_WIDTH = 1.0
PLOT_R = 45.0  # wider than v1's 20 -- want to see the outgoing wavefront too

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


def run_config(alat, cl, name, deficit, dt, slice_mask):
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
    u_start = u[slice_mask].copy()

    for step in range(N_STEPS):
        cl.set_leaf_phi(phi.copy())
        cl.set_leaf_u(u.copy())
        cl.set_leaf_lam(lam.copy())
        alat.step(0, dt)
        alat.step_sponge(0)
        phi, u = np.array(cl.leaf_phi), np.array(cl.leaf_u)

    u_end = u[slice_mask].copy()
    print(f"[{name}] done", flush=True)
    return u_start, u_end


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

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    out = {"x": x_s, "z": z_s}
    for col, name in enumerate(CONFIGS):
        u_start, u_end = run_config(alat, cl, name, deficit, dt, slice_mask)
        out[f"u_start_{name}"] = u_start
        out[f"u_end_{name}"] = u_end
        du = u_end - u_start

        ax = axes[0, col]
        vmax0 = max(np.abs(u_start).max(), 1e-12)
        sc = ax.scatter(x_s, z_s, c=u_start, cmap="RdBu_r", vmin=-vmax0, vmax=vmax0, s=5)
        ax.set_title(f"{name}: u at t=0")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="u")

        ax = axes[1, col]
        vmax1 = max(np.abs(du).max(), 1e-12)
        # linthresh at ~0.1% of peak so the linear region near zero doesn't
        # dominate -- most of the dynamic range we want (faint far-field
        # wavefront vs strong near-source signal) is in the log regime.
        norm = SymLogNorm(linthresh=max(vmax1 * 1e-3, 1e-15), vmin=-vmax1, vmax=vmax1)
        sc = ax.scatter(x_s, z_s, c=du, cmap="RdBu_r", norm=norm, s=5)
        ax.set_title(f"{name}: delta_u = u(t={N_STEPS*dt:.2f}) - u(t=0)\n(symlog scale, max|du|={vmax1:.2e})")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="delta_u (symlog)")

    fig.suptitle(
        f"XZ slice (|y|<{SLICE_HALF_WIDTH}), R_domain={R_DOMAIN}, electron_scale={ELECTRON_SCALE}, "
        f"deficit_amp={DEFICIT_AMP}, deficit_width={width:.1f}"
    )
    fig.tight_layout()
    out_path = "exp/2026/09/21/electron_dipole_perturb_restoring/charge_snapshot_diff.png"
    fig.savefig(out_path, dpi=130)
    np.savez("exp/2026/09/21/electron_dipole_perturb_restoring/charge_snapshot_slices.npz", **out)
    print(f"saved {out_path} and slice data", flush=True)


if __name__ == "__main__":
    main()
