"""At-a-glance topology review of every distinct dipole/electron-IC
construction in this repo -- built following the 2026-09-21 finding that
`init_electron_dipole_ic`/`_vecpot` (used in nearly every electron test
since 2026-07-10) has the WRONG topology (closed/hollow/neutron-lineage,
not open/compact/proton-lineage), and that `dipole_mode_ic.seed_dipole_ic`
is the corrected production replacement. Before committing to the next
concrete probe (rebuilding confinement/charge/g-factor tests on
dipole_mode_ic.py, per `next`'s item #1), this makes the shape difference
across ALL existing constructions visible side-by-side rather than asserted
from memory.

Structural review only -- no time-stepping, just seed each IC at t=0 on a
matched R_DOMAIN=80/scale=50 domain (the resolution 09-21 validated as
properly resolved, exp/2026/09/21/electron_dipole_perturb_restoring/
status.md) and plot its energy density (u^2 + 0.5*sum(phi^2)) in a
meridional (x-z) slice.

Excluded, with reasons (see status.md for detail):
- init_dipole_ic / init_phare_ic (adaptive_flux_tube.py): both require a
  PHARE relaxed-spheromak npz (ic_b.npz) that does not exist anywhere in
  this environment -- checked via `find`, not present.
- init_toroidal_winding: a lower-level primitive that seeds an
  ALREADY-FOUND ring (needs find_toroidal_ring first), not itself a
  dipole/electron IC -- every other construction here calls something at
  this level internally, or replaces it.
"""

import sys

sys.path.insert(0, "/home/deegan/git/eq")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe, ellipk

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic
from equilibrium.substrate.curl_full_spheromak_ic import seed_curl_full_pair
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic
from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _FACE_NORM,
    init_electron_dipole_ic,
    init_electron_dipole_ic_vecpot,
    init_spheromak_ic,
)

R_DOMAIN = 80.0
SCALE = 50.0  # R_sph*SCALE = 50 DST units -- matches 09-21's validated ELECTRON_SCALE
R_SPH = 1.0
N_THREADS = 8
SLICE_HALF_WIDTH = 1.0  # cells either side of y=0 for the x-z meridional slice
PLOT_R = 65.0


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)
    return alat, cl


def energy_density(cl):
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return u**2 + 0.5 * np.sum(phi**2, axis=0)


def meridional_slice(cl, energy):
    pos = np.array(cl.leaf_pos)
    mask = np.abs(pos[:, 1]) <= SLICE_HALF_WIDTH
    return pos[mask, 0], pos[mask, 2], energy[mask]


# ---------------------------------------------------------------------------
# Bare current-loop dipole (Ampere's law, A.dl seeding) -- ported from
# exp/2026/09/02/20_00_proton_loop_dipole/probe_curl_loop_dipole.py's exact
# classical circular-loop A_phi (elliptic-integral form) onto the
# AdaptiveLattice leaf/face API, same A.dl sampling pattern already used by
# init_electron_dipole_ic_vecpot (A sampled at face midpoints, dotted with
# the discrete edge displacement _FACE_HOP -- no continuum derivative of A
# anywhere). That original probe used a plain uniform SubstrateLattice with
# its own pos/nb arrays -- a different backend than every other construction
# here -- so this is a genuine, deliberate port, not a copy-paste reuse.
# u is left at zero: this is a pure current-loop construction, no scalar
# deficit/background field.
# ---------------------------------------------------------------------------
LOOP_A = 30.0  # loop radius in DST units, comparable to the double-ring
# dipole's own r_axis (~0.63*SCALE = 31.5)


def _loop_a_phi(rho, z, a):
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    k2 = np.clip(4.0 * a * safe_rho / ((a + safe_rho) ** 2 + z**2), 0.0, 1.0 - 1e-12)
    K = ellipk(k2)
    E = ellipe(k2)
    val = (
        (1.0 / np.pi)
        * np.sqrt(a / safe_rho)
        / np.sqrt(k2)
        * ((1.0 - k2 / 2.0) * K - E)
    )
    return np.where(rho > 1e-6, val, 0.0)


def seed_bare_loop_dipole(cl, pos, face_hop, loop_a, amplitude=1.0):
    N = pos.shape[0]
    midpoints = (pos[:, None, :] + 0.5 * face_hop[None, :, :]).reshape(-1, 3)
    rho = np.linalg.norm(midpoints[:, :2], axis=1)
    z = midpoints[:, 2]
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    a_phi = _loop_a_phi(rho, z, loop_a)
    phihat_x = -midpoints[:, 1] / safe_rho
    phihat_y = midpoints[:, 0] / safe_rho
    A = np.stack(
        [a_phi * phihat_x, a_phi * phihat_y, np.zeros_like(a_phi)], axis=1
    ).reshape(N, 14, 3)
    phi = np.einsum("nkd,kd->kn", A, face_hop)  # (14, N), A.dl
    phi_max = float(np.abs(phi).max()) or 1.0
    phi *= amplitude / phi_max
    cl.set_leaf_phi(np.ascontiguousarray(phi))
    cl.set_leaf_u(np.zeros(N))
    return phi


# ---------------------------------------------------------------------------
# Panel registry: (title, build_fn) -- each build_fn(cl, pos, cell_size)
# seeds cl in place.
# ---------------------------------------------------------------------------


def _seed_electron_dipole(cl, pos, cell_size):
    init_electron_dipole_ic(cl, R_sph=R_SPH, scale=SCALE, amplitude=1.0)


def _seed_electron_dipole_vecpot(cl, pos, cell_size):
    init_electron_dipole_ic_vecpot(cl, R_sph=R_SPH, scale=SCALE, amplitude=1.0)


def _seed_spheromak(cl, pos, cell_size):
    init_spheromak_ic(cl, R_sph=R_SPH, scale=SCALE, amplitude=1.0)


def _seed_dipole_mode(cl, pos, cell_size):
    seed_dipole_ic(
        cl, pos, cell_size, SCALE, _FACE_HOP, _FACE_NORM, r_sph=R_SPH, amplitude=1.0
    )


def _seed_curl_full_dipole(cl, pos, cell_size):
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


def _seed_curl_full_pair(cl, pos, cell_size):
    result = seed_curl_full_pair(cl, hops=35, amplitude_A=1.0, amplitude_B=1.0)
    if result is None:
        raise RuntimeError("seed_curl_full_pair: ring not found at hops=35")


def _seed_bare_loop(cl, pos, cell_size):
    seed_bare_loop_dipole(cl, pos, _FACE_HOP, LOOP_A, amplitude=1.0)


PANELS = [
    ("init_electron_dipole_ic\n(adaptive_flux_tube.py -- FLAGGED WRONG)", _seed_electron_dipole),
    ("init_electron_dipole_ic_vecpot\n(adaptive_flux_tube.py -- vecpot fix, same topology)", _seed_electron_dipole_vecpot),
    ("init_spheromak_ic\n(adaptive_flux_tube.py -- neutron/closed reference)", _seed_spheromak),
    ("seed_dipole_ic\n(dipole_mode_ic.py -- CORRECTED production IC)", _seed_dipole_mode),
    ("seed_curl_full_dipole_ic\n(curl_full_dipole_ic.py -- poloidal+toroidal)", _seed_curl_full_dipole),
    ("seed_curl_full_pair\n(curl_full_spheromak_ic.py -- Hopf-linked filaments)", _seed_curl_full_pair),
    ("bare current loop, A.dl seeding\n(ported from probe_curl_loop_dipole.py)", _seed_bare_loop),
]


def main():
    n = len(PANELS)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, seed_fn) in zip(axes, PANELS):
        print(f"--- building: {title.splitlines()[0]} ---", flush=True)
        alat, cl = build_cluster()
        pos = np.array(cl.leaf_pos)
        cell_size = np.full(cl.n_leaves, 1.0)
        try:
            seed_fn(cl, pos, cell_size)
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            ax.set_title(f"{title}\nFAILED: {exc}", fontsize=8, color="red")
            ax.axis("off")
            continue
        energy = energy_density(cl)
        x, z, e = meridional_slice(cl, energy)
        print(
            f"    N={cl.n_leaves}  slice_pts={len(x)}  e_max={e.max():.4e}"
            f"  e_sum={e.sum():.4e}",
            flush=True,
        )
        sc = ax.scatter(x, z, c=e, s=3, cmap="inferno", vmin=0.0, vmax=max(e.max(), 1e-12))
        ax.set_xlim(-PLOT_R, PLOT_R)
        ax.set_ylim(-PLOT_R, PLOT_R)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Dipole/electron IC gallery -- meridional (x-z) energy-density slices,"
        f" R_domain={R_DOMAIN:.0f}, scale={SCALE:.0f}, R_sph={R_SPH:.1f} (2026-09-22)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "exp/2026/09/22/dipole_ic_gallery/dipole_ic_gallery.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
