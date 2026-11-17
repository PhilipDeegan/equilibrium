"""
Discrete Kelvin-cell photon deflection test.

Lattice  : spherical, R=6, ~1200 cells.
Well     : lam(hop) = chi_0 / (hop - chi_0)  →  sigma(hop) = (hop-chi_0)/hop
           Centre cell (hop=0): lam → large, sigma → 0  (near-BH)
           Outer cells:         lam → 0,     sigma → 1  (void)
Mechanism: throughput gradient sigma = 1/(1+lam).
           No chi field.  No ell variation.  No sub-cycling.
           Time dilation = 1+lam relative to void.

Expected : near-BH side of wavefront has lower throughput → propagates slower
           → wavefront tilts → path bends toward mass.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.substrate.discrete import (
    LAM_MAX,
    DiscreteLattice,
    init_photon_packet,
    photon_energy_centroid,
)

CHI_0 = 0.9


def lam_func(hop):
    if hop == 0:
        return LAM_MAX  # BH core: sigma = 1/2880
    denom = hop - CHI_0
    if denom <= 0:
        return LAM_MAX
    return CHI_0 / denom


def main():
    # ── lattice ───────────────────────────────────────────────────────────────
    R = 6
    lat = DiscreteLattice(radius_cells=R)
    print(f"Lattice: N={lat.N} cells, R={R}")

    # ── winding profile ───────────────────────────────────────────────────────
    # sigma_target(hop) = (hop - chi_0) / hop  →  lam = chi_0 / (hop - chi_0)
    # hop=0 (BH): lam = large sentinel  →  sigma ≈ 0
    # hop=1:  chi_0=0.9  →  lam = 9,   sigma ≈ 0.10
    # hop=2:               lam ≈ 0.82, sigma ≈ 0.55
    # hop=3:               lam ≈ 0.43, sigma ≈ 0.70
    # hop=4+:              lam → 0,    sigma → 1.0

    centre_idx = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    lat.set_radial_lam(centre_idx, lam_func)

    sigma = lat.sigma
    print(f"lam  range: {lat.lam.min():.2e} – {lat.lam.max():.2e}")
    print(f"sigma range: {sigma.min():.4f} – {sigma.max():.4f}")

    # ── photon ────────────────────────────────────────────────────────────────
    B = 2.5
    origin = np.array([-R + 1.5, 0.0, B])
    direction = np.array([1.0, 0.0, 0.0])
    WAVELENGTH = 2.0

    init_photon_packet(lat, origin, direction, WAVELENGTH, amplitude=1.0)
    fwd_mask = lat.pos[:, 0] > origin[0]
    print(f"Photon: origin={origin}, b={B}, λ={WAVELENGTH}")

    # ── run ───────────────────────────────────────────────────────────────────
    dt = lat.stable_dt()
    print(f"dt = {dt:.5f}")

    N_STEPS = 1500
    RECORD = 5
    steps_rec, path_all, path_fwd = [], [], []

    for step in range(N_STEPS):
        if step % RECORD == 0:
            steps_rec.append(step)
            path_all.append(photon_energy_centroid(lat).copy())
            path_fwd.append(photon_energy_centroid(lat, mask=fwd_mask).copy())
        lat.step(dt)

    steps_rec = np.array(steps_rec)
    path_all = np.array(path_all)
    path_fwd = np.array(path_fwd)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    theta = np.linspace(0, 2 * np.pi, 300)

    ax = axes[0]
    ax.plot(
        path_all[:, 0], path_all[:, 2], "b-", alpha=0.4, lw=1.5, label="centroid (all)"
    )
    ax.plot(path_fwd[:, 0], path_fwd[:, 2], "r-", lw=2, label="centroid (fwd)")
    ax.axhline(B, color="gray", ls="--", lw=1, label=f"straight  z={B}")
    ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.4)
    ax.plot(R * np.cos(theta), R * np.sin(theta), "k:", alpha=0.15)
    ax.scatter([0], [0], c="k", s=100, zorder=5, label="BH centre")
    ax.set_xlabel("x (cell units)")
    ax.set_ylabel("z (cell units)")
    ax.set_title("Photon path — x-z plane")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(steps_rec, path_fwd[:, 2] - B, "r-", lw=2, label="fwd Δz")
    ax.plot(steps_rec, path_all[:, 2] - B, "b-", alpha=0.4, lw=1.5, label="all Δz")
    ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("Δz from straight path")
    ax.set_title("Deflection vs time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    r_rad = np.linalg.norm(lat.pos - lat.pos[centre_idx], axis=1)
    order = np.argsort(r_rad)
    ax.scatter(r_rad[order], sigma[order], s=6, alpha=0.5, c="steelblue")
    ax.set_xlabel("radial distance (cell units)")
    ax.set_ylabel("σ = 1/(1+lam)")
    ax.set_title(f"Throughput profile  chi_0={CHI_0}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Discrete Kelvin lattice — photon deflection  (R={R}, b={B}, λ={WAVELENGTH})\n"
        f"sigma=1/(1+lam) — no chi field — no ell variation — no sub-cycling"
    )
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "photon_deflection.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    dz = path_fwd[-1, 2] - path_fwd[0, 2]
    sign = "toward centre" if dz < 0 else "away from centre"
    print(f"Net z deflection (fwd): {dz:+.4f} cell units  ({sign})")


if __name__ == "__main__":
    main()
