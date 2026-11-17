"""
Probe: does step_cross's new sigma(lam)^2 factor actually resist topology-opening
for high-lam knots?

Single toroidal ring (no Ring B, no step_lam dynamics — lam is frozen per run) so
the only thing varying between runs is the ring's own lam value. Cold vacuum
(no Saha noise — needed to see clean decay, per the earlier finding that thermal
noise swamps the decay signal). Same cross_scale, amplitude, geometry across all
runs; only lam differs.

Prediction:
  Before the fix: circ_A(t) decay curves should be identical (within noise) across
  all lam values — step_cross ignored lam entirely.
  After the fix: higher lam should give slower decay (sigma(lam)^2 throttles the
  redistribution rate), i.e. a real resistive/stability-from-mass effect.

Run:
  PYTHONPATH=$PWD python3 run/substrate/probe_step_cross_lam_resistance.py
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.substrate.flux_tube import (
    _MIRROR,
    find_toroidal_ring,
    phi_ring_circulation,
)
from equilibrium.substrate.lattice import DampedLattice


def set_dc_circulation(lat, tiles, faces, amplitude=1.0):
    """Genuine unfrozen closed-loop circulation (not the zero-mean resonant
    standing wave) — real nonzero winding for step_cross to actually decay."""
    n = len(tiles)
    for j in range(n):
        i = tiles[j]
        nxt = tiles[(j + 1) % n]
        k = faces[j]
        mk = int(_MIRROR[k])
        lat.phi[i, k] = amplitude
        lat.phi[nxt, mk] = -amplitude


def main():
    R = 20.0
    CROSS_SCALE = 0.02
    N_STEPS = 20000
    REPORT = 50
    LAM_VALUES = [0.0, 1.0, 5.0, 20.0, 100.0]

    results = {}

    for lam_val in LAM_VALUES:
        lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
        A_tiles, A_faces = find_toroidal_ring(lat, axis="z")
        if A_tiles is None:
            print("ERROR: no ring found")
            sys.exit(1)
        set_dc_circulation(lat, A_tiles, A_faces, amplitude=1.0)
        if lam_val > 0.0:
            for i in A_tiles:
                lat.lam[i] = lam_val

        centre_A = lat.pos[A_tiles].mean(axis=0)
        dt = lat.stable_dt()

        n_rep = N_STEPS // REPORT + 1
        t_hist = np.zeros(n_rep)
        circ_h = np.zeros(n_rep)

        si = 0
        circ_h[0] = phi_ring_circulation(lat, A_tiles, centre_A, axis="z")
        t_hist[0] = 0.0

        for step in range(1, N_STEPS + 1):
            lat.step(dt)
            lat.step_cross(dt, scale=CROSS_SCALE)
            if step % REPORT == 0:
                si += 1
                t_hist[si] = step * dt
                circ_h[si] = phi_ring_circulation(lat, A_tiles, centre_A, axis="z")

        results[lam_val] = (t_hist[: si + 1], circ_h[: si + 1])
        env = np.abs(circ_h[: si + 1])
        print(
            f"lam={lam_val:6.1f}  sigma^2={1.0/(1.0+lam_val)**2:.4e}  "
            f"|circ_A|(t=0)={env[0]:.4e}  |circ_A|(final)={env[-1]:.4e}  "
            f"remaining_frac={env[-1]/max(env[0],1e-30):.4e}"
        )

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    for i, lam_val in enumerate(LAM_VALUES):
        t_hist, circ_h = results[lam_val]
        color = cmap(i / max(len(LAM_VALUES) - 1, 1))
        ax.semilogy(t_hist, np.abs(circ_h) + 1e-30, color=color, label=f"lam={lam_val}")
    ax.set_xlabel("time")
    ax.set_ylabel("|circ_A| (log)")
    ax.set_title(f"step_cross lam-resistance test  cross_scale={CROSS_SCALE}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("tmp/step_cross_lam_resistance.png", dpi=130)
    print("\nplot saved -> tmp/step_cross_lam_resistance.png")


if __name__ == "__main__":
    main()
