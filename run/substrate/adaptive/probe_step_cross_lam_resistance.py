"""
Probe: does step_cross's sigma(lam)^2 factor actually resist topology-opening
for high-lam knots — adaptive lattice testbed.

Mirrors run/substrate/probe_step_cross_lam_resistance.py on the AdaptiveLattice
C++ backend, using the same lam-resistance fix now ported to
AdaptiveLattice::step_cross (inc/eq/core/data/lattice/adaptive_lattice.hpp,
_step_cross_local / _step_cross_mirror).

Single toroidal ring (no Ring B, no step_lam dynamics — lam is frozen per run)
so the only thing varying between runs is the ring's own lam value. Sponge BC
only (no Saha noise) — needed to see clean decay, per the earlier finding that
thermal noise swamps the decay signal (never combine Saha noise with a decay
probe — see doc/mems/feedback_saha_noise_scope.md).

Prediction (same as the uniform version): higher lam gives slower decay
(sigma(lam)^2 throttles the redistribution rate) — a real resistive/
stability-from-mass effect, now also present on the adaptive backend.

Run:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_step_cross_lam_resistance.py
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    _MIRROR,
    find_toroidal_ring,
    phi_ring_circulation,
)


def set_dc_circulation(cluster, ring_js, ring_faces, amplitude=1.0):
    """Genuine unfrozen closed-loop circulation (not a zero-mean resonant
    standing wave) — real nonzero winding for step_cross to actually decay."""
    phi = np.array(cluster.leaf_phi)  # (14, N_leaves)
    n = len(ring_js)
    for idx in range(n):
        j_i = ring_js[idx]
        j_nxt = ring_js[(idx + 1) % n]
        k = ring_faces[idx]
        mk = int(_MIRROR[k])
        phi[k, j_i] = amplitude
        phi[mk, j_nxt] = -amplitude
    cluster.set_leaf_phi(phi)


def main():
    R = 20.0
    CROSS_SCALE = 0.02
    N_STEPS = 20000
    REPORT = 50
    LAM_VALUES = [0.0, 1.0, 5.0, 20.0, 100.0]
    SPONGE_FRACTION = 0.15
    DAMP_MAX = 0.003

    results = {}

    for lam_val in LAM_VALUES:
        alat = _lat.AdaptiveLattice()
        alat.add_cluster()
        cl = alat[0]

        _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
        alat.rebuild(0)

        r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
        r_max = float(r_all.max())
        r_inner = r_max * (1.0 - SPONGE_FRACTION)
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge(
                target=0.0, damp_max=DAMP_MAX, r_all=r_all, r_inner=r_inner
            )
        )

        A_js, A_faces = find_toroidal_ring(cl, axis="z")
        if A_js is None:
            print("ERROR: no ring found")
            sys.exit(1)
        set_dc_circulation(cl, A_js, A_faces, amplitude=1.0)
        if lam_val > 0.0:
            lam = np.array(cl.leaf_lam)
            for i in A_js:
                lam[i] = lam_val
            cl.set_leaf_lam(lam)

        centre_A = np.array(cl.leaf_pos)[A_js].mean(axis=0)
        dt = compute_dt(cl)

        n_rep = N_STEPS // REPORT + 1
        t_hist = np.zeros(n_rep)
        circ_h = np.zeros(n_rep)

        si = 0
        circ_h[0] = phi_ring_circulation(cl, A_js, centre_A)
        t_hist[0] = 0.0

        for step in range(1, N_STEPS + 1):
            alat.step(0, dt)
            alat.step_sponge(0)
            alat.step_cross(0, dt, CROSS_SCALE)
            if step % REPORT == 0:
                si += 1
                t_hist[si] = step * dt
                circ_h[si] = phi_ring_circulation(cl, A_js, centre_A)

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
    ax.set_title(
        f"step_cross lam-resistance test (adaptive)  cross_scale={CROSS_SCALE}"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("tmp/step_cross_lam_resistance_adaptive.png", dpi=130)
    print("\nplot saved -> tmp/step_cross_lam_resistance_adaptive.png")


if __name__ == "__main__":
    main()
