"""
Philip (2026-09-04): step back from knot/decay work entirely -- does a
fixed electrical-potential difference in bare VACUUM (no structures, no
lam, no step_cross) produce a real current that relaxes toward
equilibrium, the way a real capacitor/battery-plus-resistor circuit does?

Setup: two point-like fixed-u sources ("battery terminals"), u=+U0 and
u=-U0, embedded in otherwise-empty vacuum (phi=0, lam=0 at t=0). No
circulation, no knot, no step_cross -- just plain step() (the same
continuity-equation/flux-exchange update already confirmed correct in the
2026-08-27 Gauss/Coulomb probes: du_i=-dt*sigma*sum_k W_k*phi[i,k],
dphi[i,k]=dt*sigma*W_k*(u_i-u_nb)) with the two source cells' own u held
fixed (fixed_u=True), everywhere else free.

Diagnostic: "current" = net phi flux through the mid-plane between the two
sources (a disk of cells at the midpoint, oriented along the source-to-
source axis) -- summed W_k*phi[i,k] over faces crossing that plane in the
source-to-source direction, same accounting the Gauss-law charge tests use
(Q_i = sum_k A_ik * flux). Prediction if DST reproduces Ohm's-law-like
relaxation: this current rises then DECAYS toward zero as the field
settles into the static two-source (dipole-like) equilibrium -- an
RC-discharge shape, not a persistent or ringing current. A damped sponge
BC is used (damp_max>0) since the 2026-08-27 Gauss-law work already showed
damp_max=0 never settles at all (symplectic dynamics ring forever).
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

R = 40.0
SEP = 16.0  # distance between the two fixed-u sources
U0 = 1.0
N_STEPS = 20000
REPORT = 500
N_THREADS = 8
MIDPLANE_HALF_WIDTH = 1.0  # cells within this of x=0 count as "midplane"


def build(lat):
    pos = lat.pos
    src_a = np.array([-SEP / 2, 0.0, 0.0])
    src_b = np.array([SEP / 2, 0.0, 0.0])
    ia = int(np.argmin(np.linalg.norm(pos - src_a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(pos - src_b, axis=1)))
    lat.u[ia] = U0
    lat.u[ib] = -U0
    lat.fixed_u[ia] = True
    lat.fixed_u[ib] = True
    return ia, ib


def midplane_current(lat):
    """Net phi flux through the x=0 plane, from -x side toward +x side."""
    pos, nb = lat.pos, lat.nb
    near = np.abs(pos[:, 0]) <= MIDPLANE_HALF_WIDTH
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    nbr_x = pos[safe_nb, 0]
    crossing = valid & near[:, None] & (
        ((pos[:, 0, None].repeat(14, axis=1) < 0) & (nbr_x > 0))
        | ((pos[:, 0, None].repeat(14, axis=1) > 0) & (nbr_x < 0))
    )
    # sign: positive = flowing in +x direction (from - source toward + source's
    # opposite side is arbitrary; consistent sign is all that matters)
    sign = np.where(pos[:, 0, None].repeat(14, axis=1) < 0, 1.0, -1.0)
    W = np.where(np.arange(14) < 8, 3.0, 1.0)
    return float(np.sum(np.where(crossing, sign * W[None, :] * lat.phi, 0.0)))


def run():
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    ia, ib = build(lat)
    dt = lat.stable_dt()
    print(f"N={lat.N}  dt={dt:.4e}  src_a={lat.pos[ia]}  src_b={lat.pos[ib]}", flush=True)

    history = []
    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            I = midplane_current(lat)
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            u_rms = float(np.sqrt(np.mean(lat.u**2)))
            history.append((step, I, H, u_rms))
            print(f"  {step:6d}  I_mid={I:+.6e}  H={H:.6e}  u_rms={u_rms:.6e}", flush=True)
        lat.step(dt, n_threads=N_THREADS)

    print("\n=== summary ===")
    Is = np.array([h[1] for h in history])
    peak_idx = int(np.argmax(np.abs(Is)))
    print(f"peak |I_mid|={abs(Is[peak_idx]):.6e} at step {history[peak_idx][0]}")
    print(f"final I_mid={Is[-1]:+.6e} (step {history[-1][0]})")
    print(f"final/peak ratio={abs(Is[-1])/abs(Is[peak_idx]) if Is[peak_idx] else float('nan'):.6f}")
    return history


if __name__ == "__main__":
    run()
