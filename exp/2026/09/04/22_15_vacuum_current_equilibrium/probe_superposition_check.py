"""
Philip's question (2026-09-04): is the combined two-source run's I_mid a
genuine A<->B interaction ("current reaching toward the other"), or just
each fixed-u source's own IC gradient radiating outward independently,
which happen to add up (linear superposition, no real interaction at all)?

The update rule is LINEAR in (u, phi): du=-dt*sigma*sum_k W_k*phi,
dphi[i,k]=dt*sigma*W_k*(u_i-u_nb). If fixed_u is also linear (each fixed
cell's constraint doesn't depend on the other source), superposition
predicts I_mid(both) == I_mid(A alone) + I_mid(B alone) EXACTLY (to
floating point) -- i.e. no real interaction, just two independent kicks.
Runs source A alone and source B alone with the identical setup/diagnostic
as probe_vacuum_current.py (same R/SEP/U0/N_STEPS=6000/dt), for direct
comparison against the already-recorded combined run
(exp/2026/09/04/22_15_vacuum_current_equilibrium/output.log).
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

R = 40.0
SEP = 16.0
U0 = 1.0
N_STEPS = 6000
REPORT = 100
N_THREADS = 8
MIDPLANE_HALF_WIDTH = 1.0


def build(lat, which):
    pos = lat.pos
    src_a = np.array([-SEP / 2, 0.0, 0.0])
    src_b = np.array([SEP / 2, 0.0, 0.0])
    if which in ("a", "both"):
        ia = int(np.argmin(np.linalg.norm(pos - src_a, axis=1)))
        lat.u[ia] = U0
        lat.fixed_u[ia] = True
    if which in ("b", "both"):
        ib = int(np.argmin(np.linalg.norm(pos - src_b, axis=1)))
        lat.u[ib] = -U0
        lat.fixed_u[ib] = True


def midplane_current(lat):
    pos, nb = lat.pos, lat.nb
    near = np.abs(pos[:, 0]) <= MIDPLANE_HALF_WIDTH
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    nbr_x = pos[safe_nb, 0]
    x_rep = pos[:, 0, None].repeat(14, axis=1)
    crossing = valid & near[:, None] & (((x_rep < 0) & (nbr_x > 0)) | ((x_rep > 0) & (nbr_x < 0)))
    sign = np.where(x_rep < 0, 1.0, -1.0)
    W = np.where(np.arange(14) < 8, 3.0, 1.0)
    return float(np.sum(np.where(crossing, sign * W[None, :] * lat.phi, 0.0)))


def run(which):
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    build(lat, which)
    dt = lat.stable_dt()
    print(f"\n=== source={which!r} ===  N={lat.N}  dt={dt:.4e}", flush=True)
    history = []
    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            I = midplane_current(lat)
            history.append((step, I))
            if step % 1000 == 0:
                print(f"  {step:6d}  I_mid={I:+.6e}", flush=True)
        lat.step(dt, n_threads=N_THREADS)
    return history


def main():
    hist_a = run("a")
    hist_b = run("b")

    print("\n=== superposition check: I_mid(both, recorded) vs I_mid(A)+I_mid(B) ===")
    print(f"{'step':>6}  {'I_A':>14}  {'I_B':>14}  {'I_A+I_B':>14}  {'I_both(recorded)':>18}  {'match?':>8}")
    # recorded combined-run values, from output.log (6000-step run)
    recorded = {
        0: 0.0, 100: 1.268274e-12, 200: 9.462796e-07, 300: 2.118930e-03, 400: 2.116286e-01,
        500: 2.263817e+00, 600: 5.326450e+00, 700: 8.238656e+00, 800: 1.115261e+01,
        900: 1.407619e+01, 1000: 1.698049e+01, 2000: 4.643760e+01, 3000: 7.071002e+01,
        4000: 9.724516e+01, 5000: 1.172649e+02, 6000: 1.415012e+02,
    }
    for step, ia in hist_a:
        ib = dict(hist_b)[step]
        if step in recorded:
            both = recorded[step]
            summed = ia + ib
            rel = abs(summed - both) / (abs(both) + 1e-30)
            print(f"{step:6d}  {ia:14.6e}  {ib:14.6e}  {summed:14.6e}  {both:18.6e}  {rel:8.2e}")


if __name__ == "__main__":
    main()
