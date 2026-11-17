"""
Empirical check: does init_toroidal_winding's cos(K*idx) construction
produce a STANDING oscillation (fixed phase, amplitude reverses through
zero) or a genuinely TRAVELING pattern (steady phase advance, amplitude
roughly constant) under plain step()?

Method: read phi at each ring tile's own ring-face, project onto the
ring's own K=2*pi/N_ring Fourier mode -- Z(t) = (2/N) * sum_idx
phi[idx] * exp(-i*K*(idx+0.5)) -- and track |Z(t)| (amplitude) and
angle(Z(t)) (phase) over time.  No assumption about which outcome is
"expected" -- just report what the discrete update rule actually does.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_ring_standing_vs_traveling.py
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--hops", type=int, default=6)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--n-steps", type=int, default=2000)
parser.add_argument("--report", type=int, default=25)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--sponge-frac", type=float, default=0.2)
parser.add_argument("--damp-max", type=float, default=0.01)
args = parser.parse_args()


def main():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    ring_js, ring_faces = find_toroidal_ring(
        cl, axis="z", mode="square", hops=args.hops
    )
    if ring_js is None:
        raise RuntimeError("ring-finding failed")
    omega = init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = args.R * (1.0 - args.sponge_frac)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )
    print(f"sponge: r_inner={r_inner:.1f}  damp_max={args.damp_max}")

    N_ring = len(ring_js)
    K = 2.0 * np.pi / N_ring
    idx = np.arange(N_ring)
    basis_u = np.exp(-1j * K * idx)
    basis_phi = np.exp(-1j * K * (idx + 0.5))

    dt = compute_dt(cl)
    print(
        f"N_ring={N_ring}  K={K:.4f}  omega(predicted)={omega:.6f}  dt={dt:.4e}  "
        f"period(steps)~{(2*np.pi/omega)/dt if omega else float('nan'):.1f}"
    )
    print(
        f"\n{'step':>6}  {'t':>9}  {'|Zu|':>9}  {'phaseU_uw':>10}  "
        f"{'|Zphi|':>9}  {'phasePhi_uw':>12}"
    )

    prev_u, prev_phi = None, None
    uw_u, uw_phi = 0.0, 0.0
    for step in range(0, args.n_steps + 1):
        if step % args.report == 0:
            u_arr = np.array(cl.leaf_u)
            u_vals = np.array([u_arr[j] for j in ring_js])
            Zu = (2.0 / N_ring) * np.sum(u_vals * basis_u)
            amp_u, phase_u = abs(Zu), float(np.angle(Zu))

            phi = np.array(cl.leaf_phi)  # (14, N)
            phi_vals = np.array([phi[ring_faces[i], ring_js[i]] for i in range(N_ring)])
            Zp = (2.0 / N_ring) * np.sum(phi_vals * basis_phi)
            amp_p, phase_p = abs(Zp), float(np.angle(Zp))

            if prev_u is not None:
                du = phase_u - prev_u
                du -= 2 * np.pi * round(du / (2 * np.pi))
                uw_u += du
                dp = phase_p - prev_phi
                dp -= 2 * np.pi * round(dp / (2 * np.pi))
                uw_phi += dp
            prev_u, prev_phi = phase_u, phase_p
            print(
                f"{step:6d}  {step * dt:9.3f}  {amp_u:9.4f}  {uw_u:10.4f}  "
                f"{amp_p:9.4f}  {uw_phi:12.4f}"
            )
        alat.step(0, dt)
        alat.step_sponge(0)


if __name__ == "__main__":
    main()
