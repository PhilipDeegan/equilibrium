"""
CFL sweep (probe_dt_cfl_check.py) ruled out plain numerical instability: at
dt0 and dt0/3, max|psi| converges to the SAME value (1.390) at the SAME
physical time (n_steps*dt=10.25), not a dt-dependent blowup -- this is a
converged (in dt) solution of the real underlying ODE, not an integrator
artifact. Still growing at that point though (not yet plateaued), consistent
with the original 3000-step run reaching max|psi|~2.0 by physical
time~15.4.

Decisive test per Philip's leak-vs-loss framing: track TOTAL ENERGY
(H = u^2 + 0.5*phi^2 + 0.5*c2*sum(psi^2)), not just max|psi|, over a MUCH
longer physical time. The continuum (phi,psi) sub-ODE provably conserves
energy exactly (D and D^T are exact adjoints). If H stays bounded/returns
(oscillates, no secular drift) even while psi/phi individually swap energy
back and forth, that's a real but BENIGN sector-coupling (bounded exchange,
not loss). If H itself grows without bound, that's a genuine defect in the
discrete scheme (numerical energy injection), independent of the ordering
question already answered (both orderings misbehaved identically).
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium.substrate.discrete_curl import triangle_triples
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

R = 22.0
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
C2 = 1e-3
N_STEPS = 60000
LOG_EVERY = 200


def circ_D(phi, triples):
    out = []
    for (h1, h2, sqk), a, b, c in triples:
        m = _MIRROR[sqk]
        out.append(phi[a, h1] + phi[b, h2] + phi[c, m])
    return out


def scatter_DT(vals, phi_shape, triples):
    inc = np.zeros(phi_shape)
    for (fam, v) in zip(triples, vals):
        (h1, h2, sqk), a, b, c = fam
        m = _MIRROR[sqk]
        np.add.at(inc, (a, np.full_like(a, h1)), v)
        np.add.at(inc, (b, np.full_like(b, h2)), v)
        np.add.at(inc, (c, np.full_like(c, m)), v)
    return inc


def symplectic_step_baseline(lat, dt, valid, safe_nb):
    div_phi = np.sum(np.where(valid, W[None, :] * lat.phi, 0.0), axis=1)
    u_new = lat.u - dt * div_phi
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * W[None, :] * drive
    return u_new, dphi


def main():
    lat = SubstrateLattice(radius_cells=R)
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)

    u0 = np.exp(-0.5 * (np.linalg.norm(lat.pos, axis=1) / 4.0) ** 2)
    lat.u = u0
    lat.phi = np.where(valid, u0[safe_nb] - u0[:, None], 0.0)
    psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]

    def total_h():
        return float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2) + 0.5 * C2 * sum(np.sum(p**2) for p in psi))

    h0 = total_h()
    print(f"N={lat.N} R={R} dt={dt:.6f} c2={C2:.0e} n_steps={N_STEPS} h0={h0:.6f}", flush=True)

    t_start = time.time()
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals = circ_D(lat.phi, triples)
        psi_new = [p - dt * v for p, v in zip(psi, vals)]
        extra = scatter_DT(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi = u_new, lat.phi + dphi_base + dt * C2 * extra, psi_new
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            mp = max(float(np.max(np.abs(p))) for p in psi)
            hr = total_h() / h0
            t_phys = (step + 1) * dt
            print(
                f"step={step:6d} t_phys={t_phys:8.2f} max|psi|={mp:.4e} H_ratio={hr:.8f} "
                f"wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
            if not np.isfinite(hr) or hr > 1e6:
                print("BLEW UP, stopping early", flush=True)
                break


if __name__ == "__main__":
    main()
