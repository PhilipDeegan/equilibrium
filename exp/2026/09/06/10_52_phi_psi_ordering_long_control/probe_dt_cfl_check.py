"""
Quick smoke test (500 steps, both orderings, c2=1e-3) already showed the
curl-free control's max|psi| shooting from ~8e-3 to ~0.5-0.6 in 500 steps --
identically under BOTH the original and mirrored leapfrog staggering. Since
both orderings misbehave the SAME way, this is probably not an
ordering/leapfrog-symmetry bug at all: the (phi,psi) sub-system has its own
characteristic frequency (~sqrt(c2)*||D||), and `lat.stable_dt()` was
derived for the ORIGINAL (u,phi) wave speed only -- it was never checked
against the NEW sub-system's own CFL limit. Symplectic Euler on a harmonic
oscillator is only bounded/stable if omega*dt is below a fixed threshold;
past it, EVEN A CORRECT symplectic-Euler pairing diverges exponentially.

This test sweeps dt down (holding c2 fixed) and checks whether the
"leak" simply disappears below some dt -- if so, this was a plain CFL
mismatch for the added degree of freedom, not the ordering question at all.
"""

import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium.substrate.discrete_curl import triangle_triples
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

R = 22.0
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


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


def run(dt, c2, n_steps):
    lat = SubstrateLattice(radius_cells=R)
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)

    u0 = np.exp(-0.5 * (np.linalg.norm(lat.pos, axis=1) / 4.0) ** 2)
    lat.u = u0
    lat.phi = np.where(valid, u0[safe_nb] - u0[:, None], 0.0)
    psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]

    max_psi_trace = []
    for step in range(n_steps):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals = circ_D(lat.phi, triples)
        psi_new = [p - dt * v for p, v in zip(psi, vals)]
        extra = scatter_DT(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi = u_new, lat.phi + dphi_base + dt * c2 * extra, psi_new
        if step % 50 == 0 or step == n_steps - 1:
            mp = max(float(np.max(np.abs(p))) for p in psi)
            max_psi_trace.append((step, mp))
            if not np.isfinite(mp) or mp > 1e8:
                break
    return np.array(max_psi_trace)


def main():
    lat0 = SubstrateLattice(radius_cells=R)
    dt0 = lat0.stable_dt()
    n_steps = 2000
    c2 = 1e-3
    print(f"N={lat0.N}  R={R}  dt0(stable_dt)={dt0:.6f}  c2={c2:.0e}", flush=True)
    for frac in (1.0, 1.0 / 3, 1.0 / 10, 1.0 / 30, 1.0 / 100):
        dt = dt0 * frac
        n_this = int(n_steps / frac) if frac < 1 else n_steps
        n_this = min(n_this, 60000)
        trace = run(dt, c2, n_this)
        first, last = trace[0], trace[-1]
        growth = last[1] / max(first[1], 1e-300)
        print(
            f"dt={dt:.6f} (dt0*{frac:.3f}), ran {int(last[0])} steps: "
            f"max|psi| {first[1]:.3e} -> {last[1]:.3e}  (x{growth:.3g})",
            flush=True,
        )


if __name__ == "__main__":
    main()
