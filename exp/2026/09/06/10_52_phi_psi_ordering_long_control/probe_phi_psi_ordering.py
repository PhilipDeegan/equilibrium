"""
Answers both items queued in `now` at the end of 2026-09-05 night for the
(phi, psi) conjugate-pair prototype (`probe_phi_psi_symplectic.py`):

1. Try the OTHER natural leapfrog staggering (phi updated first from OLD
   psi, then psi updated from NEW phi -- the mirror image of the original
   script's order, which updates psi first from OLD phi then phi from NEW
   psi) and check whether the curl-free control finally holds psi at exact
   floating-point zero.

2. Whether or not (1) fixes it: run the curl-free control MUCH longer
   (20000 steps, vs the original 1500) under BOTH orderings and check
   max|psi| over time for a genuine oscillate-and-return signature (bounded
   -- would mean the small leak seen at 1500 steps is benign sector-coupling
   noise, not a real defect) vs. monotonic secular growth (real ordering
   bug, needs an actual fix before trusting any ring-decay verdict).
"""

import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium.substrate.discrete_curl import triangle_triples
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR, _TOROIDAL_PATTERNS
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

R = 22.0
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def build_ring(lat, axis="z"):
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    ring_js, ring_faces, cur = [c0], [], c0
    for k in _TOROIDAL_PATTERNS[axis]:
        nxt = int(lat.nb[cur, k])
        assert nxt >= 0
        ring_faces.append(k)
        ring_js.append(nxt)
        cur = nxt
    assert ring_js[-1] == c0
    return np.array(ring_js[:-1]), np.array(ring_faces)


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


def energy(u, phi, psi=None, c2=0.0):
    h = float(np.sum(u**2) + 0.5 * np.sum(phi**2))
    if psi is not None:
        h += 0.5 * c2 * sum(np.sum(p**2) for p in psi)
    return h


def step_order_A(lat, psi, dt, c2, valid, safe_nb, triples):
    """Original ordering: psi updated first from OLD phi, then phi updated
    from NEW psi (mirrors the existing u-then-phi leapfrog directly)."""
    u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
    vals = circ_D(lat.phi, triples)
    psi_new = [p - dt * v for p, v in zip(psi, vals)]
    extra = scatter_DT(psi_new, lat.phi.shape, triples)
    phi_new = lat.phi + dphi_base + dt * c2 * extra
    return u_new, phi_new, psi_new


def step_order_B(lat, psi, dt, c2, valid, safe_nb, triples):
    """Mirror ordering: phi updated first from OLD psi, then psi updated
    from NEW phi."""
    u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
    extra = scatter_DT(psi, lat.phi.shape, triples)
    phi_new = lat.phi + dphi_base + dt * c2 * extra
    vals = circ_D(phi_new, triples)
    psi_new = [p - dt * v for p, v in zip(psi, vals)]
    return u_new, phi_new, psi_new


ORDERS = {"A_psi_first": step_order_A, "B_phi_first": step_order_B}


def run_control(order_name, stepper, dt, c2, n_steps, log_every):
    lat = SubstrateLattice(radius_cells=R)
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)

    u0 = np.exp(-0.5 * (np.linalg.norm(lat.pos, axis=1) / 4.0) ** 2)
    lat.u = u0
    lat.phi = np.where(valid, u0[safe_nb] - u0[:, None], 0.0)
    psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]

    h0 = energy(lat.u, lat.phi, psi, c2)
    max_psi_trace = []
    h_trace = []
    for step in range(n_steps):
        u_new, phi_new, psi_new = stepper(lat, psi, dt, c2, valid, safe_nb, triples)
        lat.u, lat.phi, psi = u_new, phi_new, psi_new
        if step % log_every == 0 or step == n_steps - 1:
            mp = max(float(np.max(np.abs(p))) for p in psi)
            max_psi_trace.append((step, mp))
            h_trace.append(energy(lat.u, lat.phi, psi, c2) / h0)

    max_psi_trace = np.array(max_psi_trace)
    h_trace = np.array(h_trace)
    # oscillate-and-return vs monotonic: compare trailing-window min to
    # early post-transient min: if the trailing min is not clearly above
    # the early min (same order of magnitude), it's returning toward zero;
    # if it keeps climbing, it's secular.
    n = len(max_psi_trace)
    early = max_psi_trace[n // 10 : n // 4, 1]
    late = max_psi_trace[3 * n // 4 :, 1]
    verdict = "SECULAR (monotonic growth)" if late.min() > 3 * early.min() else "BOUNDED/OSCILLATING"
    print(
        f"[{order_name}] c2={c2:.0e} n_steps={n_steps}: "
        f"max|psi| early-window min={early.min():.3e} late-window min={late.min():.3e} "
        f"late-window max={late.max():.3e}  H ratio end={h_trace[-1]:.8f}  -> {verdict}",
        flush=True,
    )
    return max_psi_trace, h_trace


def main():
    lat0 = SubstrateLattice(radius_cells=R)
    dt = lat0.stable_dt()
    print(f"N={lat0.N}  R={R}  dt={dt:.6f}", flush=True)

    n_steps = 20000
    log_every = 20
    for order_name, stepper in ORDERS.items():
        for c2 in (1e-4, 1e-3, 1e-2):
            run_control(order_name, stepper, dt, c2, n_steps, log_every)


if __name__ == "__main__":
    main()
