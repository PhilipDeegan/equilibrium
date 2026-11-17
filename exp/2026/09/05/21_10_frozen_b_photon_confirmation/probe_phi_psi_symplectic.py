"""
Answers Philip's question directly: does trying (a) [a better integrator on
the SAME self-coupled term] give a reason to do (b) [a genuine second field]?

Methodology fix vs. last night's first prototype: the real SubstrateLattice.
step() is SYMPLECTIC Euler (u updated from OLD phi, then phi updated from the
NEW u) at dt<=lat.stable_dt(). Last night's prototype used plain
(non-staggered) explicit Euler at a hand-picked dt -- redone here with the
correct staggering AND the correct dt, with an explicit baseline-only sanity
check first, so a stability failure can be attributed to the NEW term and
not to an integrator/dt bug.

Part 1: is the previous "new_term" operator (D^T D, D = the 12-triangle
circulation extractor) EVER going to oscillate, under ANY sign or coupling
strength? Worked out on paper: new_term = D^T @ (D @ phi) is manifestly
SYMMETRIC POSITIVE-SEMIDEFINITE (a Gram matrix). dphi/dt = k*(D^T D)*phi is
therefore a pure real-eigenvalue (growth/decay) system in every eigen-basis,
regardless of integrator or sign of k -- staggering/leapfrog stabilises
otherwise-marginal ANTISYMMETRIC (rotation-generator) couplings, it cannot
turn a symmetric PSD self-coupling into an oscillator.

Part 2: give phi's curl content a genuine INDEPENDENT conjugate field, psi,
living on the 12-triangle plaquettes (one scalar per elementary-triangle
instance -- a discrete 2-form). Mirrors the EXISTING (u,phi) antisymmetric
pairing exactly:
    existing:  du/dt  = -Div(phi)          dphi/dt = +Grad(u)
    new:       dpsi/dt = -D(phi)  (circ)    dphi/dt += +c^2 * D^T(psi)
D and D^T are the SAME two operators validated last night.
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
    """Matches SubstrateLattice.step() exactly: du=-dt*div(phi);
    dphi=-dt*W*(u_nb-u_new) = +dt*W*(u_new-u_nb)."""
    div_phi = np.sum(np.where(valid, W[None, :] * lat.phi, 0.0), axis=1)
    u_new = lat.u - dt * div_phi
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * W[None, :] * drive
    return u_new, dphi


def energy(lat, extra=0.0):
    return float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2)) + extra


def sanity_check_baseline(valid, safe_nb):
    print("\n=== SANITY: plain baseline (u,phi) alone at dt=stable_dt(), no new term ===", flush=True)
    lat = SubstrateLattice(radius_cells=R)
    dt = lat.stable_dt()
    ring_js, ring_faces = build_ring(lat)
    lat.phi[ring_js, ring_faces] = 1.0
    h0 = energy(lat)
    for _ in range(2000):
        u_new, dphi = symplectic_step_baseline(lat, dt, valid, safe_nb)
        lat.u, lat.phi = u_new, lat.phi + dphi
    h_end = energy(lat)
    print(f"dt={dt:.6f}  H ratio after 2000 steps = {h_end/h0:.8f} (expect ~1.0)", flush=True)
    return dt


def run_part1(valid, safe_nb, triples, dt):
    print("\n=== PART 1: D^T D self-coupling, correct dt, several k_curl magnitudes ===", flush=True)
    for k_curl in (+0.001, -0.001, +0.0001, -0.0001):
        lat = SubstrateLattice(radius_cells=R)
        ring_js, ring_faces = build_ring(lat)
        lat.phi[ring_js, ring_faces] = 1.0
        h0 = energy(lat)
        circ0 = float(np.sum(lat.phi[ring_js, ring_faces]))
        blew_up = False
        for step in range(2000):
            u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
            vals = circ_D(lat.phi, triples)
            extra = scatter_DT(vals, lat.phi.shape, triples)
            lat.u = u_new
            lat.phi = lat.phi + dphi_base + dt * k_curl * extra
            if not np.all(np.isfinite(lat.phi)) or abs(energy(lat)) > 1e12:
                blew_up = True
                break
        h_end = energy(lat)
        circ_end = float(np.sum(lat.phi[ring_js, ring_faces]))
        tag = "OVERFLOW" if blew_up else ("GROWTH" if h_end > h0 * 1.01 else "DECAY" if h_end < h0 * 0.99 else "~stable")
        print(
            f"k_curl={k_curl:+.5f}: H ratio={h_end/h0:.4g}  circ {circ0:.3f} -> {circ_end:.4g}  [{tag}]",
            flush=True,
        )


def run_part2(valid, safe_nb, triples, dt):
    print("\n=== PART 2: genuine (phi, psi) conjugate pair on the plaquettes ===", flush=True)
    for c2 in (1e-4, 1e-3, 1e-2):
        lat = SubstrateLattice(radius_cells=R)
        ring_js, ring_faces = build_ring(lat)
        lat.phi[ring_js, ring_faces] = 1.0
        psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]
        h0 = energy(lat, 0.5 * c2 * sum(np.sum(p**2) for p in psi))
        circ0 = float(np.sum(lat.phi[ring_js, ring_faces]))

        circ_trace, h_trace = [circ0], [h0]
        blew_up = False
        n_steps = 3000
        for step in range(n_steps):
            u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
            vals = circ_D(lat.phi, triples)
            dpsi = [-dt * v for v in vals]
            psi_new = [p + dp for p, dp in zip(psi, dpsi)]
            extra = scatter_DT(psi_new, lat.phi.shape, triples)
            lat.u = u_new
            lat.phi = lat.phi + dphi_base + dt * c2 * extra
            psi = psi_new
            circ_trace.append(float(np.sum(lat.phi[ring_js, ring_faces])))
            h_now = energy(lat, 0.5 * c2 * sum(np.sum(p**2) for p in psi))
            h_trace.append(h_now)
            if not np.isfinite(h_now) or abs(h_now) > 1e12:
                blew_up = True
                break
        circ_trace = np.array(circ_trace)
        h_trace = np.array(h_trace)
        n_osc = int(np.sum(np.diff(np.sign(np.diff(circ_trace))) != 0)) if len(circ_trace) > 2 else 0
        tag = "OVERFLOW" if blew_up else "ok"
        print(
            f"c2={c2:.0e}: [{tag}] ran {len(circ_trace)-1}/{n_steps} steps  "
            f"H ratio={h_trace[-1]/h_trace[0]:.6f}  "
            f"circ min={circ_trace.min():.4f} max={circ_trace.max():.4f} end={circ_trace[-1]:.4f}  "
            f"sign-changes-in-dcirc/dt={n_osc}",
            flush=True,
        )

        if not blew_up:
            # control: curl-free IC, psi must stay exactly zero forever
            lat2 = SubstrateLattice(radius_cells=R)
            u0 = np.exp(-0.5 * (np.linalg.norm(lat2.pos, axis=1) / 4.0) ** 2)
            valid2 = lat2.nb >= 0
            safe_nb2 = np.where(valid2, lat2.nb, 0)
            lat2.u = u0
            lat2.phi = np.where(valid2, u0[safe_nb2] - u0[:, None], 0.0)
            triples2 = triangle_triples(lat2.nb)
            psi2 = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples2]
            h0_2 = energy(lat2, 0.0)
            for step in range(n_steps):
                u_new2, dphi_base2 = symplectic_step_baseline(lat2, dt, valid2, safe_nb2)
                vals2 = circ_D(lat2.phi, triples2)
                dpsi2 = [-dt * v for v in vals2]
                psi2_new = [p + dp for p, dp in zip(psi2, dpsi2)]
                extra2 = scatter_DT(psi2_new, lat2.phi.shape, triples2)
                lat2.u = u_new2
                lat2.phi = lat2.phi + dphi_base2 + dt * c2 * extra2
                psi2 = psi2_new
            h_end_2 = energy(lat2, 0.5 * c2 * sum(np.sum(p**2) for p in psi2))
            max_psi2 = max(float(np.max(np.abs(p))) for p in psi2)
            print(
                f"    [control, curl-free IC] H ratio={h_end_2/h0_2:.8f}, "
                f"max|psi| after {n_steps} steps={max_psi2:.3e} (expect exactly 0)",
                flush=True,
            )


def main():
    lat0 = SubstrateLattice(radius_cells=R)
    nb = lat0.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    triples = triangle_triples(nb)
    print(f"N={lat0.N}  R={R}  triangle families={len(triples)}", flush=True)

    dt = sanity_check_baseline(valid, safe_nb)
    run_part1(valid, safe_nb, triples, dt)
    run_part2(valid, safe_nb, triples, dt)


if __name__ == "__main__":
    main()
