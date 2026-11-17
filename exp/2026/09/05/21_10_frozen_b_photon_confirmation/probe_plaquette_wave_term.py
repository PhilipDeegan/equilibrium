"""
Step 2 prototype: a candidate DST-native term to un-freeze curl(phi).

Derivation (see `now`, 2026-09-05 entry "plaquette-wave term derivation"):
the CURRENT update rule is dphi[i,k]/dt = -grad(u) exactly, always, on every
face -- so any curl-full remainder of phi (call it phi_curl) has an exactly
zero time derivative, trivially: dphi_curl/dt = dphi/dt - d(-grad(u))/dt = 0
identically, regardless of u's own dynamics. This is the concrete mechanism
behind the already-proven dB/dt=0 identity. Classically, the missing piece
is exactly the Ampere-Maxwell/Faraday coupling: d(vector potential)/dt gets
a contribution from curl(curl(A)) too (equivalently, d^2A/dt^2 = c^2
laplacian(A) = c^2[grad(div A) - curl(curl A)]) -- the TRANSVERSE part of
that wave equation is what lets a real B propagate at all. DST has no
analogue of this: phi is only ever sourced by -grad(u), never by curl(B).

This script prototypes the natural DST-NATIVE discrete analogue of that
missing curl(curl(phi)) term, built from the ALREADY-VALIDATED 12-elementary-
triangle circulation primitive (equilibrium.substrate.discrete_curl), not
from any borrowed continuum formula:

  new_term[i,k] = sum, over every elementary triangle instance that uses the
                  directed edge (i,k) as one of its 3 legs, of that
                  triangle's own circulation.

This is the discrete adjoint ("co-differential") of the circulation
operator -- summing 2-form (plaquette) values back onto the 1-form (edge)
they border -- the standard discrete-exterior-calculus move for going from
"curl" back to a field that can update the thing curl was taken of. It is
automatically zero wherever phi is discrete-exact (any scalar's discrete
derivative), since every triangle's own circulation is already exactly zero
there -- so it can NEVER perturb any already-validated curl-free result
(Coulomb, the scalar energy-flux law, the traveling-wave photon).

NOT wired into equilibrium/substrate/lattice.py's real step() -- this is an
isolated, opt-in prototype only, to check the qualitative behaviour before
committing to a production change.
"""

import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.discrete_curl import TRIANGLES, triangle_triples
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR, _TOROIDAL_PATTERNS
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

R = 22.0


def build_reverse_curl(nb):
    """Precompute, once, the (family, leg, a/b/c-arrays) triples -- reused
    every step since nb (lattice topology) never changes."""
    return triangle_triples(nb)


def new_term(phi, triples):
    """new_term[i,k] = sum of circulations of every elementary triangle
    touching directed edge (i,k) as one of its 3 legs. phi: (N,14)."""
    inc = np.zeros_like(phi)
    for (h1, h2, sqk), a, b, c in triples:
        if a.size == 0:
            continue
        m = _MIRROR[sqk]
        circ = phi[a, h1] + phi[b, h2] + phi[c, m]
        np.add.at(inc, (a, np.full_like(a, h1)), circ)
        np.add.at(inc, (b, np.full_like(b, h2)), circ)
        np.add.at(inc, (c, np.full_like(c, m)), circ)
    return inc


def build_ring(lat, axis="z"):
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    pattern = _TOROIDAL_PATTERNS[axis]
    ring_js = [c0]
    ring_faces = []
    cur = c0
    for k in pattern:
        nxt = int(lat.nb[cur, k])
        assert nxt >= 0, "ring hit boundary -- shrink pattern or grow R"
        ring_faces.append(k)
        ring_js.append(nxt)
        cur = nxt
    assert ring_js[-1] == c0, "ring did not close"
    return np.array(ring_js[:-1]), np.array(ring_faces)


def main():
    lat = SubstrateLattice(radius_cells=R)
    nb = lat.nb
    print(f"N={lat.N}  R={R}", flush=True)

    triples = build_reverse_curl(nb)
    n_instances = sum(a.size for (_, a, b, c) in triples)
    print(f"triangle families={len(triples)}  total instances={n_instances}", flush=True)

    # --- correctness gate: new_term must be EXACTLY zero for any
    # discrete-exact phi (a scalar's discrete derivative) ---
    rng = np.random.default_rng(0)
    u_rand = rng.normal(size=lat.N)
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    phi_discrete_exact = np.where(valid, u_rand[safe_nb] - u_rand[:, None], 0.0)
    zero_check = new_term(phi_discrete_exact, triples)
    print(
        f"[gate] new_term on a discrete-exact (curl-free) phi: "
        f"max|.|={np.max(np.abs(zero_check)):.3e} "
        f"(expect ~0, floating-point noise)",
        flush=True,
    )
    assert np.max(np.abs(zero_check)) < 1e-8, "reverse-curl operator is NOT zero on a curl-free field -- bug, stop here"

    # --- build a genuine curl-full ring (constant phi around a closed loop) ---
    ring_js, ring_faces = build_ring(lat, axis="z")
    lat.phi = np.zeros_like(lat.phi)
    lat.u = np.zeros(lat.N)
    I0 = 1.0
    lat.phi[ring_js, ring_faces] = I0
    circ0 = float(np.sum(lat.phi[ring_js, ring_faces]))
    print(f"ring: {len(ring_js)} cells, constant phi={I0}, naive loop-sum circulation={circ0}", flush=True)

    face_map_cells = [int(j) for j in ring_js] + [int(lat.nb[ring_js[-1], ring_faces[-1]])]
    b_ring = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
    inside_pt = lat.pos[ring_js].mean(axis=0)
    inside_pt[2] = 0.0
    from scipy.spatial import cKDTree

    tree = cKDTree(lat.pos)
    _, i_center = tree.query(inside_pt)
    print(f"B at ring center (t=0): {b_ring[int(i_center)]}", flush=True)

    # --- prototype update: dphi/dt gets an EXTRA term k_curl * new_term(phi),
    # on top of the existing du/dt = -div(phi), dphi/dt = -grad(u). Plain
    # explicit Euler, small dt, small k_curl -- this is a qualitative check,
    # not a production-quality integrator. ---
    dt = 0.02
    K_CURL = 0.05
    N_STEPS = 400

    def step_with_new_term(lat, dt, k_curl, triples):
        u, phi = lat.u, lat.phi
        div_phi = np.zeros(lat.N)
        w = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
        valid = lat.nb >= 0
        div_phi = np.sum(np.where(valid, w[None, :] * phi, 0.0), axis=1)
        du = -dt * div_phi

        safe_nb = np.where(valid, lat.nb, 0)
        grad_u = np.where(valid, u[:, None] - u[safe_nb], 0.0) * w[None, :]
        extra = new_term(phi, triples)
        dphi = dt * grad_u + dt * k_curl * extra

        lat.u = u + du
        lat.phi = phi + dphi

    circ_trace = []
    b_trace = []
    h_trace = []
    for step in range(N_STEPS + 1):
        circ_now = float(np.sum(lat.phi[ring_js, ring_faces]))
        b_now = b_vec_vectorized(lat.pos, lat.nb, lat.phi)[int(i_center)]
        h_now = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
        circ_trace.append(circ_now)
        b_trace.append(b_now.copy())
        h_trace.append(h_now)
        if step < N_STEPS:
            step_with_new_term(lat, dt, K_CURL, triples)

    circ_trace = np.array(circ_trace)
    b_trace = np.array(b_trace)
    h_trace = np.array(h_trace)

    print(
        f"\nring loop-sum circulation: t=0 -> {circ_trace[0]:.6f}, "
        f"t=end -> {circ_trace[-1]:.6f}  "
        f"(min={circ_trace.min():.4f} max={circ_trace.max():.4f})",
        flush=True,
    )
    print(
        f"B at ring center (z-component): t=0 -> {b_trace[0,2]:.6f}, "
        f"t=end -> {b_trace[-1,2]:.6f}  "
        f"(min={b_trace[:,2].min():.4f} max={b_trace[:,2].max():.4f})",
        flush=True,
    )
    print(
        f"H (total energy proxy): t=0 -> {h_trace[0]:.6f}, t=end -> {h_trace[-1]:.6f}  "
        f"ratio={h_trace[-1]/h_trace[0]:.6f}",
        flush=True,
    )

    # --- control: same k_curl term applied to a CURL-FREE IC (companion-phi
    # style scalar-gradient phi) -- must show NO change at all ---
    lat2 = SubstrateLattice(radius_cells=R)
    u0 = np.exp(-0.5 * (np.linalg.norm(lat2.pos, axis=1) / 4.0) ** 2)
    valid2 = lat2.nb >= 0
    safe_nb2 = np.where(valid2, lat2.nb, 0)
    lat2.u = u0
    lat2.phi = np.where(valid2, u0[safe_nb2] - u0[:, None], 0.0)
    h0_2 = float(np.sum(lat2.u**2) + 0.5 * np.sum(lat2.phi**2))
    triples2 = build_reverse_curl(lat2.nb)
    for step in range(N_STEPS):
        step_with_new_term(lat2, dt, K_CURL, triples2)
    h_end_2 = float(np.sum(lat2.u**2) + 0.5 * np.sum(lat2.phi**2))
    extra_check = new_term(lat2.phi, triples2)
    print(
        f"\n[control] curl-free IC under the SAME new term: "
        f"H ratio={h_end_2/h0_2:.6f}, "
        f"new_term magnitude at end = {np.max(np.abs(extra_check)):.3e} "
        f"(expect: behaves like plain step() throughout, term stays ~0)",
        flush=True,
    )

    out_dir = "exp/2026/09/05/21_10_frozen_b_photon_confirmation"
    np.savez(
        f"{out_dir}/plaquette_wave_term_trace.npz",
        circ=circ_trace,
        b=b_trace,
        h=h_trace,
        dt=dt,
        k_curl=K_CURL,
    )


if __name__ == "__main__":
    main()
