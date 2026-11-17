"""
Re-check Part 1/Part 2 (the D^T D self-coupling and the (phi,psi) conjugate
pair) on a properly resolved, smooth curl-full object instead of the sharp
8-tile minimal ring -- Philip's catch: that ring is explicitly documented
(find_toroidal_ring's own docstring, project_bcc_dispersion_anisotropy) as
"inside the BCC dispersion anisotropy band", too small/sharp-cornered to
trust for real physics numbers.

Object: the toroidal half of the already-validated curl-full Taylor-state
IC (equilibrium.substrate.curl_full_dipole_ic.toroidal_envelope/_phihat,
B_phi(r,theta)=j1(K*r)*sin(theta)) -- reused directly, not reinvented --
built via the SAME analytic line-integral method already validated for the
classical current loop (test_curl_loop.py's _build_phi), on a PLAIN
(undamped) SubstrateLattice, R=50, scale=r_sph=0.9*R=45.

Philip's reasoning for the padding choice: this component isn't expected to
decay/radiate at all (matches the discrete-exact ring's own exact-fixed-
point behaviour, and the circulation-quantization energy-barrier result) --
so it doesn't need generous margin from the boundary the way an open/
radiating (poloidal) structure would; if it DOES decay here, under the new
term, on a smooth well-resolved object with a real boundary right behind it,
that would indicate a genuine issue in the (phi,psi) construction, not a
resolution or padding artifact.
"""

import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium.substrate.curl_full_dipole_ic import toroidal_envelope, _phihat
from equilibrium.substrate.discrete_curl import circulation, triangle_triples
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

R = 50.0
SPH_FRAC = 0.9
SCALE = R * SPH_FRAC
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def build_phi_tor(lat, scale, r_sph):
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    diff = pos[safe_nb] - pos[:, None, :]
    N = pos.shape[0]
    A_mid = toroidal_envelope(mid.reshape(-1, 3), scale).reshape(N, 14)
    tang_mid = _phihat(mid.reshape(-1, 3)).reshape(N, 14, 3)
    proj = np.einsum("ikd,ikd->ik", tang_mid, diff)
    phi = A_mid * proj
    r_cell = np.linalg.norm(pos, axis=1)
    r_mid = np.linalg.norm(mid, axis=2)
    phi = np.where(valid, phi, 0.0)
    phi[r_cell > r_sph, :] = 0.0
    phi[r_mid > r_sph] = 0.0
    return phi


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


def symplectic_step_baseline(u, phi, dt, valid, safe_nb):
    div_phi = np.sum(np.where(valid, W[None, :] * phi, 0.0), axis=1)
    u_new = u - dt * div_phi
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * W[None, :] * drive
    return u_new, dphi


def energy(u, phi, extra=0.0):
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2)) + extra


def circ_measure(phi, nb, idx):
    """RMS circulation over interior triangles -- a raw SUM cancels to ~0 by
    azimuthal symmetry on this IC despite real nonzero per-triangle content
    (same trap as the old bc_flux-isn't-an-energy-diagnostic lesson); RMS
    can't cancel that way."""
    c = circulation(phi.T, nb, idx)
    return float(np.sqrt(np.mean(c**2))) if c.size else 0.0


def main():
    lat = SubstrateLattice(radius_cells=R)
    print(f"N={lat.N}  R={R}  scale/r_sph={SCALE}", flush=True)
    nb = lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    triples = triangle_triples(nb)

    phi0 = build_phi_tor(lat, SCALE, SCALE)
    n_nonzero = int(np.sum(np.any(phi0 != 0.0, axis=1)))
    print(f"nonzero-phi cells: {n_nonzero} / {lat.N} ({100*n_nonzero/lat.N:.1f}%)", flush=True)

    # sanity: real, nonzero circulation (curl-full), NOT float noise
    idx_interior = np.where(np.linalg.norm(lat.pos, axis=1) < SCALE - 3.0)[0]
    circ0 = circulation(phi0.T, nb, idx_interior)
    phi_rms = float(np.sqrt(np.mean(phi0**2)))
    circ_rms = float(np.sqrt(np.mean(circ0**2))) if circ0.size else 0.0
    print(
        f"[sanity] circulation rms / phi rms = {circ_rms/phi_rms:.4f} "
        f"(expect a real O(0.1-1) ratio, not float noise -- confirms genuinely curl-full)",
        flush=True,
    )

    dt = lat.stable_dt()
    u0 = np.zeros(lat.N)

    print("\n=== baseline sanity (this smooth IC, no new term) ===", flush=True)
    u, phi = u0.copy(), phi0.copy()
    total_circ0 = circ_measure(phi, nb, idx_interior)
    h0 = energy(u, phi)
    for _ in range(1000):
        u_new, dphi = symplectic_step_baseline(u, phi, dt, valid, safe_nb)
        u, phi = u_new, phi + dphi
    h_end = energy(u, phi)
    total_circ_end = circ_measure(phi, nb, idx_interior)
    print(
        f"dt={dt:.6f}  H ratio after 1000 steps = {h_end/h0:.6f}  "
        f"total interior circulation: {total_circ0:.4f} -> {total_circ_end:.4f}",
        flush=True,
    )

    print("\n=== PART 1 (single check): D^T D self-coupling ===", flush=True)
    for k_curl in (+0.0005, -0.0005):
        u, phi = u0.copy(), phi0.copy()
        h0b = energy(u, phi)
        cb0 = circ_measure(phi, nb, idx_interior)
        for step in range(1000):
            u_new, dphi_base = symplectic_step_baseline(u, phi, dt, valid, safe_nb)
            vals = circ_D(phi, triples)
            extra = scatter_DT(vals, phi.shape, triples)
            u = u_new
            phi = phi + dphi_base + dt * k_curl * extra
        h_end_b = energy(u, phi)
        cb_end = circ_measure(phi, nb, idx_interior)
        print(
            f"k_curl={k_curl:+.5f}: H ratio={h_end_b/h0b:.4g} (baseline alone={h_end/h0:.4g})  "
            f"circ {cb0:.3f} -> {cb_end:.3f}",
            flush=True,
        )

    print("\n=== PART 2 (single check): (phi, psi) conjugate pair ===", flush=True)
    c2 = 1e-3
    u, phi = u0.copy(), phi0.copy()
    psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]
    h0c = energy(u, phi, 0.0)
    circ0c = circ_measure(phi, nb, idx_interior)
    circ_trace, h_trace = [circ0c], [h0c]
    N_STEPS = 1500
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(u, phi, dt, valid, safe_nb)
        vals = circ_D(phi, triples)
        dpsi = [-dt * v for v in vals]
        psi_new = [p + dp for p, dp in zip(psi, dpsi)]
        extra = scatter_DT(psi_new, phi.shape, triples)
        u = u_new
        phi = phi + dphi_base + dt * c2 * extra
        psi = psi_new
        circ_trace.append(circ_measure(phi, nb, idx_interior))
        h_trace.append(energy(u, phi, 0.5 * c2 * sum(np.sum(p**2) for p in psi)))
    circ_trace = np.array(circ_trace)
    h_trace = np.array(h_trace)
    n_osc = int(np.sum(np.diff(np.sign(np.diff(circ_trace))) != 0))
    print(
        f"c2={c2:.0e}: H ratio={h_trace[-1]/h_trace[0]:.6f}  "
        f"circ {circ0c:.4f} -> min={circ_trace.min():.4f} max={circ_trace.max():.4f} end={circ_trace[-1]:.4f}  "
        f"sign-changes-in-dcirc/dt={n_osc}",
        flush=True,
    )

    print("\n=== control: same (phi,psi) update on a curl-free (u-gradient) IC, R=50 ===", flush=True)
    u2 = np.exp(-0.5 * (np.linalg.norm(lat.pos, axis=1) / 6.0) ** 2)
    phi2 = np.where(valid, u2[safe_nb] - u2[:, None], 0.0)
    psi2 = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]
    h0d = energy(u2, phi2, 0.0)
    for step in range(N_STEPS):
        u2_new, dphi2 = symplectic_step_baseline(u2, phi2, dt, valid, safe_nb)
        vals2 = circ_D(phi2, triples)
        dpsi2 = [-dt * v for v in vals2]
        psi2_new = [p + dp for p, dp in zip(psi2, dpsi2)]
        extra2 = scatter_DT(psi2_new, phi2.shape, triples)
        u2 = u2_new
        phi2 = phi2 + dphi2 + dt * c2 * extra2
        psi2 = psi2_new
    h_end_d = energy(u2, phi2, 0.5 * c2 * sum(np.sum(p**2) for p in psi2))
    max_psi2 = max(float(np.max(np.abs(p))) for p in psi2)
    print(
        f"H ratio={h_end_d/h0d:.8f}  max|psi| after {N_STEPS} steps={max_psi2:.3e} (expect exactly 0)",
        flush=True,
    )


if __name__ == "__main__":
    main()
