"""
Ambient boundary: the domain sits inside a larger medium instead of an empty, absorbing vacuum.

The sponge relaxes the lattice toward a target instead of zero. The target is
    bath(t) + static_phi
- bath: a copy of the lattice (same geometry, closed, no sponge) started from an ambient state
  (ambient.thermal_state) and evolved with the same wave step as the domain. Its waves enter
  through the sponge; only the domain's departure from the bath is absorbed.
- static_phi: a field held fixed (not evolved, not self-consistent), e.g. the Coulomb exteriors of
  protons outside the domain (proton_microfield). A control, held constant while the domain is
  tested. Its divergence sits on the boundary cells (the external flux through the missing faces);
  that flux is held too: step() adds it back to u there, so the field is static under the linear rule.

With the linear step() the domain and the bath superpose: domain - bath evolves exactly as the
same departure under a zero-target sponge, so the ambient medium cannot act on a structure. Only
a nonlinear rule (e.g. self-sourced lam) couples them.
"""

import copy

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from equilibrium.substrate.ambient import K_REV, thermal_state
from equilibrium.substrate.lattice import _W, SubstrateLattice


def _gradient_operator(nb, W):
    i, k = np.nonzero(nb >= 0)
    j = nb[i, k]
    once = i < j
    fi, fk, fj = i[once], k[once], j[once]
    w = W[fk]
    F = len(fi)
    G = sp.csr_matrix(
        (np.concatenate([w, -w]), (np.concatenate([np.arange(F)] * 2), np.concatenate([fj, fi]))), shape=(F, len(nb))
    )
    return G, fi, fk, fj


def _harmonic_field(nb, W, psi_b):
    boundary = np.any(nb < 0, axis=1)
    G, fi, fk, fj = _gradient_operator(nb, W)
    L = (G.T @ G).tocsr()
    inner = np.where(~boundary)[0]
    bnd = np.where(boundary)[0]
    psi = psi_b.copy()
    psi[inner] = spsolve(L[inner][:, inner].tocsc(), -(L[inner][:, bnd] @ psi_b[bnd]))
    f = -(G @ psi)
    phi = np.zeros((len(nb), 14))
    phi[fi, fk] = f
    phi[fj, K_REV[fk]] = -f
    return phi, psi


def proton_microfield(pos, nb, W, charges):
    """
    Static face field phi (N, 14) of point charges outside the domain, charges = [(position, Q), ...],
    psi = sum Q / |r - r_p| (caller's units; Coulomb with eps: Q -> Q / (4 pi eps)).
    Exactly discrete-harmonic inside: psi is taken from the sum on the boundary cells and solved
    (G^T G psi = 0) in the interior, so div phi = 0 on every interior cell and the field is static
    under step() there with uniform u. Returns (phi, psi).
    """
    pos = np.asarray(pos, dtype=np.float64)
    psi_b = np.zeros(len(pos))
    for rp, Q in charges:
        d = np.linalg.norm(pos - np.asarray(rp, dtype=np.float64), axis=1)
        if d.min() < 1e-9:
            raise ValueError("proton_microfield: a charge sits on a cell")
        psi_b += Q / d
    return _harmonic_field(nb, W, psi_b)


def uniform_field(pos, nb, W, E):
    """
    Static face field phi (N, 14) of a uniform field E (3-vector), psi = -E . x: the far-source limit
    of proton_microfield (sunlight / a distant charge at electron scale). A linear psi is discrete-harmonic
    on the lattice (the +-k faces cancel), so the interior solve returns it exactly. Returns (phi, psi).
    """
    return _harmonic_field(nb, W, -(np.asarray(pos, dtype=np.float64) @ np.asarray(E, dtype=np.float64)))


def _wave_step(lat, dt):
    SubstrateLattice.step_cpp(lat, dt)


class AmbientBC:
    """
    Drives a DampedLattice's sponge toward bath(t) + static_phi (see module docstring).

    lat: DampedLattice (its sponge profile is used; its own u_bg / phi_bg are overwritten).
    T: bath temperature (ambient.thermal_state); T = 0 gives a quiet bath.
    static_phi: (N, 14) held-fixed field, or None.
    wave_step(obj, dt): the rule without sponge, applied to both domain and bath (default: step()).
    add_to_domain: start the domain as bath + static_phi (+ whatever the caller then adds).
    """

    def __init__(self, lat, T, rng=None, static_phi=None, wave_step=_wave_step, add_to_domain=True):
        self.lat = lat
        self.wave_step = wave_step
        self.static_phi = np.zeros_like(lat.phi) if static_phi is None else np.array(static_phi, dtype=np.float64)
        self.bath = copy.copy(lat)
        self.bath.u, self.bath.phi = thermal_state(lat.nb, T, rng)
        self.bath.lam = lat.lam.copy()
        self.bath.ell = lat.ell.copy()
        if add_to_domain:
            lat.u = lat.u + self.bath.u
            lat.phi = lat.phi + self.bath.phi + self.static_phi
        self._boundary = np.where(np.any(lat.nb < 0, axis=1))[0]
        self._set_target()

    def _hold_external_flux(self, dt):
        if not np.any(self.static_phi):
            return
        lat = self.lat
        b = self._boundary
        valid = lat.nb[b] >= 0
        sig = 1.0 / (1.0 + lat.lam)
        sf = 0.5 * (sig[b][:, None] + sig[np.where(valid, lat.nb[b], 0)])
        lat.u[b] += dt * np.sum(np.where(valid, sf * _W[None, :] * self.static_phi[b], 0.0), axis=1)

    def _set_target(self):
        self.lat._u_bg = self.bath.u
        self.lat.phi_bg = self.bath.phi + self.static_phi

    def step(self, dt):
        self.wave_step(self.bath, dt)
        self._hold_external_flux(dt)
        self.wave_step(self.lat, dt)
        self._set_target()
        self.lat.apply_sponge(dt)

    def departure(self):
        """Domain minus the ambient (bath + static): (du, dphi)."""
        return self.lat.u - self.bath.u, self.lat.phi - self.bath.phi - self.static_phi
