"""
step_cross v2 — ported to the uniform (non-adaptive) SubstrateLattice.

`run/substrate/adaptive/cross_v2.py` implements the gated-crossing prototype
(spec: doc/step_cross_v2.md) bound to AdaptiveLattice clusters (leaf_phi /
leaf_nb_j indirection). Everything it does is pure per-face-topology math on
the 14-neighbour Kelvin cell, which is identical between AdaptiveLattice and
SubstrateLattice -- only the array-access layer differs (SubstrateLattice
already exposes flat (N,14)/(N,) numpy arrays directly, no leaf transpose
needed). This module re-implements the same gates against SubstrateLattice's
.phi/.nb/.lam/.fixed_phi arrays, to confirm the ported operator reproduces:

  (a) the existing compiled/Python SubstrateLattice.step_cross() exactly in
      gate="v1" mode, and
  (b) the same conservation / null-gate / barrier-threshold properties the
      original cluster-bound self-test established.

See doc/dst_next.md section 8 for the barrier-gate design rationale.

Self-test:
    PYTHONPATH=$PWD python3 -u run/substrate/cross_v2_uniform.py
"""

import numpy as np

from equilibrium.math.geo.kelvin import KelvinLattice
from equilibrium.substrate.lattice import (
    _FACE_ADJ,
    _K_REV,
    W_HEX,
    W_SQ,
    SubstrateLattice,
)

_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
_geo = KelvinLattice(lattice_spacing=1.0)
_OFFSETS = _geo.all_offsets  # (14, 3)
_FACE_NORM = _OFFSETS / np.linalg.norm(_OFFSETS, axis=1, keepdims=True)
_ADJ_LIST = [[int(j) for j in _FACE_ADJ[k] if j >= 0] for k in range(14)]

# Vacuum capacitance barrier -- see doc/dst_next.md section 8 and
# run/substrate/adaptive/cross_v2.py's own M0_VOID for the full derivation
# (void_limit = m_p/m_Planck, scaling_factor = real knot radius / real Planck
# length). Duplicated here (not imported) to keep this module independent of
# the adaptive-side one; keep the two values in sync if the calibration
# changes.
_M_P_OVER_M_PLANCK = 7.7e-20
M0_VOID = _M_P_OVER_M_PLANCK


class CrossV2:
    """step_cross v2 prototype bound to a plain SubstrateLattice."""

    def __init__(self, lat, gate="shear", ref=1e-4, beta=0.0, u_ref=1.0, m0=M0_VOID):
        if gate not in ("v1", "shear", "product", "opposed", "curl", "barrier"):
            raise ValueError(f"unknown gate {gate!r}")
        self.lat = lat
        self.m0 = float(m0)
        self.gate = gate
        self.ref = float(ref)
        self.beta = float(beta)
        self.u_ref = float(u_ref)
        self.nb = lat.nb  # (N, 14) -- triggers cache rebuild if dirty
        self.valid = self.nb >= 0
        self.safe_nb = np.where(self.valid, self.nb, 0)
        self.W = _W
        self.norm = _FACE_NORM

    # -- gate raw variables (also used by the audit) --------------------

    def transit_vectors(self, phiv):
        """v_i = Σ_k W_k phi[i,k] n_hat_k for masked phi (N, 14) → (N, 3)."""
        return np.einsum("nk,k,kd->nd", phiv, self.W, self.norm)

    def raw_shear(self, phiv):
        """m[i,k] = |v_i||v_nb| - v_i.v_nb, zero on invalid faces. (N, 14)"""
        v = self.transit_vectors(phiv)
        vmag = np.linalg.norm(v, axis=1)
        v_nb = v[self.safe_nb]  # (N, 14, 3)
        m = vmag[:, None] * vmag[self.safe_nb] - np.einsum("nd,nkd->nk", v, v_nb)
        return np.where(self.valid, np.maximum(m, 0.0), 0.0)

    def raw_opposed(self, phiv):
        """o[i,k] = max(0, -v_i.v_nb) — >90° direction disagreement only. (N, 14)"""
        v = self.transit_vectors(phiv)
        o = -np.einsum("nd,nkd->nk", v, v[self.safe_nb])
        return np.where(self.valid, np.maximum(o, 0.0), 0.0)

    def raw_product(self, phiv):
        """q[i,k] = (-phi[i,k])_+ * Σ_{j∉{k,mk}} W_j |phi[i,j]|. (N, 14)"""
        absw = self.W[None, :] * np.abs(phiv)  # (N, 14)
        trans = absw.sum(axis=1)[:, None] - absw - absw[:, _K_REV]
        return np.maximum(-phiv, 0.0) * trans

    def _plaquette(self, phiv, ka, kb):
        """Elementary Wilson-loop sum on the (ka, kb) square-face plaquette,
        reading phi[i,k] as a discrete connection A_k(i) on the outward edge
        (i, nb_k(i)). Zero wherever either hop or its return face is
        invalid (boundary). (N,)"""
        nb_a = self.safe_nb[:, ka]
        nb_b = self.safe_nb[:, kb]
        term = phiv[:, ka] + phiv[nb_a, kb] - phiv[nb_b, ka] - phiv[:, kb]
        ok = (
            self.valid[:, ka]
            & self.valid[:, kb]
            & self.valid[nb_a, kb]
            & self.valid[nb_b, ka]
        )
        return np.where(ok, term, 0.0)

    def curl_vectors(self, phiv):
        """Discrete curl pseudovector B_i from the three Cartesian
        (square-face) plaquettes. Sq faces: 8=-x 9=+x 10=-y 11=+y 12=-z
        13=+z. (N, 3), order (Bx,By,Bz)."""
        Bx = self._plaquette(phiv, 11, 13)  # yz
        By = self._plaquette(phiv, 13, 9)  # zx
        Bz = self._plaquette(phiv, 9, 11)  # xy
        return np.stack([Bx, By, Bz], axis=1)

    def raw_curl(self, phiv):
        """m_B[i,k] = |B_i||B_nb_k| - B_i.B_nb_k. (N, 14)"""
        B = self.curl_vectors(phiv)
        Bmag = np.linalg.norm(B, axis=1)
        B_nb = B[self.safe_nb]
        m = Bmag[:, None] * Bmag[self.safe_nb] - np.einsum("nd,nkd->nk", B, B_nb)
        return np.where(self.valid, np.maximum(m, 0.0), 0.0)

    def raw_barrier_drive(self, phiv):
        """drive[i] = Σ_k W_k·max(-phi[i,k],0), pooled per cell (not per
        face). Broadcast to (N,14) so it has the same shape as the other
        raw fields."""
        drive = (self.W[None, :] * np.maximum(-phiv, 0.0)).sum(axis=1)  # (N,)
        return np.broadcast_to(drive[:, None], phiv.shape)

    def gate_field(self, phi=None):
        """(G_link, raw) for the current state — no step taken."""
        if phi is None:
            phi = self.lat.phi
        phiv = np.where(self.valid, phi, 0.0)
        if self.gate == "v1":
            return np.ones_like(phiv), np.ones_like(phiv)
        if self.gate == "barrier":
            # Stateless, instantaneous, all-or-nothing per cell (doc/dst.md:
            # "the crossing either happens or it doesn't: a gate condition on
            # the instantaneous phi state"). m0 is a fixed vacuum-level
            # constant -- density/lam affects RATE elsewhere, never this
            # barrier's magnitude.
            raw = self.raw_barrier_drive(phiv)
            fire = (raw > self.m0).astype(np.float64)
            return fire, raw
        raw = {
            "shear": self.raw_shear,
            "product": self.raw_product,
            "opposed": self.raw_opposed,
            "curl": self.raw_curl,
        }[self.gate](phiv)
        denom = raw + self.ref
        return np.where(denom > 0.0, raw / np.where(denom > 0.0, denom, 1.0), 0.0), raw

    # -- the step --------------------------------------------------------

    def step(self, dt, scale):
        lat = self.lat
        phi = lat.phi
        phiv = np.where(self.valid, phi, 0.0)
        lam = lat.lam
        sig2 = (1.0 / (1.0 + lam)) ** 2

        G_link, _ = self.gate_field(phi)
        inward = np.maximum(-phiv, 0.0)
        delta = (dt * scale) * sig2[:, None] * G_link * inward  # (N, 14) ≥ 0

        # Matches SubstrateLattice.step_cross's own boundary semantics
        # exactly: n_adj is the FIXED per-face-type adjacency count (6 hex /
        # 4 sq), not masked by neighbour validity -- unlike AdaptiveLattice's
        # step_cross (AMR-irregular boundaries), the uniform lattice's C++
        # and Python step_cross both use a static count here. Only the
        # mirror write (below) is masked by self.valid, matching production.
        L = np.zeros_like(phi)
        for k in range(14):
            d = delta[:, k]
            adj = _ADJ_LIST[k]
            n_adj = len(adj)
            L[:, k] += n_adj * d  # drain source face k
            for j in adj:
                L[:, j] += d  # isotropic redistribution

        if self.beta > 0.0:
            u = lat.u
            du = np.where(self.valid, u[self.safe_nb] - u[:, None], 0.0)
            D = np.einsum("nk,k,kd->nd", du, self.W, self.norm)  # (N, 3)
            Dmag = np.linalg.norm(D, axis=1)
            Dhat = D / np.where(Dmag > 1e-300, Dmag, 1.0)[:, None]
            G_drv = self.beta * np.abs(du) / self.u_ref
            d_drv = (dt * scale) * sig2[:, None] * G_drv * inward
            for k in range(14):
                d = d_drv[:, k]
                adj = _ADJ_LIST[k]
                live = self.valid[:, adj].astype(np.float64)  # (N, n_adj)
                n_live = live.sum(axis=1)
                w = (
                    np.stack(
                        [np.maximum(0.0, Dhat @ self.norm[j]) for j in adj], axis=1
                    )
                    * live
                )  # (N, n_adj)
                wsum = w.sum(axis=1)
                flat = wsum <= 1e-300
                w = np.where(
                    flat[:, None],
                    live / np.maximum(n_live, 1.0)[:, None],
                    w / np.where(flat, 1.0, wsum)[:, None],
                )
                L[:, k] += n_live * d
                for m_i, j in enumerate(adj):
                    L[:, j] += n_live * d * w[:, m_i]

        # mirrored counterpart of every local write, gathered via the face
        # bijection: dphi[i,k] = L[i,k] - L[nb_k(i), mirror_k]
        mirror = np.where(self.valid, L[self.safe_nb, _K_REV[None, :]], 0.0)
        new_phi = phi + L - mirror
        lat.phi = np.where(lat.fixed_phi, phi, new_phi)


# ---------------------------------------------------------------------------
# Self-test: v1-mode equivalence against SubstrateLattice.step_cross(),
# plus interior conservation for all gates -- same battery as
# run/substrate/adaptive/cross_v2.py's _selftest(), ported to SubstrateLattice.
# ---------------------------------------------------------------------------


def _make_lat(R=6.0, seed=7, lam0=0.5):
    lat = SubstrateLattice(radius_cells=R)
    valid = lat.nb >= 0
    safe = np.where(valid, lat.nb, 0)
    rng = np.random.default_rng(seed)
    N = lat.N
    A = rng.normal(size=(N, 14))
    phi = np.where(valid, A - A[safe, _K_REV[None, :]], 0.0)  # antisymmetric
    lat.phi = phi
    lat.lam = np.full(N, lam0)
    return lat, valid


def _selftest():
    dt, scale = 5e-3, 0.01

    # 1) v1-mode equivalence vs SubstrateLattice.step_cross()
    lat_a, _ = _make_lat()
    lat_b, valid = _make_lat()
    lat_a.step_cross(dt, scale)
    CrossV2(lat_b, gate="v1").step(dt, scale)
    err = np.abs(lat_a.phi - lat_b.phi)[valid].max()
    print(f"v1 equivalence: max|phi_stepcross - phi_crossv2| = {err:.3e}")
    assert err < 1e-12, "v1-mode does not match SubstrateLattice.step_cross"

    # 2) interior conservation + antisymmetry for each gate
    for gate in ("v1", "shear", "product", "curl", "barrier"):
        lat, valid = _make_lat()
        op = CrossV2(lat, gate=gate, ref=1e-2)
        phi0 = lat.phi.copy()
        op.step(dt, scale)
        phi1 = lat.phi
        dphi = np.where(valid, phi1 - phi0, 0.0)
        tot = dphi[valid].sum()
        anti = phi1 + np.where(valid, phi1[op.safe_nb, _K_REV[None, :]], -phi1)
        print(
            f"gate={gate:8s} global sum(dphi)={tot:+.3e}"
            f"  max|antisymmetry viol|={np.abs(anti).max():.3e}"
            f"  |dphi|_max={np.abs(dphi).max():.3e}"
        )
        assert abs(tot) < 1e-9

    # 3) shear gate null on a coherent parallel-transit pattern
    lat, valid = _make_lat()
    op = CrossV2(lat, gate="shear", ref=1e-2)
    d = np.array([0.3, -0.5, 0.8])
    pattern = _FACE_NORM @ d  # (14,)
    phi = np.where(valid, np.broadcast_to(pattern, valid.shape), 0.0)
    m = op.raw_shear(np.where(valid, phi, 0.0))
    interior = valid.all(axis=1)
    both_int = interior[:, None] & interior[op.safe_nb] & valid
    m_int = m[both_int]
    print(
        f"shear null on uniform coherent transit (interior pairs):"
        f" max m = {m_int.max():.3e}  (rim max {m.max():.3e})"
    )
    assert m_int.max() < 1e-12

    # 4) curl gate: B_i is EXACTLY zero on the same uniform coherent pattern
    B = op.curl_vectors(np.where(valid, phi, 0.0))
    print(f"curl null on uniform coherent transit: max|B| = {np.abs(B).max():.3e}")
    assert np.abs(B).max() < 1e-12

    # 5) barrier gate: sub-barrier drive never crosses, however many ticks
    lat, valid = _make_lat(lam0=0.0)
    op = CrossV2(lat, gate="barrier", m0=1e-3)
    rng = np.random.default_rng(11)
    small = 1e-6 * rng.normal(size=(lat.N, 14))  # drive << m0
    small = np.where(valid, small - small[op.safe_nb, _K_REV[None, :]], 0.0)
    lat.phi = small
    phi_before = lat.phi.copy()
    for _ in range(500):
        op.step(dt, scale)
    phi_after = lat.phi
    drift = np.abs(phi_after - phi_before)[valid].max()
    print(
        f"barrier sub-threshold: 500 ticks, max drive="
        f"{op.raw_barrier_drive(np.where(valid, small, 0.0))[valid].max():.3e}"
        f" (m0={op.m0:.1e})  max|phi drift|={drift:.3e}"
    )
    assert drift < 1e-15, "sub-barrier drive crossed when it should never fire"

    # 6) barrier gate: over-barrier drive fires and matches the v1 kernel
    #    exactly on the cells that fire
    lat_bar, valid = _make_lat(lam0=0.0)
    lat_v1, _ = _make_lat(lam0=0.0)
    nb_bar = lat_bar.nb
    safe_bar = np.where(valid, nb_bar, 0)
    rng = np.random.default_rng(12)
    big = 1.0 * rng.normal(size=(lat_bar.N, 14))  # drive >> m0
    big = np.where(valid, big - big[safe_bar, _K_REV[None, :]], 0.0)
    lat_bar.phi = big.copy()
    lat_v1.phi = big.copy()
    op_bar = CrossV2(lat_bar, gate="barrier", m0=1e-3)
    op_v1 = CrossV2(lat_v1, gate="v1")
    drive = op_bar.raw_barrier_drive(np.where(valid, big, 0.0))
    frac_fire = float((drive[:, 0] > op_bar.m0).mean())
    op_bar.step(dt, scale)
    op_v1.step(dt, scale)
    fired = drive[:, 0] > op_bar.m0
    match = np.abs(lat_bar.phi[fired] - lat_v1.phi[fired]).max() if fired.any() else 0.0
    print(
        f"barrier over-threshold: frac firing={frac_fire:.3f}"
        f"  max|phi_barrier - phi_v1| on firing cells={match:.3e}"
    )
    assert match < 1e-12

    print("cross_v2_uniform selftest OK")


if __name__ == "__main__":
    _selftest()
