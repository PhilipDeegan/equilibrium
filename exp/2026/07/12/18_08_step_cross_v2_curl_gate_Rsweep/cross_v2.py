"""
step_cross v2 — Python prototype on the adaptive cluster leaf arrays.

Spec: doc/step_cross_v2.md.  Gate × kernel split of the interior crossing:

    delta[i,k] = dt * scale * sigma(lam_i)^2 * (G_link + G_drv) * (-phi[i,k])
                 (inward faces only, phi[i,k] < 0)

Gate modes (D1 comparison — decision "prototype both, compare"):
  "v1"      G_link = 1                       — reference; matches the C++
            substrate_step_cross / AdaptiveLattice::step_cross semantics.
  "shear"   G_link = g(m_k),  m_k = |v_i||v_nb_k| - v_i.v_nb_k  — winding
            direction disagreement across the face (discrete current-sheet
            signature); v_i = Σ_k W_k phi[i,k] n_hat_k is the cell transit
            vector.  Zero for parallel transit and amplitude profiles.
  "product" G_link = g(q_k),  q_k = (-phi[i,k]) * Σ_{j not in {k,mk}} W_j |phi[i,j]|
            — the literal single-cell "transverse co-located flux" reading.
            Spec §4 predicts this cannot separate linkage from oblique
            transit; kept as the empirical check of that claim.
  "opposed" G_link = g(o_k),  o_k = max(0, -v_i.v_nb_k)  — >90 deg-only variant
            of shear (fires past direction disagreement, not on any
            misalignment). Exactly zero on any static loop family whose
            per-hop rotation stays below 90 deg, so it is immune by
            construction to the shear gate's predicted false-positive on the
            electron's own ring curvature (small per-hop rotation, shear
            fires progressively from 0).
  "curl"    G_link = g(m_B(k)), the shear formula applied to the discrete
            curl B_i instead of the transit vector v_i — phi read as a
            connection (A_k(i) on the outward edge (i, nb_k(i))), B_i built
            from the three Cartesian (square-face) plaquettes.  v1-3's gates
            all compare directions built from ONE cell's own 14 phi values
            (v_i and v_nb are each per-cell moments); B_i genuinely mixes
            three cells' phi per component and vanishes EXACTLY (not just
            asymptotically small, cf. shear's ~1/R^2) for any spatially
            uniform coherent-transit pattern (doc/step_cross_v2.md §4) —
            the holonomy/"field lines are relational" candidate.

g(x) = x / (x + ref) saturating, ref anchored to the audited raw-value
distribution (see probe_cross_v2_gates.py).

Drive channel (D2a, decided): pressure asymmetry couples via
G_drv = beta * |du_k| / u_ref with du_k = u[nb_k] - u[i], and the drive
share redistributes anisotropically toward the cell's pressure-asymmetry
direction D_i = Σ_k W_k du_k n_hat_k (weights w_j ∝ max(0, n_hat_j . D_hat)).
The linkage share keeps the v1 kernel (drain + isotropic redistribution to
edge-adjacent faces, mirrored) — reconnection/unwinding.

Conservation: every local write has its mirrored counterpart gathered via
the face bijection (nb, mirror), so Σ dphi = 0 over interior faces exactly,
matching the v1 drain+mirror discipline.

Self-test (v1 equivalence vs compiled step_cross + conservation):
  PYTHONPATH=$PWD python3 -u tst/substrate/adaptive/cross_v2.py
"""

import numpy as np

from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_NORM,
    _HEX_DIR,
    _MIRROR,
    _SQ_DIR,
    _W_FACE,
    leaf_nb_j,
)


def _build_face_adj():
    """Edge-adjacency of Kelvin-cell faces, derived from the sign rules.

    hex-hex: adjacent iff exactly two sign components differ.
    hex-sq:  sq face (axis a, sign s) adjacent to the 4 hex faces with that
             sign component.  Straight-through (mirror) is never adjacent.
    """
    adj = [[] for _ in range(14)]
    for k in range(8):
        for j in range(8):
            if int(np.sum(_HEX_DIR[k] != _HEX_DIR[j])) == 2:
                adj[k].append(j)
        for j in range(6):
            axis = int(np.nonzero(_SQ_DIR[j])[0][0])
            if _HEX_DIR[k][axis] == _SQ_DIR[j][axis]:
                adj[k].append(8 + j)
                adj[8 + j].append(k)
    for k in range(14):
        adj[k] = sorted(adj[k])
    # topology invariants
    assert all(len(adj[k]) == 6 for k in range(8))
    assert all(len(adj[k]) == 4 for k in range(8, 14))
    for k in range(14):
        assert int(_MIRROR[k]) not in adj[k]
        assert all(k in adj[j] for j in adj[k])
    return adj


_FACE_ADJ = _build_face_adj()


class CrossV2:
    """step_cross v2 prototype bound to one single-level cluster."""

    def __init__(
        self,
        cluster,
        gate="shear",
        ref=1e-4,
        beta=0.0,
        u_ref=1.0,
    ):
        if gate not in ("v1", "shear", "product", "opposed", "curl"):
            raise ValueError(f"unknown gate {gate!r}")
        self.cl = cluster
        self.gate = gate
        self.ref = float(ref)
        self.beta = float(beta)
        self.u_ref = float(u_ref)
        self.nb = leaf_nb_j(cluster)  # (N, 14)
        self.valid = self.nb >= 0
        self.safe_nb = np.where(self.valid, self.nb, 0)
        self.W = _W_FACE.astype(np.float64)  # (14,)
        self.norm = _FACE_NORM  # (14, 3)

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
        """o[i,k] = max(0, -v_i.v_nb) — fires only on >90° direction
        disagreement (opposed transit: reconnection/collision), exactly zero
        on any static loop family whose rotation per hop stays below 90°.
        (N, 14)"""
        v = self.transit_vectors(phiv)
        o = -np.einsum("nd,nkd->nk", v, v[self.safe_nb])
        return np.where(self.valid, np.maximum(o, 0.0), 0.0)

    def raw_product(self, phiv):
        """q[i,k] = (-phi[i,k])_+ * Σ_{j∉{k,mk}} W_j |phi[i,j]|. (N, 14)"""
        absw = self.W[None, :] * np.abs(phiv)  # (N, 14)
        trans = absw.sum(axis=1)[:, None] - absw - absw[:, _MIRROR]
        return np.maximum(-phiv, 0.0) * trans

    def _plaquette(self, phiv, ka, kb):
        """Elementary Wilson-loop sum on the (ka, kb) square-face plaquette,
        reading phi[i,k] as a discrete connection A_k(i) on the outward edge
        (i, nb_k(i)):

            F_ab[i] = A_a(i) + A_b(i+a) - A_a(i+b) - A_b(i)
                    = phi[i,ka] + phi[nb_a(i),kb] - phi[nb_b(i),ka] - phi[i,kb]

        Zero wherever either hop or its return face is invalid
        (boundary/AMR-irregular). (N,)"""
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
        (square-face) plaquettes — phi read as a connection, not a flux.
        Sq faces: 8=-x 9=+x 10=-y 11=+y 12=-z 13=+z. (N, 3), order (Bx,By,Bz)."""
        Bx = self._plaquette(phiv, 11, 13)  # yz
        By = self._plaquette(phiv, 13, 9)  # zx
        Bz = self._plaquette(phiv, 9, 11)  # xy
        return np.stack([Bx, By, Bz], axis=1)

    def raw_curl(self, phiv):
        """m_B[i,k] = |B_i||B_nb_k| - B_i.B_nb_k — the shear misalignment
        test applied to the discrete curl B_i instead of the transit vector
        v_i. (N, 14)"""
        B = self.curl_vectors(phiv)
        Bmag = np.linalg.norm(B, axis=1)
        B_nb = B[self.safe_nb]
        m = Bmag[:, None] * Bmag[self.safe_nb] - np.einsum("nd,nkd->nk", B, B_nb)
        return np.where(self.valid, np.maximum(m, 0.0), 0.0)

    def gate_field(self, phi=None):
        """(G_link, raw) for the current state — no step taken."""
        if phi is None:
            phi = np.array(self.cl.leaf_phi).T
        phiv = np.where(self.valid, phi, 0.0)
        if self.gate == "v1":
            return np.ones_like(phiv), np.ones_like(phiv)
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
        cl = self.cl
        phi = np.array(cl.leaf_phi).T  # (N, 14)
        phiv = np.where(self.valid, phi, 0.0)
        lam = np.array(cl.leaf_lam)
        sig2 = (1.0 / (1.0 + lam)) ** 2

        G_link, _ = self.gate_field(phi)
        inward = np.maximum(-phiv, 0.0)
        delta = (dt * scale) * sig2[:, None] * G_link * inward  # (N, 14) ≥ 0

        # Adaptive-C++ boundary semantics (_step_cross_local): faces without a
        # neighbour are skipped entirely, and n_adj counts live faces only.
        L = np.zeros_like(phi)
        for k in range(14):
            d = delta[:, k]
            adj = _FACE_ADJ[k]
            live = self.valid[:, adj]  # (N, n_adj)
            L[:, k] += live.sum(axis=1) * d  # drain source face k
            for m_i, j in enumerate(adj):
                L[:, j] += d * live[:, m_i]  # isotropic redistribution

        if self.beta > 0.0:
            u = np.array(cl.leaf_u)
            du = np.where(self.valid, u[self.safe_nb] - u[:, None], 0.0)
            D = np.einsum("nk,k,kd->nd", du, self.W, self.norm)  # (N, 3)
            Dmag = np.linalg.norm(D, axis=1)
            Dhat = D / np.where(Dmag > 1e-300, Dmag, 1.0)[:, None]
            G_drv = self.beta * np.abs(du) / self.u_ref
            d_drv = (dt * scale) * sig2[:, None] * G_drv * inward
            for k in range(14):
                d = d_drv[:, k]
                adj = _FACE_ADJ[k]
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
        mirror = np.where(self.valid, L[self.safe_nb, _MIRROR[None, :]], 0.0)
        cl.set_leaf_phi(np.ascontiguousarray((phi + L - mirror).T))


# ---------------------------------------------------------------------------
# Self-test: v1-mode equivalence against the compiled step_cross,
# plus interior conservation for all three gates.
# ---------------------------------------------------------------------------


def _make_cluster(R=6.0, seed=7, lam0=0.5):
    from equilibrium import lattice as _lat

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(1)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    N = cl.n_leaves

    rng = np.random.default_rng(seed)
    nb = leaf_nb_j(cl)
    valid = nb >= 0
    safe = np.where(valid, nb, 0)
    A = rng.normal(size=(N, 14))
    phi = np.where(valid, A - A[safe, _MIRROR[None, :]], 0.0)  # antisymmetric
    cl.set_leaf_phi(np.ascontiguousarray(phi.T))
    cl.set_leaf_lam(np.full(N, lam0))
    return alat, cl, valid


def _selftest():
    dt, scale = 5e-3, 0.01

    # 1) v1-mode equivalence vs compiled step_cross
    alat_a, cl_a, _ = _make_cluster()
    alat_b, cl_b, valid = _make_cluster()
    alat_a.step_cross(0, dt, scale)
    CrossV2(cl_b, gate="v1").step(dt, scale)
    pa = np.array(cl_a.leaf_phi)
    pb = np.array(cl_b.leaf_phi)
    err = np.abs(pa - pb)[:, np.array(valid.any(axis=1))].max()
    print(f"v1 equivalence: max|phi_cpp - phi_py| = {err:.3e}")
    assert err < 1e-12, "v1-mode does not match compiled step_cross"

    # 2) interior conservation + antisymmetry for each gate
    for gate in ("v1", "shear", "product", "curl"):
        _, cl, valid = _make_cluster()
        op = CrossV2(cl, gate=gate, ref=1e-2)
        phi0 = np.array(cl.leaf_phi).T
        op.step(dt, scale)
        phi1 = np.array(cl.leaf_phi).T
        dphi = np.where(valid, phi1 - phi0, 0.0)
        # every valid face has a valid partner (bijection), so the global sum
        # over valid faces cancels pairwise; invalid faces receive no writes
        tot = dphi[valid].sum()
        anti = phi1 + np.where(valid, phi1[op.safe_nb, _MIRROR[None, :]], -phi1)
        print(
            f"gate={gate:8s} global sum(dphi)={tot:+.3e}"
            f"  max|antisymmetry viol|={np.abs(anti).max():.3e}"
            f"  |dphi|_max={np.abs(dphi).max():.3e}"
        )
        assert abs(tot) < 1e-9

    # 3) shear gate null on a coherent parallel-transit pattern:
    #    phi[i,k] = n_hat_k . d for every cell → all v_i parallel → m = 0
    _, cl, valid = _make_cluster()
    op = CrossV2(cl, gate="shear", ref=1e-2)
    d = np.array([0.3, -0.5, 0.8])
    pattern = _FACE_NORM @ d  # (14,)
    phi = np.where(valid, np.broadcast_to(pattern, valid.shape), 0.0)
    m = op.raw_shear(np.where(valid, phi, 0.0))
    # rim cells have masked faces, so their v_i legitimately points off-axis;
    # the null statement applies where both sides carry the full pattern
    interior = valid.all(axis=1)
    both_int = interior[:, None] & interior[op.safe_nb] & valid
    m_int = m[both_int]
    print(
        f"shear null on uniform coherent transit (interior pairs):"
        f" max m = {m_int.max():.3e}  (rim max {m.max():.3e})"
    )
    assert m_int.max() < 1e-12

    # 4) curl gate: B_i is EXACTLY zero on the same uniform coherent-transit
    #    pattern (stronger than shear's near-null above) — a holonomy read
    #    has no spatial variation to see, so no curl, by construction.
    B = op.curl_vectors(np.where(valid, phi, 0.0))
    print(f"curl null on uniform coherent transit: max|B| = {np.abs(B).max():.3e}")
    assert np.abs(B).max() < 1e-12

    print("cross_v2 selftest OK")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    _selftest()
