"""DEC curl curl on the BCC Delaunay / Kelvin complex, as a drop-in for continuous.curl_curl_sym (numpy,
sparse). Faces carry phi_ik = W_k A(mid).d_k, so the edge integral is a = h phi / W. Triangles
(i -k-> j -m-> l, l = i + d_s) with d_s = d_k + d_m, s a square hop (12 per cell). Kelvin stars
(a_c = 2h/sqrt3): *1 = 3a_c/8 (hex), a_c/8 (square); *2 = 1/a_c. Output on the faces:
D1 holds the h/W conversion, so the pure incidence transpose is (W/h) D1^T:
cc_face = (W/h)^2 * antisym(D1^T *2 D1 phi) / *1, the face sample of curl curl A. `star_ratio` overrides
the hex:square *1 ratio (3 = geometric) keeping the square value.
"""
import numpy as np
import scipy.sparse as sp

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

W, D, MIR = C.W, C.D_HAT, C.K_MIRROR
TPL = [(s, k, m) for s in (9, 11, 13) for k in range(8) for m in range(8) if np.allclose(D[k] + D[m], D[s])]
assert len(TPL) == 12


class DecCurl:
    def __init__(self, lat, star_ratio=3.0):
        self.lat = lat
        nb = lat.nb
        self.h = h = C.spacing_of(lat)
        ac = 2 * h / 3**0.5
        self.star1 = np.where(np.arange(14) < 8, star_ratio * ac / 8, ac / 8)
        self.star2 = 1.0 / ac
        N = len(nb)
        rows, cols, vals = [], [], []
        t = 0
        for s, k, m in TPL:
            i = np.nonzero((nb[:, k] >= 0) & (nb[:, s] >= 0))[0]
            j = nb[i, k]
            ok = nb[j, m] == nb[i, s]
            i, j = i[ok], j[ok]
            n = len(i)
            r = t + np.arange(n)
            rows += [r, r, r]
            cols += [i * 14 + k, j * 14 + m, i * 14 + s]
            vals += [np.full(n, h / W[k]), np.full(n, h / W[m]), np.full(n, -h / W[s])]
            t += n
        self.n_tri = t
        self.D1 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(t, N * 14))
        self.D1T = self.D1.T.tocsr()

    def circulation(self, phi):
        return self.D1 @ np.where(self.lat.nb >= 0, phi, 0.0).ravel()

    def __call__(self, phi):
        y = (self.D1T @ (self.star2 * self.circulation(phi))).reshape(-1, 14)
        y = C._antisym(self.lat, y) / self.star1[None, :]
        return np.where(self.lat.nb >= 0, (W / self.h)[None, :] ** 2 * y, 0.0)


class DecAE:
    """continuous.AE with curl_curl_sym replaced by DecCurl (C0SQ * cc, same staggering)."""

    def __init__(self, lat, e=None, star_ratio=3.0, **_):
        self.lat = lat
        self.cc = DecCurl(lat, star_ratio)
        nb = lat.nb
        self.e = np.zeros_like(lat.phi) if e is None else np.where(nb >= 0, e, 0.0)
        self._W_nb = np.where(nb >= 0, W[None, :], 0.0)

    def E_cell(self):
        return C.reconstruct_A(self.lat, self.e)

    def step(self, dt, J_faces=None):
        lat = self.lat
        h = self.cc.h
        valid = lat.nb >= 0
        du = -(dt / h) * (np.where(valid, lat.phi, 0.0) @ W)
        u_new = np.where(lat.fixed_u, lat.u, lat.u + du)
        de = dt * C.C0SQ * self.cc(lat.phi)
        if J_faces is not None:
            de = de - dt * J_faces
        self.e = np.where(valid, self.e + de, 0.0)
        grad_u = self._W_nb * (C._gather(lat, u_new) - u_new[:, None])
        lat.u = u_new
        lat.phi = np.where(valid & ~lat.fixed_phi, lat.phi - (dt / h) * grad_u - dt * self.e, lat.phi)

    def advance(self, dt, n_steps, J_faces=None):
        for _ in range(n_steps):
            self.step(dt, J_faces)
        return n_steps * dt
