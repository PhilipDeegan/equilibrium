"""
Continuum view of the uniform (u, phi) substrate: faces as anchors.

Convention (Philip 2026-09-24): phi ~ A (vector potential), u ~ V.

Mapping. With hop vectors d_k (lattice units, |d|=1 hex, 2/sqrt3 sq), spacing h
and physical dt, a continuum pair (u, A) is sampled as

    u_i     = u(x_i)
    phi_i,k = W_k * A(x_i + h d_k / 2) . d_k

and step() is advanced with lattice dt_l = dt / h. The factor W_k is required:
step() drives each face as dphi_k = -dt_l sigma W_k (u_j - u_i), so a single
vector A satisfies A_t = -sigma grad u on every face only if phi_k carries W_k.
With that mapping step() is, to O(h^2), the continuum system

    u_t = -sigma C0SQ div A        (Lorenz gauge condition, c^2 = C0SQ)
    A_t = -sigma grad u            (E = 0)

C0SQ = 1/2 sum_k W_k^2 d_kx^2 (isotropic on BCC for any W).

Weights (set_weights; must match the C++ AdaptiveLattice's). The continuous
model approximates the discrete lattice's long-wave limit, so it has its own:
  "iso"       W = sqrt2 : 1 (default since 2026-09-28): star W^2 = 2:1, the only
              hex:square ratio whose 2nd-order error is direction independent in
              both sectors (exp/2026/09/28/hodge_star); C0SQ = 4, c = 2.
  "discrete"  step()'s W = sqrt3 : 1 (Kelvin star 3:1, since 2026-09-28);
              C0SQ = 16/3, c = 2.309. Needed to read a discrete lattice
              (advance(), which runs discrete step()).
  "legacy"    W = 3 : 1 (star 9:1), C0SQ = 40/3, c = 3.651: both models before
              2026-09-28; reproduces those runs (step() itself is no longer 9:1,
              so legacy can't drive advance()).

Time staggering: step() updates u from the old phi, then phi from the new u,
so the stored phi sits half a step ahead of u. Seed phi at t0 + dt/2.
"""

import numpy as np

from equilibrium.math.geo.kelvin import KelvinLattice
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.lattice import _W as _LATTICE_W

WEIGHTS = {
    "iso": np.array([2.0**0.5] * 8 + [1.0] * 6),
    "discrete": _LATTICE_W.copy(),  # step()'s, sqrt3 : 1 since 2026-09-28
    "legacy": np.array([3.0] * 8 + [1.0] * 6),  # step()'s before 2026-09-28
}
D_HAT = KelvinLattice(lattice_spacing=1.0).all_offsets
K_MIRROR = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12])
_D_PINV = np.linalg.pinv(D_HAT)


def set_weights(name):
    """Select the face weights ("iso" | "discrete"): rebinds W, C0SQ, C0 and
    the W-dependent operators; per-lattice caches are keyed on the choice."""
    global WEIGHTS_NAME, W, C0SQ, C0, _DW, _DW_PINV
    WEIGHTS_NAME = name
    W = WEIGHTS[name]
    C0SQ = 0.5 * float(np.sum(W**2 * D_HAT[:, 0] ** 2))
    C0 = C0SQ**0.5
    _DW = W[:, None] * D_HAT
    _DW_PINV = np.linalg.pinv(_DW)


set_weights("iso")


def _cached(lat, name):
    """Per-lattice cache entry valid for the current neighbours and weights."""
    c = getattr(lat, name, None)
    return c[2] if c is not None and c[0] is lat.nb and c[1] == WEIGHTS_NAME else None


def _store(lat, name, value):
    setattr(lat, name, (lat.nb, WEIGHTS_NAME, value))
    return value


def spacing_of(lat):
    """Hex-neighbour distance h, read off positions (minimum over cells: a periodic box's wrapped faces are longer)."""
    nb = lat.nb
    i = np.nonzero(nb[:, 0] >= 0)[0]
    return float(np.min(np.linalg.norm(lat.pos[nb[i, 0]] - lat.pos[i], axis=1)))


def lattice_dt(lat, dt):
    return dt / spacing_of(lat)


def stable_dt(lat, cfl=0.4):
    """Physical dt at the step() CFL bound (lattice 0.4 / sum W^2, times h)."""
    return cfl / float(np.sum(W**2)) * spacing_of(lat)


def face_midpoints(lat):
    h = spacing_of(lat)
    return lat.pos[:, None, :] + 0.5 * h * D_HAT[None, :, :]


def sample_u(lat, u_fn):
    return np.asarray(u_fn(lat.pos), dtype=np.float64)


def sample_A(lat, A_fn):
    """phi_i,k = W_k A(midpoint_i,k) . d_k. A_fn maps (M,3) -> (M,3).
    Antisymmetric across every face by construction (shared midpoint,
    d_mirror = -d_k)."""
    mid = face_midpoints(lat).reshape(-1, 3)
    A = np.asarray(A_fn(mid), dtype=np.float64).reshape(lat.N, 14, 3)
    phi = W[None, :] * np.einsum("nkd,kd->nk", A, D_HAT)
    return np.where(lat.nb >= 0, phi, 0.0)


def _recon_ops(lat):
    """Per-cell least-squares operators A = P_i (phi_i / W), cached on lat."""
    cache = getattr(lat, "_cont_recon", None)
    if cache is not None and cache[0] is lat.nb:
        return cache[1]
    valid = lat.nb >= 0
    edge = np.nonzero(~valid.all(axis=1))[0]
    P = np.broadcast_to(_D_PINV, (lat.N, 3, 14))
    if len(edge):
        P = P.copy()
    for i in edge:
        D = np.where(valid[i][:, None], D_HAT, 0.0)
        P[i] = np.linalg.pinv(D)
    lat._cont_recon = (lat.nb, P)
    return P


def _recon_rows(lat, idx):
    """_recon_ops restricted to cells idx, without the (N, 3, 14) cache."""
    valid = lat.nb[idx] >= 0
    P = np.broadcast_to(_D_PINV, (len(idx), 3, 14)).copy()
    for r in np.nonzero(~valid.all(axis=1))[0]:
        P[r] = np.linalg.pinv(np.where(valid[r][:, None], D_HAT, 0.0))
    return P


def reconstruct_A_at(lat, idx, phi_rows):
    """reconstruct_A at cells idx; phi_rows = phi[idx]."""
    a = np.where(lat.nb[idx] >= 0, phi_rows, 0.0) / W[None, :]
    return np.einsum("ndk,nk->nd", _recon_rows(lat, idx), a)


def curl_A_at(lat, idx, rows):
    """curl_A at cells idx; rows(j) returns phi[j] for an index array j.
    Touches only idx and its neighbours."""
    nb = lat.nb[idx]
    valid = nb >= 0
    need = np.union1d(idx, nb[valid])
    A = reconstruct_A_at(lat, need, rows(need))
    Ai = A[np.searchsorted(need, idx)]
    Aj = A[np.searchsorted(need, np.where(valid, nb, idx[:, None]))]
    grad = np.einsum("ndk,nke->nde", _recon_rows(lat, idx), Aj - Ai[:, None, :]) / spacing_of(lat)
    return np.stack(
        [grad[:, 1, 2] - grad[:, 2, 1], grad[:, 2, 0] - grad[:, 0, 2], grad[:, 0, 1] - grad[:, 1, 0]], axis=1
    )


def drop_caches(lat):
    """Free the per-cell (N, 3, 14) operator caches (336 B/cell each) and the
    DEC d1 (~1.5 KB/cell)."""
    for a in ("_cont_recon", "_cont_recon_w", "_cont_dec"):
        if hasattr(lat, a):
            delattr(lat, a)


def reconstruct_A(lat, phi=None):
    """Cell-centred A from the 14 anchors: least squares of phi_k/W_k = A.d_k
    over the faces present. Exact for linear A at interior cells."""
    phi = lat.phi if phi is None else phi
    a = np.where(lat.nb >= 0, phi, 0.0) / W[None, :]
    return np.einsum("ndk,nk->nd", _recon_ops(lat), a)


def anchor_residual(lat, phi=None):
    """Odd part of each opposite-face pair minus A_rec . d_k. The even part
    carries grad A (1/2 d.G.d) and is not a residual. phi is antisymmetric,
    so a cell owns 7 face DOF: 3 are A, the other 4 are what a single smooth
    vector field cannot hold. O(h^2) for a sampled smooth A; zero where a
    face is missing (not defined there)."""
    phi = lat.phi if phi is None else phi
    valid = lat.nb >= 0
    a = phi / W[None, :]
    odd = 0.5 * (a - a[:, K_MIRROR])
    fit = reconstruct_A(lat, phi) @ D_HAT.T
    ok = valid & valid[:, K_MIRROR]
    return np.where(ok, odd - fit, 0.0)


def div_A(lat, phi=None):
    """Continuum div A from the anchors: sum_k W_k phi_k / (C0SQ h)."""
    phi = lat.phi if phi is None else phi
    return (np.where(lat.nb >= 0, phi, 0.0) @ W) / (C0SQ * spacing_of(lat))


def energy_density(lat):
    """Per-cell energy of step(): u^2/(2 sigma) + 1/4 sum_k phi_k^2 / sigma_face.
    For sampled (u, A) at sigma=1 this is (u^2 + C0SQ |A|^2)/2."""
    nb = lat.nb
    valid = nb >= 0
    sigma = 1.0 / (1.0 + lat.lam)
    s_nb = np.where(valid, sigma[np.where(valid, nb, 0)], sigma[:, None])
    s_face = 0.5 * (sigma[:, None] + s_nb)
    return lat.u**2 / (2 * sigma) + 0.25 * np.sum(np.where(valid, lat.phi**2 / s_face, 0.0), axis=1)


def advance(lat, dt, n_steps, n_threads=1):
    """n_steps of the discrete step() at physical dt (lattice dt = dt / h);
    needs set_weights("discrete") (the anchor mapping of step()'s W)."""
    if WEIGHTS_NAME != "discrete":
        raise ValueError('advance() runs the discrete step(): call set_weights("discrete") first')
    # (legacy runs of advance() need a pre-2026-09-28 checkout: step() is W = sqrt3 now)
    dt_l = lattice_dt(lat, dt)
    step = lat.step if isinstance(lat, DampedLattice) else lat.step_cpp
    for _ in range(n_steps):
        step(dt_l, n_threads)
    return n_steps * dt


def plane_wave(direction, f, c=None):
    """Exact continuum solution of the sigma=1 system: u = f(n.x - c t),
    A = n f(n.x - c t) / c. Returns (u_fn(x, t), A_fn(x, t))."""
    c = C0 if c is None else c
    n = np.asarray(direction, dtype=np.float64)
    n = n / np.linalg.norm(n)

    def u_fn(x, t):
        return f(x @ n - c * t)

    def A_fn(x, t):
        return (f(x @ n - c * t) / c)[:, None] * n[None, :]

    return u_fn, A_fn


def seed(lat, u_fn, A_fn, t0=0.0, dt=None):
    """Seed u at t0 and phi at t0 + dt/2 (step()'s staggering); dt=None seeds
    both at t0."""
    t_phi = t0 + 0.5 * dt if dt is not None else t0
    lat.u = sample_u(lat, lambda x: u_fn(x, t0))
    lat.phi = sample_A(lat, lambda x: A_fn(x, t_phi))


LAP_W = KelvinLattice().laplacian_weights


def _gather(lat, f):
    nb = lat.nb
    valid = nb >= 0
    fj = f[np.where(valid, nb, 0)]
    fi = f[:, None] if f.ndim == 1 else f[:, None, :]
    m = valid if f.ndim == 1 else valid[..., None]
    return np.where(m, fj, fi)


def laplacian(lat, f):
    """BCC Laplacian (w=2/3 hex, 1/3 sq, / a^2), zero-gradient at missing faces.
    f is (N,) or (N,3)."""
    a = spacing_of(lat) * 2.0 / 3.0**0.5
    d = _gather(lat, f) - (f[:, None] if f.ndim == 1 else f[:, None, :])
    return np.einsum("k,nk...->n...", LAP_W, d) / a**2


def sample_faces(lat, V_cell):
    """Anchor samples W_k V(midpoint).d_k of a cell-centred vector field,
    midpoint value = mean of the two cells."""
    Vm = 0.5 * (V_cell[:, None, :] + _gather(lat, V_cell))
    return np.where(lat.nb >= 0, W[None, :] * np.einsum("nkd,kd->nk", Vm, D_HAT), 0.0)


def curl_curl_faces(lat, phi=None):
    """Anchor samples of curl curl A = grad div A - lap A, from the anchors.
    grad div at the midpoint along d_k: (div_j - div_i)/h."""
    phi = lat.phi if phi is None else phi
    h = spacing_of(lat)
    div = div_A(lat, phi)
    gdiv = W[None, :] * (_gather(lat, div) - div[:, None]) / h
    gdiv = np.where(lat.nb >= 0, gdiv, 0.0)
    return gdiv - sample_faces(lat, laplacian(lat, reconstruct_A(lat, phi)))


def curl_A(lat, phi=None):
    """Cell-centred B = curl A from least-squares gradients of the
    reconstructed A over the 14 neighbours."""
    phi = lat.phi if phi is None else phi
    A = reconstruct_A(lat, phi)
    dA = _gather(lat, A) - A[:, None, :]
    h = spacing_of(lat)
    P = _recon_ops(lat)
    grad = np.einsum("ndk,nke->nde", P, dA) / h
    return np.stack(
        [
            grad[:, 1, 2] - grad[:, 2, 1],
            grad[:, 2, 0] - grad[:, 0, 2],
            grad[:, 0, 1] - grad[:, 1, 0],
        ],
        axis=1,
    )


class AE:
    """EXPERIMENTAL extension of step() with an E field at the face anchors
    (same sampling as phi). Continuum system, sigma = 1:

        u_t   = -c^2 div A
        A_t   = -grad u - E
        E_t   =  c^2 curl curl A - J

    step() is the E = 0, J = 0 sector. Order per step: u (old phi), E (old
    phi), then phi (new u, new E), so phi is half a step ahead of u and E.
    Numpy only; lam must be 0.

    Boundary: none beyond curl_curl_sym's own adjoint closure (edge_mask=True
    holds E = 0 within two layers of the edge, but that breaks the energy
    invariant -- measured: |u| 8 -> 500 on noise -- so it is off by default).

    curl curl is curl_curl_sym (K^T K with K's exact adjoint): symmetric, so
    E and A exchange energy exactly and no mode grows. The first version used
    curl_curl_faces (grad div - Laplacian of the reconstruction), which is not
    symmetric and had modes growing ~2.3 e-folds per unit time at every dt.
    curl_curl_sym is NOT discretely div-free: sum_k W_k cc_k is O(h^2) for
    smooth fields (0.11 / 0.017 / 0.0022 of |cc| at h = 0.5 / 0.25 / 0.125,
    exp/2026/09/27/dec), the transpose side of the curl-grad leak. clean=True
    projects onto the discretely div-free subspace; off by default.
    Default curl="dec": curl_curl_dec with the u sector's star (exactly
    div-free in sum_k W_k, far less dispersive); "dec_geo" the geometric star;
    "ls" the curl_curl_sym reference described above."""

    def __init__(self, lat, e=None, clean=False, edge_mask=False, curl="dec"):
        self.lat = lat
        self.clean = clean
        self._cc = curl_curl_sym if curl == "ls" else (lambda lat_: curl_curl_dec(lat_, star=curl))
        self._psi = None
        nb = lat.nb
        safe = np.where(nb >= 0, nb, 0)
        full = (nb >= 0).all(axis=1)
        deep = full & full[safe].all(axis=1)
        self.e_mask = deep[:, None] & (nb >= 0) & deep[safe] if edge_mask else (nb >= 0)
        self.e = np.zeros_like(lat.phi) if e is None else np.where(self.e_mask, e, 0.0)
        self._W_nb = np.where(nb >= 0, W[None, :], 0.0)

    def E_cell(self):
        return reconstruct_A(self.lat, self.e)

    def step(self, dt, J_faces=None):
        lat = self.lat
        h = spacing_of(lat)
        valid = lat.nb >= 0
        du = -(dt / h) * (np.where(valid, lat.phi, 0.0) @ W)
        u_new = np.where(lat.fixed_u, lat.u, lat.u + du)
        cc = self._cc(lat)
        if self.clean:
            cc, self._psi = project_div_free(lat, np.where(self.e_mask, cc, 0.0), tol=1e-10,
                                             x0=self._psi, return_psi=True)
        de = dt * C0SQ * cc
        if J_faces is not None:
            de = de - dt * J_faces
        self.e = np.where(self.e_mask, self.e + de, 0.0)
        grad_u = self._W_nb * (_gather(lat, u_new) - u_new[:, None])
        dphi = -(dt / h) * grad_u - dt * self.e
        lat.u = u_new
        lat.phi = np.where(valid & ~lat.fixed_phi, lat.phi + dphi, lat.phi)

    def advance(self, dt, n_steps, J_faces=None):
        for _ in range(n_steps):
            self.step(dt, J_faces)
        return n_steps * dt


def _graph_laplacian(lat):
    """L psi_i = sum_k W_k^2 (psi_j - psi_i): div of the anchor-sampled
    gradient W_k (psi_j - psi_i). Cached on lat."""
    cache = _cached(lat, "_cont_lap")
    if cache is not None:
        return cache
    from scipy.sparse import coo_matrix

    nb = lat.nb
    valid = nb >= 0
    i = np.repeat(np.arange(lat.N), 14).reshape(lat.N, 14)[valid]
    j = nb[valid]
    w2 = np.broadcast_to(W**2, nb.shape)[valid]
    L = coo_matrix((w2, (i, j)), shape=(lat.N, lat.N)).tocsr()
    L = L - coo_matrix((np.asarray(L.sum(axis=1)).ravel(), (np.arange(lat.N),) * 2), shape=L.shape)
    return _store(lat, "_cont_lap", L.tocsr())


def project_div_free(lat, faces, tol=1e-12, x0=None, return_psi=False):
    """Remove the anchor-gradient part of face samples so that
    sum_k W_k faces_k = 0 at every cell (discrete div-free, exact to tol)."""
    from scipy.sparse.linalg import cg

    valid = lat.nb >= 0
    f = np.where(valid, faces, 0.0)
    rhs = f @ W
    rhs = rhs - rhs.mean()
    L = _graph_laplacian(lat)
    scale = max(float(np.abs(rhs).max()), 1e-300)
    psi, info = cg(-L, -rhs / scale, x0=None if x0 is None else x0 / scale, rtol=tol, atol=0.0, maxiter=20000)
    if info != 0:
        raise RuntimeError(f"project_div_free: cg did not converge ({info})")
    psi *= scale
    grad = np.where(valid, W[None, :] * (_gather(lat, psi) - psi[:, None]), 0.0)
    out = np.where(valid, f - grad, 0.0)
    return (out, psi) if return_psi else out




def _recon_ops_w(lat):
    """Per-cell W-weighted least squares A = pinv(diag(W) D) phi: the
    Euclidean adjoint of sample_A's W_k A.d_k (step()'s energy metric is
    Euclidean in phi). Cached on lat."""
    cache = _cached(lat, "_cont_recon_w")
    if cache is not None:
        return cache
    valid = lat.nb >= 0
    edge = np.nonzero(~valid.all(axis=1))[0]
    P = np.broadcast_to(_DW_PINV, (lat.N, 3, 14))
    if len(edge):
        P = P.copy()
    for i in edge:
        P[i] = np.linalg.pinv(np.where(valid[i][:, None], _DW, 0.0))
    return _store(lat, "_cont_recon_w", P)


def _antisym(lat, x):
    """Per-face-copy -> antisymmetric: x_ik - x_j,mirror(k); 0 on missing faces."""
    nb = lat.nb
    valid = nb >= 0
    back = x[np.where(valid, nb, 0), K_MIRROR[None, :]]
    return np.where(valid, x - back, 0.0)


def _curl_of_grad(G):
    return np.stack([G[:, 1, 2] - G[:, 2, 1], G[:, 2, 0] - G[:, 0, 2], G[:, 0, 1] - G[:, 1, 0]], axis=1)


def curl_K(lat, phi):
    """K phi: cell B = curl A, A from W-weighted reconstruction, gradient by
    the same per-cell least squares on neighbour differences."""
    P = _recon_ops_w(lat)
    A = np.einsum("ndk,nk->nd", P, np.where(lat.nb >= 0, phi, 0.0))
    dA = _gather(lat, A) - A[:, None, :]
    Pg = _recon_ops(lat)
    G = np.einsum("ndk,nke->nde", Pg, dA) / spacing_of(lat)
    return _curl_of_grad(G)


def curl_K_adjoint(lat, B):
    """Exact Euclidean adjoint of curl_K."""
    nb = lat.nb
    valid = nb >= 0
    safe = np.where(valid, nb, 0)
    G = np.zeros((lat.N, 3, 3))
    G[:, 1, 2] += B[:, 0]
    G[:, 2, 1] -= B[:, 0]
    G[:, 2, 0] += B[:, 1]
    G[:, 0, 2] -= B[:, 1]
    G[:, 0, 1] += B[:, 2]
    G[:, 1, 0] -= B[:, 2]
    dA = np.einsum("ndk,nde->nke", _recon_ops(lat), G) / spacing_of(lat)
    dA = np.where(valid[..., None], dA, 0.0)
    A = -dA.sum(axis=1)
    np.add.at(A, safe[valid], dA[valid])
    return np.where(valid, np.einsum("ndk,nd->nk", _recon_ops_w(lat), A), 0.0)


def curl_curl_sym(lat, phi=None):
    """Symmetric positive semi-definite curl curl on the anchors:
    C0SQ * antisym(K^T K phi), K = curl_K. The factor C0SQ (40/3 discrete) is the
    measured limit of a fitted scale (13.380 / 13.345 / 13.336 at h = 0.5 /
    0.25 / 0.125). Its output is NOT Euclidean-orthogonal to anchor
    gradients (<grad u, K^T K phi> = <K grad u, K phi>), so sum_k W_k cc_k
    is O(h^2), not zero. Nor does it vanish on gradients: the gradient
    W_k (u_j - u_i) of a non-linear u reconstructs to an A with curl, which
    vanishes at 2nd order for smooth u and is O(1) for grid-scale u."""
    phi = lat.phi if phi is None else phi
    return C0SQ * _antisym(lat, curl_K_adjoint(lat, curl_K(lat, phi)))


_DEC_TRIS = [(s, k, m) for s in (9, 11, 13) for k in range(8) for m in range(8)
             if np.allclose(D_HAT[k] + D_HAT[m], D_HAT[s])]


def _dec_ops(lat):
    """Sparse d1 over the BCC Delaunay triangles (i -k-> j -m-> l, d_s = d_k +
    d_m, s = +x/+y/+z square hop; 12 per cell), acting on the face copies
    phi (N*14) as edge integrals a = h phi / W. Cached on lat."""
    cache = _cached(lat, "_cont_dec")
    if cache is not None:
        return cache
    import scipy.sparse as sp

    nb = lat.nb
    h = spacing_of(lat)
    rows, cols, vals, t = [], [], [], 0
    for s, k, m in _DEC_TRIS:
        i = np.nonzero((nb[:, k] >= 0) & (nb[:, s] >= 0))[0]
        j = nb[i, k]
        ok = nb[j, m] == nb[i, s]
        i, j = i[ok], j[ok]
        r = t + np.arange(len(i))
        rows += [r, r, r]
        cols += [i * 14 + k, j * 14 + m, i * 14 + s]
        vals += [np.full(len(i), h / W[k]), np.full(len(i), h / W[m]), np.full(len(i), -h / W[s])]
        t += len(i)
    D1 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(t, lat.N * 14))
    return _store(lat, "_cont_dec", (D1, D1.T.tocsr()))


def dec_star(star):
    """(r, s): *1 = r a_c/8 (hex) : a_c/8 (square), *2 = s/a_c. In a = phi/W the
    operator is plain DEC times C0SQ, so c_T^2 = C0SQ 4s/(r+1) and s = (r+1)/4
    gives c_T = c. "dec": r = (W_hex/W_sq)^2 (2 iso, 9 discrete); "dec_geo": the
    coarse cell's Kelvin geometry, r = 3."""
    r = 3.0 if star == "dec_geo" else float(W[0] ** 2 / W[8] ** 2)
    return r, 0.25 * (r + 1.0)


def curl_curl_dec(lat, phi=None, star="dec"):
    """DEC curl curl on the BCC Delaunay / Kelvin complex: circulation of the
    edge integrals around each Delaunay triangle, face sample of curl curl A =
    (W/h)^2 antisym(D1^T *2 D1 phi) / *1 (D1 carries h/W), a_c = 2h/sqrt3.
    curl(grad) = 0 exactly (d1 d0 = 0, any star); symmetric in the *1/W^2
    face metric; exactly div-free in sum_k (*1_k / W_k) out_k. Stars (dec_star):
      "dec"     r = W_hex^2/W_sq^2, the u sector's own star, so sum_k W_k out_k
                = 0 exactly (no transverse -> u coupling). iso: r = 2, error
                isotropic (exp/2026/09/28/hodge_star); discrete: r = 9, the
                09-27 default (transverse -0.011..-0.020 (kh)^2).
      "dec_geo" r = 3 = |Kelvin face|/|hop|, s = 1: the coarse cell's geometry;
                div-free in sum_k W_k only if W_hex/W_sq = sqrt3, so it leaks
                into u under both weights.
    curl_curl_sym: -0.12..-0.18 (exp/2026/09/27/dispersion). The 4 extra face
    DOF per cell are gapped (omega h / c ~ 4.5-8); a smooth field puts
    O((kh)^2) amplitude there, so the pointwise output error is O(1) while the
    evolved fields converge (exp/2026/09/27/dec)."""
    phi = lat.phi if phi is None else phi
    r, s2 = dec_star(star)
    D1, D1T = _dec_ops(lat)
    h = spacing_of(lat)
    ac = 2 * h / 3**0.5
    star1 = np.where(np.arange(14) < 8, r * ac / 8, ac / 8)
    y = (D1T @ (D1 @ np.where(lat.nb >= 0, phi, 0.0).ravel()) * (s2 / ac)).reshape(-1, 14)
    y = _antisym(lat, y) / star1[None, :]
    return np.where(lat.nb >= 0, (W / h)[None, :] ** 2 * y, 0.0)


def cartan_transport(lat, phi=None, hold=None):
    """Self-transport of phi as -i_v(da) (Cartan: L_v a = i_v da + d(i_v a), the
    gradient part left to step()), a = h phi / W the edge integrals, da the
    Delaunay triangle loop sums. Per triangle: edge e += omega_t (a_next -
    a_prev) along the loop, times 3/4 / (W_e h^3) to a face sample. No
    vectors: loop sums and the triangle's other two edges only. Antisymmetric
    per triangle in v = W phi, so sum phi . out = 0 exactly (step()'s energy);
    zero on a face whose triangles all have omega = 0, so no circulation is
    created in irrotational flow (the 10-11 exchange-matrix commutator does,
    rising under refinement). 3/4 and hex-hex : hex-square = 1/2 (in v) make
    the sum over a face's triangles exactly -W (Omega x u) . d for uniform
    curl Omega and flow u, d = D_HAT (exp/2026/10/13/cartan/fit.py). kappa = 1 is Euler's
    -i_u du in these units; it matches the commutator's ring speed at its
    picked kappa 0.132 to 0.5 %.
    hold (N,): context-blended vorticity, om -> kh om + (1 - kh) <om>, kh the
    mean hold of the triangle's three cells, <om> the same template's om
    averaged over the cell and its face neighbours (exp/2026/10/22/depression/
    contextual_hold.md); still a scalar per triangle, so energy-orthogonal."""
    phi = lat.phi if phi is None else phi
    nb = lat.nb
    h = spacing_of(lat)
    a = np.where(nb >= 0, phi, 0.0) / W[None, :]
    out = np.zeros_like(a)
    for s, k, m in _DEC_TRIS:
        i = np.nonzero((nb[:, k] >= 0) & (nb[:, s] >= 0))[0]
        j = nb[i, k]
        ok = nb[j, m] == nb[i, s]
        i, j = i[ok], j[ok]
        a1, a2, a3 = a[i, k], a[j, m], -a[i, s]
        w = a1 + a2 + a3
        if hold is not None:
            om = np.full(lat.N, np.nan)
            om[i] = w
            g = np.where(nb >= 0, nb, 0)
            nbo = np.where(nb >= 0, om[g], np.nan)
            tot = np.nansum(nbo, 1)[i] + w
            cnt = np.sum(~np.isnan(nbo), 1)[i] + 1
            kh = (hold[i] + hold[j] + hold[nb[i, s]]) / 3
            w = kh * w + (1 - kh) * tot / cnt
        out[i, k] += w * (a2 - a3)
        out[j, m] += w * (a3 - a1)
        out[i, s] -= w * (a1 - a2)
    out = _antisym(lat, out)
    return np.where(nb >= 0, 0.75 / (h * W[None, :]) * out, 0.0)


def current_from_magnetisation(lat, m_cell):
    """J = curl m on the anchors, as C0SQ * antisym(K^T m) for a cell-centred
    m: lies in range(K^T), so C0SQ * curl_curl_sym(A) = J has a static
    solution. Anchor-sampled J = sample_A(curl m) does not: its null(K) part
    (non-gradient, curl-free on the anchors) has no restoring force and grows
    linearly under AE."""
    return C0SQ * _antisym(lat, curl_K_adjoint(lat, m_cell))


class AnchorCluster:
    """numpy view of cluster i of a C++ lattice.continuous.AdaptiveLattice
    (level 0, rebuilt), with the attributes the functions above read:
    pos, nb (leaf-indexed), u, phi (N, 14), lam, fixed_u, fixed_phi. Reads
    and writes go through the cluster; nb and pos are fixed at construction."""

    def __init__(self, alat, i=0):
        if alat.weights != WEIGHTS_NAME:
            raise ValueError(f"AnchorCluster: C++ weights {alat.weights!r} != continuous.set_weights({WEIGHTS_NAME!r})")
        self.alat, self.i = alat, i
        self.cl = alat[i]
        self.pos = np.array(self.cl.leaf_pos)
        self.nb = np.array(alat.leaf_nb_j(i))
        self.N = len(self.pos)

    u = property(lambda s: np.array(s.cl.leaf_u), lambda s, v: s.cl.set_leaf_u(np.ascontiguousarray(v, dtype=np.float64)))
    phi = property(
        lambda s: np.array(s.cl.leaf_phi).T.copy(),
        lambda s, v: s.cl.set_leaf_phi(np.ascontiguousarray(np.asarray(v, dtype=np.float64).T)),
    )
    lam = property(lambda s: np.array(s.cl.leaf_lam))
    fixed_u = property(lambda s: np.array(s.cl.leaf_fixed_u).astype(bool))
    fixed_phi = property(lambda s: ((np.array(s.cl.leaf_fixed_phi)[:, None] >> np.arange(14)) & 1).astype(bool))

    @property
    def e(self):
        return np.array(self.alat.leaf_e(self.i)).T.copy()

    @e.setter
    def e(self, v):
        self.alat.set_leaf_e(self.i, np.ascontiguousarray(np.asarray(v, dtype=np.float64).T))

    def curl_curl_sym(self, phi=None):
        phi = self.phi if phi is None else phi
        return np.array(self.alat.curl_curl_sym(self.i, np.ascontiguousarray(phi.T))).T.copy()

    def curl_curl_dec(self, phi=None, star="dec"):
        phi = self.phi if phi is None else phi
        return np.array(self.alat.curl_curl_dec(self.i, np.ascontiguousarray(phi.T), star)).T.copy()

    def cartan_transport(self, phi=None):
        phi = self.phi if phi is None else phi
        return np.array(self.alat.cartan_transport(self.i, np.ascontiguousarray(phi.T))).T.copy()

    def set_cartan_hold(self, kappa):
        """Per-leaf hold kappa of the Cartan vorticity (None clears); see continuous.cartan_transport(hold=)."""
        if kappa is None:
            return self.alat.clear_cartan_hold(self.i)
        self.alat.set_cartan_hold(self.i, np.array(np.broadcast_to(np.asarray(kappa, dtype=np.float64), (self.N,))))

    def set_curl(self, curl):
        """'ls' (curl_curl_sym), 'dec' or 'dec_geo' (curl_curl_dec stars) for step_ae."""
        self.alat.set_curl(curl)

    def set_face_medium(self, dsig, dn):
        """Medium at the face anchors, (N, 14): deviations of sigma (tick rate)
        and n = L/h (coarse size / mean size of the discrete cells covered)."""
        f = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v, dtype=np.float64), (self.N, 14)).T)
        self.alat.set_face_medium(self.i, f(dsig), f(dn))

    def phi_rows(self, idx):
        return np.asarray(self.cl.leaf_phi)[:, idx].T

    def e_rows(self, idx):
        return np.asarray(self.alat.leaf_e(self.i))[:, idx].T

    @staticmethod
    def faces_T(J_faces):
        """(N, 14) -> the (14, N) layout step_ae takes without a copy."""
        return np.ascontiguousarray(np.asarray(J_faces, dtype=np.float64).T)

    def step_ae(self, dt, J_faces=None, scale=1.0, J_T=None):
        """J = scale * J_faces; pass J_T = faces_T(J) instead to reuse one layout."""
        J = J_T if J_T is not None else (None if J_faces is None else self.faces_T(J_faces))
        self.alat.step_ae(self.i, dt, J, scale)


def anchor_cluster(R, h, n_threads=8):
    """Level-0 continuous AdaptiveLattice sphere, radius R (physical), hex
    spacing h, with the current set_weights; returns its AnchorCluster."""
    from equilibrium import lattice as _lat

    alat = _lat.continuous.AdaptiveLattice()
    alat.set_weights(WEIGHTS_NAME)
    _lat.continuous.build_bcc_sphere(alat[0], R / h, d1=h, u0=0.0, lam0=0.0)
    alat.rebuild(0)
    alat.set_n_threads(n_threads)
    return AnchorCluster(alat, 0)
