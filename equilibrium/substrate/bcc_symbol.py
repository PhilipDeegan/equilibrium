"""Bloch symbols of DEC on the BCC Delaunay / Kelvin (Voronoi) complex, with a free hex:square star.

Cube constant a = 1: hex hops (+-1/2)^3 (length h = sqrt3/2), square hops the unit axes; 12
hex-hex-square Delaunay triangles per vertex. *0 = 1/2 (Kelvin cell volume), *2 = 1,
*1 = (r/8 hex, 1/8 square); r = 3 is the Kelvin geometry (|dual face| / |edge|), step() is r = 9
(W^2, W = 3:1 applied on both legs).
  u sector (0-forms):  omega^2 = *0^-1 d0^H *1 d0
  T sector (1-forms):  omega^2 = *1^-1 d1^H *2 d1   (curl curl; 3 exact zero modes = gradients,
                       2 transverse branches, 4 gapped)
Used by exp/2026/09/28/hodge_star/star_ratio.py and tst/substrate/test_hodge_star.py; the 09-27
exp/2026/09/27/dispersion/dec_symbol.py is the fixed-star original.
"""

import itertools

import numpy as np

S3 = np.sqrt(3.0)
H = S3 / 2  # hex distance
HEX = [np.array(v) * 0.5 for v in itertools.product((1, -1), repeat=3)]
SQ = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
STAR0 = 0.5


def _key(v):
    return tuple(np.round(2 * v).astype(int))


OWNED = [v for v in HEX + SQ if _key(v) > _key(-v)]
_OIDX = {_key(v): i for i, v in enumerate(OWNED)}
IS_HEX = np.array([abs(abs(v[0]) - 0.5) < 1e-9 for v in OWNED])
TRIS = [(eh, es) for es in OWNED if abs(abs(es).max() - 1) < 1e-9 for eh in HEX
        if any(np.allclose(es - eh, x) for x in HEX)]
assert len(TRIS) == 12


def kelvin_star1():
    """(*1 hex, *1 square, cell volume) from the truncated octahedron: |dual face| / |primal edge|."""
    s = 1 / (2 * np.sqrt(2))  # Kelvin cell edge
    return 1.5 * S3 * s**2 / H, s**2 / 1.0, 8 * np.sqrt(2) * s**3


def star1(r):
    return np.where(IS_HEX, r / 8, 1 / 8)


def _edge_row(x0, e, k):
    row = np.zeros(len(OWNED), complex)
    if _key(e) in _OIDX:
        row[_OIDX[_key(e)]] += np.exp(1j * k @ x0)
    else:
        row[_OIDX[_key(-e)]] -= np.exp(1j * k @ (x0 + e))
    return row


def d0(k):
    return np.array([np.exp(1j * k @ e) - 1 for e in OWNED])


def d1(k):
    z = np.zeros(3)
    return np.array([_edge_row(z, eh, k) + _edge_row(eh, es - eh, k) + _edge_row(es, -es, k) for eh, es in TRIS])


def w2_u(k, r):
    return float((star1(r) * np.abs(d0(k)) ** 2).sum() / STAR0)


def spectrum_T(k, r):
    D1 = d1(k) / np.sqrt(star1(r))[None, :]  # similar Hermitian form *1^-1/2 d1^H d1 *1^-1/2
    return np.linalg.eigvalsh(D1.conj().T @ D1)


def w2_T(k, r):
    """The two transverse branches (the 3 zero modes are exact gradients, O(1e-8) numerically)."""
    ev = spectrum_T(k, r)
    return ev[ev > 0.05 * (k @ k)][:2]


def speed_error(n, r, kh=0.02):
    """Spatial dc/c / (kh)^2 along unit n: (u, (T1, T2)), each sector normalised to its exact k -> 0
    speed, c_u^2 = (r+1)/4, c_T^2 = 4/(r+1) (asserted in test_hodge_star; measuring c at small k instead
    puts its round-off / (kh)^2 into the result)."""
    k = kh / H
    cu, cT = np.sqrt((r + 1) / 4), np.sqrt(4 / (r + 1))
    du = (np.sqrt(w2_u(k * n, r)) / (cu * k) - 1) / kh**2
    dT = (np.sqrt(w2_T(k * n, r)) / (cT * k) - 1) / kh**2
    return du, dT


def directions(n=400, seed=1):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    v = np.vstack([v, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 2, 3]]])
    return v / np.linalg.norm(v, axis=1)[:, None]
