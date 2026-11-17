"""Bloch symbols of discrete exterior calculus (DEC) on the BCC Delaunay / Kelvin (Voronoi) complex.

Cube constant a = 1: hex hops e = (+-1/2)^3 (length h = sqrt3/2), square hops (+-1,0,0).. (length 1).
Delaunay triangles: (0, e_h, e_s) with e_s - e_h also hex (all triangles are hex-hex-square; 12/vertex).
Hodge stars from the Kelvin cell (edge s = 1/(2 sqrt2)): *0 = 1/2 (cell volume);
*1 = |dual face| / |edge|: hex 3/8, square 1/8 (ratio 3 = W); *2 = |Kelvin edge| / |triangle| = 1.
  0-forms: omega^2 = *0^-1 d0^H *1 d0           (the u sector)
  1-forms: omega^2 = *1^-1 d1^H *2 d1           (transverse curl curl)
In continuum units c = 1 (checked below). Compares with continuous.py's step() (u sector) and the LS
curl_curl_sym symbol (symbol.py), at the same kh (h = hex distance).
"""
import itertools

import numpy as np

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

s3 = np.sqrt(3.0)
HEX = [np.array(v) * 0.5 for v in itertools.product((1, -1), repeat=3)]
SQ = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
ALL = HEX + SQ


def key(v):
    return tuple(np.round(2 * v).astype(int))


owned = [v for v in ALL if key(v) > key(-v)]
oidx = {key(v): i for i, v in enumerate(owned)}
star1 = np.array([3 / 8 if abs(abs(v[0]) - 0.5) < 1e-9 else 1 / 8 for v in owned])
star0 = 0.5
star2 = 1.0
tris = [(eh, es) for es in owned if abs(abs(es).max() - 1) < 1e-9 for eh in HEX
        if any(np.allclose(es - eh, x) for x in HEX)]
assert len(tris) == 12, len(tris)


def edge_row(x0, e, k):
    """Bloch coefficient vector of the 1-form on edge x0 -> x0 + e."""
    r = np.zeros(len(owned), complex)
    if key(e) in oidx:
        r[oidx[key(e)]] += np.exp(1j * k @ x0)
    else:
        r[oidx[key(-e)]] -= np.exp(1j * k @ (x0 + e))
    return r


def d0(k):
    return np.array([np.exp(1j * k @ e) - 1 for e in owned])[:, None]


def d1(k):
    z = np.zeros(3)
    return np.array([edge_row(z, eh, k) + edge_row(eh, es - eh, k) + edge_row(es, -es, k) for eh, es in tris])


def check_exact():
    k = np.array([0.3, -0.7, 1.1])
    return np.abs(d1(k) @ d0(k)).max()


def spectra(kvec):
    D0, D1 = d0(kvec), d1(kvec)
    w0 = float((D0.conj().T @ (star1[:, None] * D0)).real[0, 0] / star0)
    M1 = (D1.conj().T @ (star2 * D1)) / star1[:, None]
    return w0, np.sort(np.linalg.eigvals(M1).real)


print(f"d1 d0 = 0: max |d1 d0| = {check_exact():.1e};  triangles/vertex = {len(tris)}")
k_small = 1e-3
for n in ([1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 2, 3]):
    n = np.array(n, float) / np.linalg.norm(n)
    w0, w1 = spectra(k_small * n)
    print(f"continuum check n={np.round(n, 3)}: u omega^2/k^2 = {w0 / k_small**2:.6f}; "
          f"1-form omega^2/k^2 = " + " ".join(f"{v / k_small**2:.4g}" for v in w1))

print("\ndc/c / (kh)^2 at kh = 0.01 (h = hex distance = sqrt3/2 a); T = the two lowest nonzero 1-form modes")
kh = 0.01
for n in ([1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 2, 3]):
    n = np.array(n, float) / np.linalg.norm(n)
    k = kh / (s3 / 2)
    w0, w1 = spectra(k * n)
    nz = w1[w1 > 1e-6 * k**2]
    T = np.sqrt(nz[:2]) / k - 1
    print(f"  n={np.round(n, 3)}: u {(np.sqrt(w0) / k - 1) / kh**2:+.4f}   DEC T {T[0] / kh**2:+.4f} {T[1] / kh**2:+.4f}"
          f"   zero modes {int((w1 <= 1e-6 * k**2).sum())}")

print("\nfull 1-form spectrum omega/(c k) at kh = 0.196 and omega h / c (grid frequencies of the extra modes):")
kh = 0.196
for n in ([1, 0, 0], [1, 1, 1], [1, 1, 0]):
    n = np.array(n, float) / np.linalg.norm(n)
    k = kh / (s3 / 2)
    _, w1 = spectra(k * n)
    w = np.sqrt(np.clip(w1, 0, None))
    print(f"  n={np.round(n, 3)}: omega/ck = " + " ".join(f"{v / k:.4f}" for v in w) +
          "   omega h/c = " + " ".join(f"{v * s3 / 2:.3f}" for v in w))

print("\nbranch at the zone edge along x (kh = pi/ sqrt3 * sqrt3/2 ... sampled): omega h / c")
for kh in (0.5, 1.0, 1.5, 2.0, 2.5):
    k = kh / (s3 / 2)
    _, w1 = spectra(k * np.array([1.0, 0, 0]))
    print(f"  kh={kh}: " + " ".join(f"{v:.3f}" for v in np.sqrt(np.clip(w1, 0, None)) * s3 / 2))

print("\nsmooth transverse plane wave sampled as edge integrals a_e = A(mid).e: fraction in the 4 gapped modes")
for kh in (0.785, 0.393, 0.196):
    for n in ([1, 0, 0], [1, 1, 1], [1, 1, 0]):
        n = np.array(n, float) / np.linalg.norm(n)
        ref = np.eye(3)[int(np.argmin(np.abs(n)))]
        p = ref - (ref @ n) * n
        p /= np.linalg.norm(p)
        k = kh / (s3 / 2) * n
        D1 = d1(k)
        M1 = (D1.conj().T @ (star2 * D1)) / star1[:, None]
        ev, V = np.linalg.eig(M1)
        samp = np.array([np.exp(0.5j * k @ e) * (p @ e) for e in owned])
        cv = np.linalg.solve(V, samp)
        gap = ev.real > 10 * (k @ k)
        frac = np.linalg.norm(V[:, gap] @ cv[gap]) / np.linalg.norm(samp)
        print(f"  kh={kh:.3f} n={np.round(n, 3)}: {frac:.2e}")
