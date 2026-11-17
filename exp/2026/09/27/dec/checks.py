"""DEC curl vs LS curl_curl_sym on the real lattice: exactness, divergence of the output, symmetry
(in the *1/W^2 metric), smooth-field accuracy, plane-wave dispersion, long-run stability."""
import sys
import time

import numpy as np

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)
from equilibrium.substrate.lattice import SubstrateLattice

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dec import DecAE, DecCurl  # noqa: E402

W = C.W


def lattice(R, h):
    return SubstrateLattice(radius_cells=R, spacing=h)


def interior(lat, depth):
    return np.linalg.norm(lat.pos, axis=1) < (np.linalg.norm(lat.pos, axis=1).max() - depth)


def grad_faces(lat, u):
    nb = lat.nb
    return np.where(nb >= 0, W[None, :] * (u[np.maximum(nb, 0)] - u[:, None]), 0.0)


print("== 1. curl curl of a gradient (max |cc| / max |grad|, interior)")
for h in (0.5, 0.25):
    lat = lattice(5.0, h)
    dec = DecCurl(lat)
    inn = interior(lat, 3 * h)
    for name, u in (("smooth", np.exp(-0.5 * (lat.pos**2).sum(1))), ("random", np.random.default_rng(0).standard_normal(lat.N))):
        g = grad_faces(lat, u)
        a = np.abs(dec(g)[inn]).max() / np.abs(g[inn]).max()
        b = np.abs(C.curl_curl_sym(lat, g)[inn]).max() / np.abs(g[inn]).max()
        print(f"  h={h} {name:6s}: DEC {a:.1e}   LS {b:.1e}")

print("== 2. discrete divergence of the output, sum_k W_k cc_k (step()'s div), relative to |cc|")
for h in (0.5, 0.25, 0.125):
    lat = lattice(4.0 if h < 0.2 else 5.0, h)
    dec = DecCurl(lat)
    inn = interior(lat, 3 * h)
    x = lat.pos
    g = np.exp(-0.5 * (x**2).sum(1))
    phi = C.sample_A(lat, lambda y: np.stack([y[:, 1], -y[:, 0], 0 * y[:, 0]], 1) * np.exp(-0.5 * (y**2).sum(1))[:, None])
    for name, op in (("DEC", dec), ("LS", lambda p: C.curl_curl_sym(lat, p))):
        cc = op(phi)
        dv = np.abs((cc @ W)[inn]).max() / np.abs(cc[inn]).max()
        print(f"  h={h} {name}: {dv:.2e}", end="")
    print()

print("== 3. symmetry <x, K y>_w = <K x, y>_w, w = *1 / W^2 (DEC); Euclidean (LS)")
lat = lattice(4.0, 0.5)
dec = DecCurl(lat)
rng = np.random.default_rng(1)
xa, ya = (C._antisym(lat, rng.standard_normal((lat.N, 14))) for _ in range(2))
w = (dec.star1 / W**2)[None, :]
print(f"  DEC {abs((xa * w * dec(ya)).sum() - (dec(xa) * w * ya).sum()) / abs((xa * w * dec(ya)).sum()):.1e}"
      f"   LS {abs((xa * C.curl_curl_sym(lat, ya)).sum() - (C.curl_curl_sym(lat, xa) * ya).sum()) / abs((xa * C.curl_curl_sym(lat, ya)).sum()):.1e}")

print("== 4. smooth-field accuracy: cc of sampled A vs sampled exact curl curl A (interior rel. L2)")
def A_fn(y):
    g = np.exp(-0.5 * (y**2).sum(1))
    return np.stack([y[:, 1] * g, -y[:, 0] * g, 0.3 * y[:, 0] * y[:, 2] * g], 1)


def ccA_fn(y, eps=1e-4):
    def curl(F, y):
        J = np.zeros((len(y), 3, 3))
        for d in range(3):
            dy = np.zeros(3); dy[d] = eps
            J[:, :, d] = (F(y + dy) - F(y - dy)) / (2 * eps)
        return np.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], 1)
    return curl(lambda z: curl(A_fn, z), y)


for h in (0.5, 0.25, 0.125):
    lat = lattice(4.5, h)
    dec = DecCurl(lat)
    inn = np.linalg.norm(lat.pos, axis=1) < 3.0
    phi = C.sample_A(lat, A_fn)
    ex = C.sample_A(lat, lambda y: ccA_fn(y, 1e-3))
    for name, op in (("DEC", dec), ("LS", lambda p: C.curl_curl_sym(lat, p))):
        cc = op(phi)
        err = np.linalg.norm((cc - ex)[inn]) / np.linalg.norm(ex[inn])
        print(f"  h={h} {name}: {err:.3e}", end="")
    print()
