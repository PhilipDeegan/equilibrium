"""Checks for whitney.py: tet count / volume, Whitney orientation (uniform curl reproduced), gauge exactness,
symmetry H(a, b) = H(b, a), and H_W vs LS helicity on the a 9 seed."""
import sys
import numpy as np
from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box
sys.path.insert(0, "exp/2026/11/09/helicity")
import whitney as Wt

C.spacing_of = lambda lat: 1.0
tm = Wt.templates()
print("templates", len(tm), tm)
B = np.array([0.3, -0.7, 0.5])
for s, p, q in tm:
    X = np.array([np.zeros(3), Wt.D[s], Wt.D[p], Wt.D[q]]) + np.array([0.2, 0.1, -0.3])
    A = lambda x: 0.5 * np.cross(B, x)
    a = {e: A(0.5 * (X[e[0]] + X[e[1]])) @ (X[e[1]] - X[e[0]]) for e in Wt.EDGES}
    da = [a[(i, j)] + a[(j, k)] - a[(i, k)] for i, j, k in Wt.FACES]
    _, vol, g = Wt.whitney_matrix(X)
    lam = Wt.QP[0]
    curl = sum(d * 2 * (lam[i] * np.cross(g[j], g[k]) + lam[j] * np.cross(g[k], g[i]) + lam[k] * np.cross(g[i], g[j]))
               for d, (i, j, k) in zip(da, Wt.FACES))
    Wa = sum(a[(i, j)] * (lam[i] * g[j] - lam[j] * g[i]) for i, j in Wt.EDGES)
    xq = lam @ X
    print(f"  tet {s,p,q} vol {vol:.4f}  W(da) {np.round(curl, 6)} (B {B})  |W(a) - A| {np.linalg.norm(Wa - A(xq)):.2e}")
P = 96
ac = C.AnchorCluster(cpp_box(P, n_threads=4))
w = Wt.Whitney(ac)
print("volume", w.vol, "box", (P / np.sqrt(3)) ** 3)
phi = np.load("/bk/dst/2026/11/08/cascade/seed_a9.npz")["phi"]
H = w.H(phi)
CH = np.array_split(np.arange(ac.N), max(1, ac.N // 65536))
rows = lambda j: phi[j]
HL = sum(np.sum(C.reconstruct_A_at(ac, i, phi[i]) * C.curl_A_at(ac, i, rows)) for i in CH)
print(f"seed a 9: H_W {H:.6e}  H_LS {HL:.6e}  ratio {H / HL:.5f}")
rng = np.random.default_rng(1)
f = rng.standard_normal(ac.N)
gphi = np.where(ac.nb >= 0, C.W[None, :] * (f[np.where(ac.nb >= 0, ac.nb, 0)] - f[:, None]), 0.0)
print(f"gauge (random f): H(phi + df) / H - 1 = {w.H(phi + gphi) / H - 1:.2e}   H(df) {w.H(gphi):.2e}")
psi = np.load("/bk/dst/2026/11/08/cascade/a9/phi_t0200.7.npz")["phi"].astype(float)
print(f"symmetry: H(phi, psi) {w.H(phi, psi):.6e}  H(psi, phi) {w.H(psi, phi):.6e}")
print(f"t 200.7: H_W / H_W(seed) {w.H(psi) / H:.4f}")
