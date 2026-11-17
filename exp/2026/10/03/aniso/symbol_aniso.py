"""2-fold anisotropy of the walk symbol about z: frequencies of the Dirac walk symbol at k in the xy plane vs angle,
for sign +1, sign -1, and the two-tick product (+1 then -1). Also checks whether a 90 deg turn about z maps +1 to -1."""
import numpy as np
from equilibrium.substrate import walk as W

m = 0.8
n = np.sqrt(1 - m * m)


def sym(k, sign):
    hs, As = W.transitions(sign)
    ph = np.exp(1j * (np.asarray(k) / np.sqrt(3)) @ hs.T)
    Wk = np.einsum("h,hij->ij", ph, As)
    I = np.eye(2)
    return np.block([[n * Wk, 1j * m * I], [1j * m * I, n * Wk.conj().T]])


def freqs(U, ticks=1):
    om = np.sort(np.abs(np.angle(np.linalg.eigvals(U)))) / ticks
    return om


ths = np.radians(np.arange(0, 181, 15))
for k in [0.3, 0.6, 1.0]:
    print(f"|k| {k}  (theta from x in the xy plane; columns: om_upper sign+, sign-, alternating(+,-)/2)")
    for th in ths:
        kv = k * np.array([np.cos(th), np.sin(th), 0.0])
        a, b = freqs(sym(kv, +1)), freqs(sym(kv, -1))
        c = freqs(sym(kv, -1) @ sym(kv, +1), 2)
        print(f"  th {np.degrees(th):5.0f}: {a[-1]:.6f} {b[-1]:.6f} {c[-1]:.6f}   lower {a[0]:.6f} {b[0]:.6f} {c[0]:.6f}")
# 90 deg about z: is sym(+1, Rk) unitarily equivalent to sym(-1, k)?  compare spectra over random k
rng = np.random.default_rng(0)
Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
d_same = d_flip = 0
for _ in range(200):
    k = rng.normal(size=3)
    s0 = np.sort(np.angle(np.linalg.eigvals(sym(k, +1))))
    d_same = max(d_same, np.abs(np.sort(np.angle(np.linalg.eigvals(sym(Rz @ k, +1)))) - s0).max())
    d_flip = max(d_flip, np.abs(np.sort(np.angle(np.linalg.eigvals(sym(Rz @ k, -1)))) - s0).max())
print(f"spectrum(+1, Rz k) vs (+1, k): max diff {d_same:.2e};  spectrum(-1, Rz k) vs (+1, k): {d_flip:.2e}")
