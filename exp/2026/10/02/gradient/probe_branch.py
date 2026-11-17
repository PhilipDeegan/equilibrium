import numpy as np
src = open("gradient.py").read()
src = src[: src.index("ax_ = np.linspace")]
import sys; sys.argv = ["x"]
exec(src)
ax_ = np.linspace(-np.pi, np.pi, 12, endpoint=False) + np.pi / 12
Q = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)
lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), m))
om = -np.angle(lam)
pos = np.where(om > 0)[0]
print("k=0 eigen-phases", np.round(om, 4), "positive idx", pos)
P = V[:, pos]
P, _ = np.linalg.qr(P)


def growth_Y(Y0c, k=np.zeros(3)):
    Y0 = np.concatenate([np.stack([Y0c[:2].real, Y0c[:2].imag], -1).ravel(), np.stack([Y0c[2:].real, Y0c[2:].imag], -1).ravel()])
    Y0 /= np.linalg.norm(Y0)
    Z0 = (H_of(k, np.zeros(3)) @ Y0).real
    om_ = -np.angle(np.vdot(Y0c, symbol_dirac(k, m) @ Y0c) / np.vdot(Y0c, Y0c))
    rho0 = Z0 @ Z0
    dT = rot8(-g * rho0) @ (np.eye(8) - 2 * g * np.outer(J8 @ Z0, Z0))
    C = rot8(om_ + g * rho0) @ dT
    return max(np.abs(np.linalg.eigvals(C @ H_of(k, q))).max() - 1 for q in Q)


rng = np.random.default_rng(0)
for t in range(5):
    c = rng.normal(size=2) + 1j * rng.normal(size=2)
    Y = P @ c
    print(f"k=0 combo {t}: growth {growth_Y(Y):.2e}")
for kk in (1e-4, 1e-2, 0.05):
    k = np.array([kk, 0, 0])
    lam, V = np.linalg.eig(symbol_dirac(k, m))
    omk = -np.angle(lam)
    for i in np.where(omk > 0)[0]:
        print(f"k {kk}: branch omega {omk[i]:.5f}: growth {growth_Y(V[:, i], k):.2e}")
