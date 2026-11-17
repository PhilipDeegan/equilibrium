"""residual cubic anisotropy of the parity walk (10-03 aniso): mix in a tick on the 6 axial (square-face) second
neighbours. Axial Weyl tick: W_c(k) = W_x W_y W_z, W_a = exp(-i k_a' sigma_a), k_a' = 2 k_a / sqrt3 (hop (+-2, 0, 0) grid
= the square face), realised as a shift in the sigma_a eigenbasis (lossless, local, real form via J). Cycle = (X Xp)^p
(C Cp)^q, Cp = parity image of C (B S_c(-k) B) with the reversed product order. Rel spread = (max - min) of the upper
frequency over k directions / kinetic part."""
import numpy as np

from equilibrium.substrate import walk as W

m = 0.8
n = np.sqrt(1 - m * m)
hs, As = W.transitions(+1)


def S(k):
    Wk = np.einsum("h,hij->ij", np.exp(1j * (np.asarray(k) / np.sqrt(3)) @ hs.T), As)
    return np.block([[n * Wk, 1j * m * np.eye(2)], [1j * m * np.eye(2), n * Wk.conj().T]])


def R(axis):
    c = {"x": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "y": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], "z": [[0, -1, 0], [1, 0, 0], [0, 0, 1]]}
    return np.array(c[axis], float)


def om(U, ticks):
    return np.sort(np.abs(np.angle(np.linalg.eigvals(U))))[[0, -1]] / ticks

I2, Z2 = np.eye(2), np.zeros((2, 2))
B = np.block([[Z2, I2], [I2, Z2]])
sig = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]


def Wa(ka, a):
    return np.cos(ka) * I2 - 1j * np.sin(ka) * sig[a]


def Sc(k, order=(0, 1, 2), mc=m, scale=2 / np.sqrt(3)):
    kk = np.asarray(k) * scale
    W = I2
    for a in order:
        W = W @ Wa(kk[a], a)
    nc = np.sqrt(1 - mc * mc)
    return np.block([[nc * W, 1j * mc * I2], [1j * mc * I2, nc * W.conj().T]])


def X(k):
    return S(k)


def Xp(k):
    return B @ S(-k) @ B


def C(k, **kw):
    return Sc(k, (0, 1, 2), **kw)


def Cp(k, **kw):
    return B @ Sc(-np.asarray(k), (2, 1, 0), **kw) @ B


rng = np.random.default_rng(1)
dirs = rng.normal(size=(600, 3))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
o0 = np.arcsin(m)


def measure(f, ticks, kk):
    w = np.array([om(f(kk * d), ticks) for d in dirs])
    kin = w[:, 1].mean() - o0
    two = max(np.abs(om(f(kk * (R(a) @ d)), ticks) - om(f(kk * d), ticks)).max() for d in dirs[:80] for a in "xyz")
    return kin, np.ptp(w[:, 1]) / kin, two / kin


def cycle(p, q, **kw):
    def f(k):
        U = np.eye(4)
        for _ in range(p):
            U = Xp(k) @ X(k) @ U
        for _ in range(q):
            U = Cp(k, **kw) @ C(k, **kw) @ U
        return U
    return f, 2 * (p + q)


if __name__ == "__main__":
    print(f"m {m}  om0 {o0:.6f}")
    for name, (f, t) in {"XXp": cycle(1, 0), "CCp": cycle(0, 1)}.items():
        for kk in [0.1, 0.2, 0.3, 0.5]:
            kin, sp, two = measure(f, t, kk)
            print(f"  {name:4s} |k| {kk}: kinetic {kin:.3e} (/k^2 {kin / kk**2:.4f}) rel spread {sp:.2e} 90-deg {two:.1e}")
