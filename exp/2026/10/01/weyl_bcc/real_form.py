"""The Weyl / Dirac BCC rule without complex numbers: each complex amplitude is a PAIR of real quantities (a, b), and
multiplication by i is the quarter turn (a, b) -> (-b, a) of that pair. zeta = (1 +- i)/4 = e^{+-i pi/4} / (2 sqrt2) is
a 45 deg turn of every pair plus a 1/(2 sqrt2) weight. So: 4 real quantities per cell (two pairs), each hop a fixed
real 4x4 (pair turn x 0/+-1 mixing pattern). Checks that the real rule reproduces the complex one exactly, and that
it commutes with the quarter turn J (the rule never needs J itself: it only appears through the fixed 45 deg turns
and, for Dirac, the mass coupling i m = a quarter turn of one half's pairs into the other half)."""

import numpy as np

src = open(__file__.replace("real_form.py", "symbol.py")).read()
src = src[: src.index("rng = np.random")].replace("Path(__file__)", "Path('.')")
ns = {}
exec(src, ns)


def real(Mc):
    """complex n x n -> real 2n x 2n acting on (a_1, b_1, a_2, b_2, ...)."""
    n = Mc.shape[0]
    R = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            z = Mc[i, j]
            R[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = [[z.real, -z.imag], [z.imag, z.real]]
    return R


def turn(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


rng = np.random.default_rng(1)
hs, As = ns["transitions"](+1)
Ar = np.array([real(A) for A in As])
J = np.kron(np.eye(2), turn(np.pi / 2))
print("hop matrices (real 4x4), each = weight 1/(2 sqrt2) x pair turn (+-45 deg) x 0/+-1 pattern:")
for h, A, R in zip(hs, As, Ar):
    pat = np.round(A / (A[np.abs(A) > 0][0] / abs(A[np.abs(A) > 0][0].real) if False else 1), 3)
    print(f"  h {h.astype(int).tolist()}: {np.round(R * 2 * np.sqrt(2), 3).tolist()}")
print(f"  max |R J - J R| over hops: {max(np.abs(R @ J - J @ R).max() for R in Ar):.1e}")
print(f"  each hop's pair-turn block is {np.degrees(np.arctan2(Ar[0][1,0], Ar[0][0,0])):.1f} deg (zeta* hops) / "
      f"{np.degrees(np.arctan2(Ar[1][1,0], Ar[1][0,0]) if abs(Ar[1][0,0])+abs(Ar[1][1,0]) > 0 else np.arctan2(Ar[1][3,2], Ar[1][2,2])):.1f} deg (zeta hops)")

# real-space step on a periodic cubic box of BCC sites (2Z^3 u 2Z^3 + 1 via a full Z^3 grid with parity)
L = 8
psi_c = rng.normal(size=(2 * L,) * 3 + (2,)) + 1j * rng.normal(size=(2 * L,) * 3 + (2,))
psi_r = np.stack([psi_c[..., 0].real, psi_c[..., 0].imag, psi_c[..., 1].real, psi_c[..., 1].imag], -1)


def step_c(psi):
    out = np.zeros_like(psi)
    for h, A in zip(hs, As):
        out += np.einsum("ij,...j->...i", A, np.roll(psi, tuple(-h.astype(int)), axis=(0, 1, 2)))
    return out


def step_r(psi):
    out = np.zeros_like(psi)
    for h, R in zip(hs, Ar):
        out += np.einsum("ij,...j->...i", R, np.roll(psi, tuple(-h.astype(int)), axis=(0, 1, 2)))
    return out


a, b = psi_c, psi_r
for _ in range(20):
    a, b = step_c(a), step_r(b)
back = np.stack([a[..., 0].real, a[..., 0].imag, a[..., 1].real, a[..., 1].imag], -1)
print(f"\n20 steps, random state: max |real rule - complex rule| = {np.abs(back - b).max():.1e}; "
      f"norm sum psi^2 {np.sum(psi_r**2):.6f} -> {np.sum(b**2):.6f} (orthogonal per k: conserved)")
m = 0.1
n = np.sqrt(1 - m * m)
print(f"Dirac mass term i m I = m x quarter turn of each pair: real form {np.round(real(1j * m * np.eye(2)), 3).tolist()}")
