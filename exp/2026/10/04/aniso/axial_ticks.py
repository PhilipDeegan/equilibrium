"""genuinely axial ticks: one axis per Dirac tick, W = W_a(k_a') only (hops +-2 grid along a = square faces), mass
coupling between ticks. Cycles of X / Xp (diagonal, parity) and C_a ticks; frequency relative to the k = 0
positive branch (unwrapped: cycles longer than pi / om0 ticks)."""
import numpy as np

from axial_mix import B, X, Xp, Sc, m

o0 = np.arcsin(m)
rng = np.random.default_rng(1)
dirs = rng.normal(size=(600, 3))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
e100, e111 = np.array([1, 0, 0.]), np.ones(3) / np.sqrt(3)


def Ca(a, par=False):
    if par:
        return lambda k: B @ Sc(-np.asarray(k), (a,)) @ B
    return lambda k: Sc(k, (a,))


def seq(ticks):
    def f(k):
        U = np.eye(4, dtype=complex)
        for g in ticks:
            U = g(k) @ U
        return U
    return f, len(ticks)


def kin(U, t):
    lam = np.linalg.eigvals(U)
    d = np.angle(lam * np.exp(1j * t * o0))  # positive branch e^{-i t o0} -> near 0
    return np.abs(d).min() / t


def report(name, f, t, ks=(0.1, 0.2, 0.3)):
    out = []
    for kk in ks:
        w = np.array([kin(f(kk * d), t) for d in dirs])
        a = kin(f(kk * e100), t) - kin(f(kk * e111), t)
        out.append(f"|k| {kk}: kin/k^2 {w.mean() / kk**2:.4f} spread {np.ptp(w) / w.mean():.2e} (100-111) {a / w.mean():+.2e}")
    print(f"{name:22s} " + " | ".join(out), flush=True)


if __name__ == "__main__":
    x, y, z = (Ca(a) for a in range(3))
    xp, yp, zp = (Ca(a, True) for a in range(3))
    report("XXp", *seq([X, Xp]))
    report("xyzzyx", *seq([x, y, z, z, y, x]))
    report("xyz xp yp zp", *seq([x, y, z, xp, yp, zp]))
    report("xyz zp yp xp", *seq([x, y, z, zp, yp, xp]))
