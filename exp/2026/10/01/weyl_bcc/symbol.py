"""Exact Bloch symbols of the two-component BCC rule (option (a)): D'Ariano-Perinotti Weyl walk (PRA 90 062106,
2014; transition matrices as in D'Ariano, Mosco, Perinotti, Tosini, Phil. Trans. A 375 20160394, arXiv:1705.08552)
and the Dirac rule built from two of them with a local coupling m (n^2 + m^2 = 1).

Rule (per cell, two quantities psi = (psi_1, psi_2), complex): psi(x, t+1) = sum_h A_h psi(x + h, t) over the 8
hex-diagonal neighbours h = (+-1, +-1, +-1) (lattice with hex distance sqrt3 here; ours has 1 -> k_ours = sqrt3 k).
Checks:
  1. unitarity of W(k) = sum_h A_h e^{i k.h} for all k (both zeta choices)
  2. W(k) = d(k) I - i sigma.a(k): eigenphases +-omega(k), cos omega = d
  3. small-k: omega = |k| / sqrt3 (c = 1/sqrt3 of a hex hop per tick), dispersion error vs direction
  4. Dirac: cos omega = n d(k) -> gap omega(0) = arccos n ~ m; compare with sqrt(m^2 + k^2/3)
  5. spin-1/2 covariance: for the 3-fold rotation about [111] (a BCC symmetry) find U in SU(2) with
     W(R k) = U W(k) U^dagger; U^3 = -I (a full 2 pi turn of the lattice symmetry gives the sign)
  6. step() comparison: the u/phi rule's dispersion error at the same |k|
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
H = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def transitions(sign):
    zeta = (1 + sign * 1j) / 4
    zc = np.conj(zeta)
    M = {
        1: (zc * np.array([[1, 0], [1, 0]]), zeta * np.array([[0, -1], [0, 1]])),
        2: (zc * np.array([[0, 1], [0, 1]]), zeta * np.array([[1, 0], [-1, 0]])),
        3: (zc * np.array([[0, -1], [0, 1]]), zeta * np.array([[1, 0], [1, 0]])),
        4: (zc * np.array([[1, 0], [-1, 0]]), zeta * np.array([[0, 1], [0, 1]])),
    }
    hs, As = [], []
    for l in range(1, 5):
        hs += [H[l - 1], -H[l - 1]]
        As += [M[l][0], M[l][1]]
    return np.array(hs), np.array(As, dtype=complex)


def W_of(k, hs, As):
    """k: (..., 3) -> (..., 2, 2)."""
    ph = np.exp(1j * np.einsum("...d,hd->...h", k, hs))
    return np.einsum("...h,hij->...ij", ph, As)


rng = np.random.default_rng(0)
K = rng.uniform(-np.pi, np.pi, size=(20000, 3))
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2)
out = []
for sign in (+1, -1):
    hs, As = transitions(sign)
    Wk = W_of(K, hs, As)
    un = np.abs(np.einsum("nij,nkj->nik", Wk, Wk.conj()) - I2).max()
    det = np.linalg.det(Wk)
    d = 0.5 * np.trace(Wk, axis1=1, axis2=2)
    a = np.stack([0.5j * np.trace(Wk @ s, axis1=1, axis2=2) for s in (sx, sy, sz)], -1)
    recon = np.abs(d[:, None, None] * I2 - 1j * np.einsum("nc,cij->nij", a, np.array([sx, sy, sz])) - Wk).max()
    out.append(f"zeta = (1 {'+' if sign > 0 else '-'} i)/4: max |W W^+ - I| = {un:.1e}; det range "
               f"[{np.abs(det).min():.6f}, {np.abs(det).max():.6f}], arg(det) spread {np.ptp(np.angle(det)):.2e}; "
               f"d real to {np.abs(d.imag).max():.1e}, a real to {np.abs(a.imag).max():.1e}, "
               f"d^2 + |a|^2 - 1 = {np.abs(d.real**2 + np.sum(a.real**2, 1) - 1).max():.1e}; recon {recon:.1e}")
print("\n".join(out))

hs, As = transitions(+1)
dirs = {"[100]": [1, 0, 0], "[110]": [1, 1, 0], "[111]": [1, 1, 1], "[123]": [1, 2, 3]}
kk = np.linspace(1e-4, np.pi, 400)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
print("\nWeyl dispersion (k in units of 1/(hex distance); omega per tick; massless continuum omega = |k|/3 here?)")
for lab, v in dirs.items():
    n = np.array(v, float) / np.linalg.norm(v)
    Wk = W_of(kk[:, None] * n, hs, As)
    d = 0.5 * np.trace(Wk, axis1=1, axis2=2).real
    om = np.arccos(np.clip(d, -1, 1))
    k_ours = kk * np.sqrt(3)  # hex distance sqrt3 -> 1
    c_eff = om[:5] / k_ours[:5]
    out_c = om / k_ours
    ax[0].plot(k_ours, om, label=lab)
    ax[1].plot(k_ours, out_c / out_c[0] - 1, label=lab)
    j = np.searchsorted(k_ours, 0.5)
    print(f"  {lab}: c = omega/k at k->0 (per hex hop per tick) {c_eff[0]:.6f}; dc/c at k_ours = 0.5: "
          f"{out_c[j] / out_c[0] - 1:+.5f} -> coefficient of (k h)^2 {(out_c[j] / out_c[0] - 1) / k_ours[j] ** 2:+.4f}")
ax[0].set_xlabel("k (1 / hex distance)")
ax[0].set_ylabel("omega (per tick)")
ax[0].set_title("Weyl walk dispersion")
ax[1].set_xlabel("k h")
ax[1].set_ylabel("c(k)/c(0) - 1")
ax[1].set_title("phase-speed error")
ax[1].set_xlim(0, 1.5)
ax[1].set_ylim(-0.3, 0.05)

print("\nDirac rule: cos omega = n d(k), n = sqrt(1 - m^2)")
for m in (0.05, 0.1, 0.3):
    nn = np.sqrt(1 - m**2)
    n_ = np.array([1, 0, 0.0])
    Wk = W_of(kk[:, None] * n_, hs, As)
    d = 0.5 * np.trace(Wk, axis1=1, axis2=2).real
    om = np.arccos(nn * d)
    k_ours = kk * np.sqrt(3)
    c0 = om[1] / k_ours[1] if False else None
    om_rel = np.sqrt(np.arcsin(m) ** 2 + (k_ours / np.sqrt(3)) ** 2)
    print(f"  m {m}: gap omega(0) {om[0]:.6f} (arcsin m {np.arcsin(m):.6f}); omega vs sqrt(m^2 + (c k)^2) at k = 0.3: "
          f"{om[np.searchsorted(k_ours, 0.3)]:.6f} vs {om_rel[np.searchsorted(k_ours, 0.3)]:.6f}")
    ax[2].plot(k_ours, om, label=f"m {m}")
    ax[2].plot(k_ours, om_rel, ls=":", color=ax[2].lines[-1].get_color())
ax[2].set_xlabel("k (1 / hex distance) along [100]")
ax[2].set_ylabel("omega")
ax[2].set_title("Dirac rule (solid) vs sqrt(m^2 + (c k)^2) (dotted)")
ax[2].set_xlim(0, 1.5)
ax[2].set_ylim(0, 0.8)
for a_ in ax[:2]:
    a_.legend()
ax[2].legend()
fig.tight_layout()
fig.savefig(here / "weyl_dispersion.png", dpi=100)

print("\nlattice symmetries: proper cube rotations R with U A_h U^+ = A_{Rh} (U unitary, solved by SVD)")
import itertools


def lift(R):
    rows = []
    for h, A in zip(hs, As):
        j = int(np.argmin(np.linalg.norm(hs - R @ h, axis=1)))
        rows.append(np.kron(I2, A.T) - np.kron(As[j], I2))
    _, sv, V = np.linalg.svd(np.vstack(rows))
    U = V[-1].conj().reshape(2, 2)
    return sv[-1], U / np.sqrt(np.linalg.det(U))


Ks = rng.uniform(-np.pi, np.pi, size=(200, 3))
W1 = W_of(Ks, hs, As)
lifts = {}
for perm in itertools.permutations(range(3)):
    for sg in itertools.product([1, -1], repeat=3):
        R = np.zeros((3, 3))
        for i, j in enumerate(perm):
            R[i, j] = sg[i]
        if np.linalg.det(R) < 0:
            continue
        smin, U = lift(R)
        if smin < 1e-10:
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            err = np.abs(W_of(Ks @ R.T, hs, As) - U @ W1 @ U.conj().T).max()
            lifts[tuple(R.astype(int).ravel())] = U
            print(f"  R = {R.astype(int).tolist()} ({ang:.0f} deg): symmetry, |W(Rk) - U W(k) U^+| = {err:.1e}, "
                  f"U = {np.round(U, 4).tolist()}")
print(f"  {len(lifts)} of 24 rotations are symmetries (identity + three 180 deg axis turns = D2, transitive on"
      f" h1..h4): no 90 or 120 deg symmetry, consistent with the [111] dispersion anisotropy")
Ux = lifts[(1, 0, 0, 0, -1, 0, 0, 0, -1)]
Uy = lifts[(-1, 0, 0, 0, 1, 0, 0, 0, -1)]
Uz = lifts[(-1, 0, 0, 0, -1, 0, 0, 0, 1)]
for a_, b_, lab in ((Ux, Uy, "x,y"), (Uy, Uz, "y,z"), (Uz, Ux, "z,x")):
    comm = np.abs(a_ @ b_ - b_ @ a_).max()
    anti = np.abs(a_ @ b_ + b_ @ a_).max()
    print(f"  lifts of the 180 deg turns about {lab}: |[U,U']| = {comm:.2e}, |{{U,U'}}| = {anti:.2e}")
print("  (rotations about x and y commute; if their lifts ANTIcommute for every choice of phases, the walk carries a"
      " projective = spinor representation of D2 (quaternion group): the phase-independent form of the 2 pi sign."
      f" U_x^2 = {np.round(Ux @ Ux, 6).tolist()} with det U = 1)")

print("\nstep() (u/phi scalar rule) for comparison: dc/c coefficients of (k h)^2 from exp/2026/09/27 dispersion:"
      " -0.018 [100], -0.031 [111]")
