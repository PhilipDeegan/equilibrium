"""Exact Bloch symbol of step_ae's transverse operator (C0SQ * curl_curl_sym) on the infinite BCC lattice.

Per cell, Bloch amplitudes phi_m e^{ik.x_i} on the 14 face copies, constrained antisymmetric:
phi_{m'} = -phi_m e^{-i theta_m}, theta_m = h k.d_m, m' = mirror(m). Forward curl_K:
  A = P_w phi                        (W-weighted LS reconstruction, _DW_PINV)
  g = sum_m P_g[:, m] (e^{i theta_m} - 1) / h,   B = g x A        (LS gradient of neighbour differences)
Adjoint = conjugate transpose; antisym: out_m = x_m - e^{i theta_m} x_{m'}. phi_tt = -C0SQ^2 antisym(K^H K) phi.
Eigenvalues of the 7x7 owned-face block are omega^2. Transverse modes found by overlap with the
midpoint-sampled plane wave W_m e^{i theta_m / 2} p.d_m. Also: step()'s u sector symbol for comparison,
and the factorisation of the transverse error into g (gradient) and the reconstruction.
"""
import json
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

W, D, MIR = C.W, C.D_HAT, C.K_MIRROR
Pw, Pg = C._DW_PINV, C._D_PINV
C0SQ, C0 = C.C0SQ, C.C0
owned = [m for m in range(14) if m < MIR[m]]


def cross_mat(g):
    return np.array([[0, -g[2], g[1]], [g[2], 0, -g[0]], [-g[1], g[0], 0]])


def symbol(kvec, h):
    th = h * (D @ kvec)
    Q = np.zeros((14, 7), complex)
    for c, m in enumerate(owned):
        Q[m, c] = 1.0
        Q[MIR[m], c] = -np.exp(-1j * th[m])
    g = Pg @ (np.exp(1j * th) - 1) / h
    MK = cross_mat(g) @ Pw
    As = np.eye(14, dtype=complex)
    for m in range(14):
        As[m, MIR[m]] -= np.exp(1j * th[m])
    L = C0SQ**2 * (As @ MK.conj().T @ MK @ Q)[owned, :]
    return L, g, th


def transverse(kvec, h, pol):
    L, g, th = symbol(kvec, h)
    ev, V = np.linalg.eig(L)
    samp = (W * np.exp(0.5j * th) * (D @ pol))[owned]
    ov = np.abs(V.conj().T @ samp) / (np.linalg.norm(V, axis=0) * np.linalg.norm(samp))
    j = int(np.argmax(ov))
    return ev, ev[j].real, ov[j], g, th


def u_sector(kvec, h):
    th = h * (D @ kvec)
    return float(np.sum(W**2 * (1 - np.cos(th)))) / h**2


rows = [json.loads(l) for l in open(Path(__file__).parent / "results.jsonl")]
lam = 4.0
k = 2 * np.pi / lam
print("mode dir     kh     measured   symbol     (leapfrog incl.)  overlap")
for r in rows:
    n = np.array([float(v) for v in r["dir"].split(",")])
    n /= np.linalg.norm(n)
    h, dt = r["h"], r["dt"]
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    pol = ref - (ref @ n) * n
    pol /= np.linalg.norm(pol)
    if r["mode"] == "T":
        ev, w2, ov, g, th = transverse(k * n, h, pol)
        meas = r["E"]["dc_c"]
    else:
        w2, ov = u_sector(k * n, h), 1.0
        meas = r["u"]["dc_c"]
    pred = (2 / dt) * np.arcsin(0.5 * dt * np.sqrt(w2)) / (C0 * k) - 1
    print(f"{r['mode']}    {r['dir']:6s} {k * h:.3f}  {meas:+.5f}  {np.sqrt(w2) / (C0 * k) - 1:+.5f}  {pred:+.5f}          {ov:.4f}")

print("\n7 eigenvalues omega^2/(c k)^2 at kh=0.196 (h=0.125):")
for n in ([1, 0, 0], [1, 1, 1], [1, 1, 0]):
    n = np.array(n, float) / np.linalg.norm(n)
    ev = np.sort(np.linalg.eigvals(symbol(k * n, 0.125)[0]).real) / (C0 * k) ** 2
    print(f"  n={np.round(n, 3)}: " + " ".join(f"{v:.4g}" for v in ev))

print("\nsmall-kh coefficients of dc/c / (kh)^2 (kh=0.01), and the gradient part alone (|g x p| / k - 1):")
for n in ([1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 2, 3]):
    n = np.array(n, float) / np.linalg.norm(n)
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    for pol in (ref - (ref @ n) * n, None):
        if pol is None:
            pol = np.cross(n, ref - (ref @ n) * n)
        pol = pol / np.linalg.norm(pol)
        hh = 0.01 / k
        _, w2, ov, g, th = transverse(k * n, hh, pol)
        tot = (np.sqrt(w2) / (C0 * k) - 1) / 1e-4
        grad = (np.linalg.norm(np.cross(g, pol)) / k - 1) / 1e-4
        u = (np.sqrt(u_sector(k * n, hh)) / (C0 * k) - 1) / 1e-4
        print(f"  n={np.round(n, 3)} p={np.round(pol, 3)}: T {tot:+.4f}  gradient {grad:+.4f}  rest {tot - grad:+.4f}"
              f"   u {u:+.4f}  overlap {ov:.5f}")

print("\nsampled plane wave split into eigen-subspaces (|component| / |sample|), and the 2 transverse speeds:")
for kh in (0.785, 0.393, 0.196):
    for n in ([1, 0, 0], [1, 1, 1], [1, 1, 0]):
        n = np.array(n, float) / np.linalg.norm(n)
        ref = np.eye(3)[int(np.argmin(np.abs(n)))]
        p1 = ref - (ref @ n) * n
        p1 /= np.linalg.norm(p1)
        hh = kh / k
        L, g, th = symbol(k * n, hh)
        ev, V = np.linalg.eig(L)
        T = ev.real > 1e-8 * ev.real.max()
        out = []
        for pol in (p1, np.cross(n, p1)):
            samp = (W * np.exp(0.5j * th) * (D @ pol))[owned]
            cvec = np.linalg.solve(V, samp)
            null_part = V[:, ~T] @ cvec[~T]
            out.append(np.linalg.norm(null_part) / np.linalg.norm(samp))
        sp = np.sort(np.sqrt(ev[T].real) / (C0 * k) - 1)
        print(f"  kh={kh:.3f} n={np.round(n, 3)}: null-mode fraction {out[0]:.4f} / {out[1]:.4f}   "
              f"transverse dc/c {sp[0]:+.5f} {sp[1]:+.5f}")
