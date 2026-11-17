"""Quasi-energy of the walk in a periodic box (the linear tick's exact invariant; the gradient E of deficit_budget.py is
not conserved). Two-tick map with parity: U(k) = S_par(k) S(k), S_par = B S(-k) B (walk.py); U is unitary with doubly
degenerate eigenvalues e^{-+i Om}, so i log U = (Om / sin Om) i (U - U^dag) / 2, cos Om = Re tr U / 4 (no eigensolve).
Per tick:  E_q = sum_k psi_k^dag H2(k) psi_k / (2 P^3)  (psi on the P^3 grid, zero off the BCC sites; numpy fft),
           E_nl = (g / 2) sum_x rho^2  (the local turn is the time-1 flow of this; first-order invariant E_q + E_nl).
Reported relative to the background: E_q - om0 sum rho, E_nl - (g / 2) rho0 sum rho.
--test: small box, seed, step and check E_q at g = 0 (exact) and E_q + E_nl at the run's g."""
import argparse
import ast
from pathlib import Path

import numpy as np

from equilibrium.substrate.walk import AS, HS, to_complex, rest_state

p = argparse.ArgumentParser()
p.add_argument("--state", action="append", default=[], help="state_t*.npz of a springbox run (pos.npy beside it)")
p.add_argument("--P", type=int, default=0)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--g", type=float, default=0.0, help="nonlinear turn (0 = take it from the run's rec.npz if given)")
p.add_argument("--rec", default="", help="rec.npz of the run (for g)")
p.add_argument("--test", action="store_true")
p.add_argument("--regions", action="store_true", help="split E_q, E_nl by deficit_budget.py regions (needs --r0)")
p.add_argument("--branch", action="store_true", help="split norm / E_q into +- quasi-energy branches, P+- = (1 +- H2 / Om) / 2, per region")
p.add_argument("--r0", type=float, default=80.0)
p.add_argument("--a", type=float, default=10.0)
p.add_argument("--xi", type=float, default=3.9)
args = p.parse_args()
m, n = args.m, np.sqrt(1 - args.m**2)
_, om0 = rest_state(m)
I2 = np.eye(2)


def symbols(kx, KY, KZ):
    """S(k), S_par(k) on a (ky, kz) slab, k in rad / grid unit (hops HS in grid units)."""
    K = np.stack([np.full_like(KY, kx), KY, KZ], -1)
    W = np.einsum("...h,hij->...ij", np.exp(1j * K @ HS.T), AS)
    Wm = np.einsum("...h,hij->...ij", np.exp(-1j * K @ HS.T), AS)
    sh = KY.shape
    S = np.zeros(sh + (4, 4), complex)
    Sp = np.zeros(sh + (4, 4), complex)
    S[..., :2, :2], S[..., 2:, 2:] = n * W, n * np.conj(np.swapaxes(W, -1, -2))
    Sp[..., :2, :2], Sp[..., 2:, 2:] = n * Wm, n * np.conj(np.swapaxes(Wm, -1, -2))
    for M in (S, Sp):
        M[..., :2, 2:] = M[..., 2:, :2] = 1j * m * I2
    Sp = Sp[..., [2, 3, 0, 1], :][..., :, [2, 3, 0, 1]]  # B S(-k) B
    return S, Sp


def quasi_energy(psi, ix, P, local=False):
    """psi (N, 4) complex on cells at grid index ix (N, 3) in [0, P). local: also the per-cell density
    e(x) = Re psi^dag (H2 psi)(x) / 2 (sums to E_q; H2 is nonlocal, so the split by region is a convention)."""
    F = np.zeros((4, P, P, P), complex)
    for c in range(4):
        F[c][ix[:, 0], ix[:, 1], ix[:, 2]] = psi[:, c]
    F = np.fft.fftn(F, axes=(1, 2, 3))
    k = 2 * np.pi * np.fft.fftfreq(P)
    KY, KZ = np.meshgrid(k, k, indexing="ij")
    E, om_max = 0.0, 0.0
    for i, kx in enumerate(k):
        S, Sp = symbols(kx, KY, KZ)
        U = Sp @ S
        cO = np.clip(np.real(np.trace(U, axis1=-2, axis2=-1)) / 4, -1, 1)
        Om = np.arccos(cO)
        om_max = max(om_max, Om.max())
        H = (Om / np.sin(Om))[..., None, None] * 0.5j * (U - np.conj(np.swapaxes(U, -1, -2)))
        f = F[:, i].transpose(1, 2, 0)
        Hf = np.einsum("...ij,...j->...i", H, f)
        E += np.real(np.sum(f.conj() * Hf))
        if local:
            F[:, i] = Hf.transpose(2, 0, 1)
    if not local:
        return E / (2 * P**3), om_max
    F = np.fft.ifftn(F, axes=(1, 2, 3))
    e = 0.5 * np.real(sum(np.conj(psi[:, c]) * F[c][ix[:, 0], ix[:, 1], ix[:, 2]] for c in range(4)))
    return E / (2 * P**3), om_max, e


def branch_split(psi, ix, P, want_psi=False):
    """psi_- = P_- psi per cell (P_- = (1 - H2 / Om) / 2 per k; H2 / Om = i (U - U^dag) / (2 sin Om), 50 / 50 where
    sin Om ~ 0); returns |psi_-|^2 per cell, total norm_-, E_- = -sum Om |psi_-k|^2 / (2 P^3) and the norm at Om ~ pi."""
    F = np.zeros((4, P, P, P), complex)
    for c in range(4):
        F[c][ix[:, 0], ix[:, 1], ix[:, 2]] = psi[:, c]
    F = np.fft.fftn(F, axes=(1, 2, 3))
    k = 2 * np.pi * np.fft.fftfreq(P)
    KY, KZ = np.meshgrid(k, k, indexing="ij")
    Nm, Em, Ndeg = 0.0, 0.0, 0.0
    for i, kx in enumerate(k):
        S, Sp = symbols(kx, KY, KZ)
        U = Sp @ S
        cO = np.clip(np.real(np.trace(U, axis1=-2, axis2=-1)) / 4, -1, 1)
        Om = np.arccos(cO)
        sO = np.sin(Om)
        deg = sO < 1e-9
        Hn = np.where(deg, 0, 1 / np.where(deg, 1, sO))[..., None, None] * 0.5j * (U - np.conj(np.swapaxes(U, -1, -2)))
        f = F[:, i].transpose(1, 2, 0)
        fm = 0.5 * (f - np.einsum("...ij,...j->...i", Hn, f))
        nm = np.sum(np.abs(fm)**2, -1)
        Nm += nm.sum()
        Em -= np.sum(Om * nm)
        Ndeg += np.sum(np.abs(f[deg])**2)
        F[:, i] = fm.transpose(2, 0, 1)
    F = np.fft.ifftn(F, axes=(1, 2, 3))
    if want_psi:
        return np.stack([F[c][ix[:, 0], ix[:, 1], ix[:, 2]] for c in range(4)], 1)
    rm = sum(np.abs(F[c][ix[:, 0], ix[:, 1], ix[:, 2]])**2 for c in range(4))
    return rm, Nm / P**3, Em / (2 * P**3), Ndeg / P**3


def regions(pos, rho, P):
    side = P / np.sqrt(3)
    rho_c = np.hypot(pos[:, 0], pos[:, 1])
    cc = (rho < 0.25) & (rho_c > 0.5 * args.r0)
    wt = (1 - rho)[cc]
    kz = 2 * np.pi / side
    rc, zc = np.sum(wt * rho_c[cc]) / wt.sum(), np.angle(np.sum(wt * np.exp(1j * kz * pos[cc, 2]))) / kz
    dz = (pos[:, 2] - zc + side / 2) % side - side / 2
    dt = np.hypot(rho_c - rc, dz)
    a_, xi = args.a, args.xi
    R = {"interior": dt < a_ - 1.5 * xi, "strands": np.abs(dt - a_) < 1.5 * xi,
         "near": (dt >= a_ + 1.5 * xi) & (dt < a_ + 4 * xi), "outer": (dt >= a_ + 4 * xi) & (dt < a_ + 10 * xi)}
    tube = dt < a_ + 10 * xi
    R["axis"] = (rho_c < rc - a_ - 4 * xi) & (np.abs(dz) < a_ + 4 * xi) & ~tube
    R["floor"] = ~tube & ~R["axis"]
    return R


def report(t, L, R, ix, P, g, rho0=1.0, pos=None):
    psi = np.concatenate([to_complex(L), to_complex(R)], 1)
    rho = np.sum(L**2, 1) + np.sum(R**2, 1)
    if args.branch:
        rm, Nm, Em, Ndeg = branch_split(psi, ix, P)
        print(f"t {t:6d} branch-: norm {Nm:.6e} (check {rm.sum():.6e})  E_- {Em:+.4f}  norm at Om~pi {Ndeg:.3e}", flush=True)
        if pos is not None:
            Rg = regions(pos, rho, P)
            s_ = "  ".join(f"{k} {rm[m_].sum():.4e}/{m_.sum()}" for k, m_ in Rg.items())
            print(f"t {t:6d} branch- norm / cells by region: {s_}", flush=True)
        return
    if pos is not None:
        Eq, om_max, e = quasi_energy(psi, ix, P, local=True)
        eq = e - om0 * rho
        enl = 0.5 * g * rho * (rho - rho0)
        Rg = regions(pos, rho, P)
        s_ = "  ".join(f"{k} {eq[m_].sum():+8.2f}/{enl[m_].sum():+8.2f}" for k, m_ in Rg.items())
        st = Rg["interior"] | Rg["strands"] | Rg["near"]
        print(f"t {t:6d} regions E_q/E_nl: {s_}", flush=True)
        print(f"t {t:6d} structure E_q {eq[st].sum():+.2f} E_nl {enl[st].sum():+.2f} sum {eq[st].sum() + enl[st].sum():+.2f}"
              f" | rest sum {eq[~st].sum() + enl[~st].sum():+.2f}", flush=True)
    else:
        Eq, om_max = quasi_energy(psi, ix, P)
    Nn = rho.sum()
    Enl = 0.5 * g * np.sum(rho**2)
    dq, dnl = Eq - om0 * Nn, Enl - 0.5 * g * rho0 * Nn
    print(f"t {t:6d}: norm {Nn:.6f}  E_q-bg {dq:+.6f}  E_nl-bg {dnl:+.6f}  sum {dq + dnl:+.6f}  (Om max {om_max:.3f})",
          flush=True)
    return dq, dnl


if args.test:
    from equilibrium.substrate.walk import CppWalk, cpp_box

    P = 48
    alat = cpp_box(P, n_threads=4)
    pos = np.array(alat[0].leaf_pos)
    ix = (np.rint(pos * np.sqrt(3)).astype(np.int64) + P // 2) % P
    side = P / np.sqrt(3)
    rng = np.random.default_rng(0)
    ph = sum(rng.normal() * np.sin(2 * np.pi * (rng.integers(1, 3, 3) @ pos.T) / side + rng.random() * 6)
             for _ in range(4))
    d = np.linalg.norm(pos - pos[0], axis=1)
    chi = np.sqrt(1 - 0.6 * np.exp(-d**2 / 20)) * np.exp(1j * ph)
    if args.branch:
        w = CppWalk(alat, 0, m=m, g=0.0, parity=True)
        w.seed(np.ones(len(pos), complex))
        print("test uniform rest state (expect branch- norm ~ 0)")
        report(w.t, *w.state(), ix, P, 0.0)
        w.step(50)
        report(w.t, *w.state(), ix, P, 0.0)
    for g in (0.0, 0.0082, 0.05):
        w = CppWalk(alat, 0, m=m, g=g, parity=True)
        w.seed(chi)
        print(f"test P {P} g {g}")
        for _ in range(4):
            report(w.t, *w.state(), ix, P, g)
            w.step(50)
else:
    g = args.g
    if not g and args.rec:
        g = float(np.load(args.rec)["g"])
    P = args.P
    ix = None
    for f in map(Path, args.state):
        if ix is None:
            pos = np.load(f.parent / "pos.npy")
            ix = (np.rint(pos * np.sqrt(3)).astype(np.int64) + P // 2) % P
        ck = np.load(f)
        report(int(ck["t"]), ck["L"], ck["R"], ix, P, g, pos=pos if (args.regions or args.branch) else None)
