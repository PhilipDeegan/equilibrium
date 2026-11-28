import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from scipy.sparse.linalg import LinearOperator, lobpcg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "27/bohm_free"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hydrogen_swarm"))
from bohm_free import laplacian  # noqa: E402
from hydrogen_swarm import CORNERS, VOL, voronoi_inv_r  # noqa: E402

WORKERS = 4


class Lattice:
    def __init__(self, n):
        self.n = n
        self.c = np.full(3, n // 2, float)
        g = np.indices((n, n, n)).astype(float)
        self.pos = [g, g + 0.5]
        self.z = [self.pos[s][2] - self.c[2] for s in (0, 1)]
        self.r = [np.sqrt(((self.pos[s] - self.c[:, None, None, None]) ** 2).sum(0)) for s in (0, 1)]
        inv0 = voronoi_inv_r()
        m = 2 * n
        d = np.indices((m, m, m)).astype(float)
        d = np.where(d >= n, d - m, d)
        self.kfft = {}
        for s in (0, 1):
            for t in (0, 1):
                dd = d + 0.5 * (t - s)
                r = np.sqrt((dd**2).sum(0))
                k = np.where(r > 0, 1 / np.maximum(r, 1e-12), inv0)
                self.kfft[s, t] = sfft.rfftn(k, workers=WORKERS)

    def coulomb(self, rho):
        n, m = self.n, 2 * self.n
        rf = [sfft.rfftn(rho[t], s=(m, m, m), workers=WORKERS) for t in (0, 1)]
        return [VOL * sfft.irfftn(sum(self.kfft[s, t] * rf[t] for t in (0, 1)), s=(m, m, m), workers=WORKERS)[:n, :n, :n] for s in (0, 1)]


def split(v, n):
    return [v[: n**3].reshape(n, n, n), v[n**3 :].reshape(n, n, n)]


def ground(lat, V, mass, x0=None, tol=1e-9, maxiter=4000):
    n = lat.n

    def H(v):
        cols = np.asarray(v).reshape(2 * n**3, -1)
        out = np.empty_like(cols)
        for j in range(cols.shape[1]):
            phi = split(cols[:, j], n)
            L = laplacian(phi)
            out[:, j] = np.concatenate([(-0.5 / mass * L[s] + V[s] * phi[s]).ravel() for s in (0, 1)])
        return out

    op = LinearOperator((2 * n**3, 2 * n**3), matvec=H, matmat=H, dtype=float)
    if x0 is None:
        x0 = np.concatenate([np.exp(-lat.r[s] / 4).ravel() for s in (0, 1)])
    E, X = lobpcg(op, x0[:, None], largest=False, tol=tol, maxiter=maxiter)
    phi = split(X[:, 0], n)
    norm = VOL * sum((phi[s] ** 2).sum() for s in (0, 1))
    return float(E[0]), [phi[s] ** 2 / norm for s in (0, 1)], X[:, 0]


def kinetic(rho, mass, eps=0.0):
    phi = [np.sqrt(rho[s] + eps) for s in (0, 1)]
    L = laplacian(phi)
    return VOL * sum((-0.5 / mass * phi[s] * L[s]).sum() for s in (0, 1)), phi, L


def zmean(lat, rho):
    return VOL * sum((rho[s] * lat.z[s]).sum() for s in (0, 1))


def hartree(lat, M, kappa, F, iters=60, mix=0.5, tol=1e-9, log=None):
    rho_p = [np.exp(-2 * lat.r[s] * 2.0) for s in (0, 1)]
    norm = VOL * sum(r.sum() for r in rho_p)
    rho_p = [r / norm for r in rho_p]
    rho_e, xe, xp, Eold = None, None, None, None
    for it in range(iters):
        Vp = lat.coulomb(rho_p)
        _, re, xe = ground(lat, [-kappa * Vp[s] + F * lat.z[s] for s in (0, 1)], 1.0, xe)
        rho_e = re if rho_e is None else [mix * re[s] + (1 - mix) * rho_e[s] for s in (0, 1)]
        Ve = lat.coulomb(rho_e)
        _, rp, xp = ground(lat, [-kappa * Ve[s] - F * lat.z[s] for s in (0, 1)], M, xp)
        rho_p = [mix * rp[s] + (1 - mix) * rho_p[s] for s in (0, 1)]
        Te, Tp = kinetic(rho_e, 1.0)[0], kinetic(rho_p, M)[0]
        Eint = -kappa * VOL * sum((rho_e[s] * lat.coulomb(rho_p)[s]).sum() for s in (0, 1))
        Ef = F * (zmean(lat, rho_e) - zmean(lat, rho_p))
        E = Te + Tp + Eint + Ef
        if log and (it % 10 == 0):
            log(f"  hartree M={M} F={F:.2e} it {it} E={E / kappa**2:.6f}")
        if Eold is not None and abs(E - Eold) < tol * abs(E):
            break
        Eold = E
    return E, rho_e, rho_p


class Pop:
    def __init__(self, x, mass):
        self.x, self.mass, self.N = x, mass, len(x)

    def stencil(self, n, s):
        g = self.x - 0.5 * s
        i0 = np.floor(g).astype(int)
        f = g - i0
        idx, w, dw = [], [], []
        for c in CORNERS:
            idx.append(np.ravel_multi_index(((i0 + c) % n).T, (n, n, n)))
            fa = [f[:, a] if c[a] else 1 - f[:, a] for a in range(3)]
            sg = [1.0 if c[a] else -1.0 for a in range(3)]
            w.append(fa[0] * fa[1] * fa[2])
            dw.append(np.stack([sg[0] * fa[1] * fa[2], sg[1] * fa[0] * fa[2], sg[2] * fa[0] * fa[1]], 1))
        return np.array(idx), np.array(w), np.array(dw)

    def deposit(self, n):
        st = [self.stencil(n, s) for s in (0, 1)]
        rho = [np.bincount(st[s][0].ravel(), weights=st[s][1].ravel() / self.N, minlength=n**3).reshape(n, n, n) for s in (0, 1)]
        return st, rho

    @staticmethod
    def push(st, mu):
        acc = 0
        for s in (0, 1):
            idx, _, dw = st[s]
            acc = acc - VOL * (dw * mu[s].ravel()[idx][..., None]).sum(0)
        return acc


def swarm_pair(lat, M, kappa, F, xe, xp, steps, dmax, beta, coulomb_every, log, tag, t0):
    n = lat.n
    pe, pp = Pop(xe.copy(), 1.0), Pop(xp.copy(), M)
    ve, vp = np.zeros_like(pe.x), np.zeros_like(pp.x)
    Phi_e = Phi_p = None
    for k in range(steps + 1):
        ste, rho_e = pe.deposit(n)
        stp, rho_p = pp.deposit(n)
        if k % coulomb_every == 0 or k == steps:
            Phi_e, Phi_p = lat.coulomb(rho_e), lat.coulomb(rho_p)
        Te, phe, Le = kinetic(rho_e, 1.0, 1.0 / pe.N)
        Tp, php, Lp = kinetic(rho_p, M, 1.0 / pp.N)
        mu_e = [-0.5 * Le[s] / phe[s] - kappa * Phi_p[s] + F * lat.z[s] for s in (0, 1)]
        mu_p = [-0.5 / M * Lp[s] / php[s] - kappa * Phi_e[s] - F * lat.z[s] for s in (0, 1)]
        ae, ap = Pop.push(ste, mu_e), Pop.push(stp, mu_p)
        if k % 250 == 0 or k == steps:
            Eint = -kappa * VOL * sum((rho_e[s] * Phi_p[s]).sum() for s in (0, 1))
            E = Te + Tp + Eint + F * (pe.x[:, 2].mean() - pp.x[:, 2].mean())
            log(f"[{tag}] step {k:5d} E={E / kappa**2:.6f} |a_e| rms={np.sqrt((ae**2).sum(1).mean()):.2e} "
                f"|a_p| rms={np.sqrt((ap**2).sum(1).mean()):.2e} z_e={pe.x[:, 2].mean() - lat.c[2]:+.4f} "
                f"z_p={pp.x[:, 2].mean() - lat.c[2]:+.4f} wall={time.time() - t0:.0f}s")
        if k == steps:
            break
        d = dmax * min(1.0, max(0.02, 2 * (1 - k / steps)))
        for pop, a, v in ((pe, ae, ve), (pp, ap, vp)):
            step = a * (d / max(np.sqrt((a**2).sum(1).mean()), 1e-30))
            nrm = np.linalg.norm(step, axis=1, keepdims=True)
            v *= beta
            v += step * np.minimum(1.0, 5 * d / np.maximum(nrm, 1e-30))
            pop.x += v
    return E, rho_e, rho_p, pe.x, pp.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--kappa", type=float, default=0.25, help="Coulomb coupling: V = -kappa/r; m_e = 1")
    ap.add_argument("--M", type=float, nargs="+", default=[1836.0, 50.0, 10.0, 1.0])
    ap.add_argument("--F", type=float, default=0.01, help="field, units of kappa^3")
    ap.add_argument("--swarm-M", type=float, default=10.0)
    ap.add_argument("--Ne", type=int, default=640000)
    ap.add_argument("--Np", type=int, default=160000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--steps-field", type=int, default=1500)
    ap.add_argument("--dmax", type=float, default=0.04)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--coulomb-every", type=int, default=5)
    ap.add_argument("--no-swarm", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    t0 = time.time()

    def log(s):
        print(s, flush=True)

    kappa = a.kappa
    lat = Lattice(a.n)
    F = a.F * kappa**3
    log(f"n={a.n} kappa={kappa} (a0 = {1 / kappa:g} cells for m=1), F={a.F} au; setup {time.time() - t0:.0f}s")
    out = {"args": np.array(str(vars(a)))}
    rows = []
    inv0 = voronoi_inv_r()
    for M in a.M:
        mu = M / (M + 1)
        V_rel = [np.where(lat.r[s] > 0, -kappa / np.maximum(lat.r[s], 1e-12), -kappa * inv0) for s in (0, 1)]
        E_rel, _, x0 = ground(lat, V_rel, mu)
        E_rel_F, rho_rel_F, _ = ground(lat, [V_rel[s] + F * lat.z[s] for s in (0, 1)], mu, x0)
        al_rel = -zmean(lat, rho_rel_F) * kappa / a.F
        EH0, re0, rp0 = hartree(lat, M, kappa, 0.0, log=None)
        EHp, rep, rpp = hartree(lat, M, kappa, F, log=None)
        EHm, rem, rpm = hartree(lat, M, kappa, -F, log=None)
        al_H = ((zmean(lat, rpp) - zmean(lat, rep)) - (zmean(lat, rpm) - zmean(lat, rem))) / 2 * kappa / a.F
        sig_e = np.sqrt(VOL * sum((re0[s] * lat.r[s] ** 2).sum() for s in (0, 1)) / 3) * kappa
        sig_p = np.sqrt(VOL * sum((rp0[s] * lat.r[s] ** 2).sum() for s in (0, 1)) / 3) * kappa
        rows.append((M, mu, -mu / 2, E_rel / kappa**2, EH0 / kappa**2, 4.5 / mu**3, al_rel, al_H, sig_e, sig_p))
        log(f"M={M:7g} mu={mu:.4f}: E exact {-mu / 2:.5f} | relative on lattice {E_rel / kappa**2:.5f} | mean-field {EH0 / kappa**2:.5f} "
            f"({(EH0 - E_rel) / abs(E_rel):+.2%})   alpha exact {4.5 / mu**3:.3f} | relative {al_rel:.3f} | mean-field {al_H:.3f}   "
            f"rms width e {sig_e:.3f} p {sig_p:.3f} (a0)  wall={time.time() - t0:.0f}s")
        out[f"rhoH_e_{M:g}"] = re0[0][:, a.n // 2, :]
        out[f"rhoH_p_{M:g}"] = rp0[0][:, a.n // 2, :]
        if M == a.swarm_M:
            ref = dict(E_rel=E_rel, EH0=EH0, al_rel=al_rel, al_H=al_H, re0=re0, rp0=rp0)
    out["table"] = np.array(rows)

    if not a.no_swarm:
        M = a.swarm_M
        rng = np.random.default_rng(a.seed)
        xe = lat.c + rng.normal(0, 1.5 / kappa, (a.Ne, 3))
        xp = lat.c + rng.normal(0, 0.5 / kappa, (a.Np, 3))
        E0, rho_e, rho_p, xe, xp = swarm_pair(lat, M, kappa, 0.0, xe, xp, a.steps, a.dmax, a.beta, a.coulomb_every, log, f"swarm M={M:g} F=0", t0)
        d = {}
        for sign in (1, -1):
            res = swarm_pair(lat, M, kappa, sign * F, xe, xp, a.steps_field, a.dmax, a.beta, a.coulomb_every, log, f"swarm M={M:g} F={sign * a.F}", t0)
            d[sign] = res[4][:, 2].mean() - res[3][:, 2].mean()
            if sign == 1:
                cm_shift = (res[3][:, 2].mean() + M * res[4][:, 2].mean()) / (1 + M) - (xe[:, 2].mean() + M * xp[:, 2].mean()) / (1 + M)
        al_s = (d[1] - d[-1]) / 2 * kappa / a.F
        l1e = VOL * sum(np.abs(rho_e[s] - ref["re0"][s]).sum() for s in (0, 1))
        l1p = VOL * sum(np.abs(rho_p[s] - ref["rp0"][s]).sum() for s in (0, 1))
        log(f"[swarm M={M:g}] E={E0 / kappa**2:.5f} vs mean-field {ref['EH0'] / kappa**2:.5f} ({(E0 - ref['EH0']) / abs(ref['EH0']):+.2%}) "
            f"vs relative {ref['E_rel'] / kappa**2:.5f}; L1 e {l1e:.4f} p {l1p:.4f}; alpha {al_s:.3f} vs mean-field {ref['al_H']:.3f} "
            f"vs relative {ref['al_rel']:.3f}; CM shift under +F {cm_shift * kappa:+.5f} a0  wall={time.time() - t0:.0f}s")
        out.update(rho_e=rho_e[0][:, a.n // 2, :], rho_p=rho_p[0][:, a.n // 2, :], swarm=np.array([E0 / kappa**2, al_s, l1e, l1p, cm_shift * kappa]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
