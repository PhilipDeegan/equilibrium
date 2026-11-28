import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "27/bohm_free"))
from bohm_free import laplacian  # noqa: E402

VOL = 0.5
CORNERS = list(itertools.product((0, 1), repeat=3))


def voronoi_inv_r(samples=400000, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-1, 1, (samples, 3))
    nb = np.array([c for c in itertools.product((-0.5, 0.5), repeat=3)] + [v for a in range(3) for v in (np.eye(3)[a], -np.eye(3)[a])])
    inside = ((p[:, None, :] - nb) ** 2).sum(2).min(1) > (p**2).sum(1)
    return (1 / np.linalg.norm(p[inside], axis=1)).mean()


class Grid:
    def __init__(self, n, Z, F):
        self.n = n
        self.c = np.full(3, n // 2, float)
        g = np.indices((n, n, n)).astype(float)
        self.pos = [g, g + 0.5]
        inv0 = voronoi_inv_r()
        self.V0 = []
        self.z = []
        for s in (0, 1):
            d = self.pos[s] - self.c[:, None, None, None]
            r = np.sqrt((d**2).sum(0))
            self.V0.append(np.where(r > 0, -Z / np.maximum(r, 1e-12), -Z * inv0))
            self.z.append(d[2])
        self.set_field(F)

    def set_field(self, F):
        self.F = F
        self.V = [self.V0[s] + F * self.z[s] for s in (0, 1)]


def split(v, n):
    return [v[: n**3].reshape(n, n, n), v[n**3 :].reshape(n, n, n)]


def copenhagen(gr, tol=1e-9, maxiter=4000, x0=None):
    n = gr.n

    def H(v):
        v = np.asarray(v)
        cols = v.reshape(2 * n**3, -1)
        out = np.empty_like(cols)
        for j in range(cols.shape[1]):
            phi = split(cols[:, j], n)
            L = laplacian(phi)
            out[:, j] = np.concatenate([(-0.5 * L[s] + gr.V[s] * phi[s]).ravel() for s in (0, 1)])
        return out

    op = LinearOperator((2 * n**3, 2 * n**3), matvec=H, matmat=H, dtype=float)
    if x0 is None:
        x0 = np.concatenate([np.exp(-np.sqrt(((gr.pos[s] - gr.c[:, None, None, None]) ** 2).sum(0)) / 4).ravel() for s in (0, 1)])
    E, X = lobpcg(op, x0[:, None], largest=False, tol=tol, maxiter=maxiter)
    phi = split(X[:, 0], n)
    norm = VOL * sum((phi[s] ** 2).sum() for s in (0, 1))
    rho = [phi[s] ** 2 / norm for s in (0, 1)]
    return float(E[0]), rho, X[:, 0]


def moments(gr, rho):
    zbar = VOL * sum((rho[s] * gr.z[s]).sum() for s in (0, 1))
    return zbar


class Swarm:
    def __init__(self, gr, N, sigma, seed):
        self.gr, self.N = gr, N
        rng = np.random.default_rng(seed)
        self.x = gr.c + rng.normal(0, sigma, (N, 3))

    def stencil(self, s):
        n = self.gr.n
        g = self.x - 0.5 * s
        i0 = np.floor(g).astype(int)
        f = g - i0
        idx, w, dw = [], [], []
        for c in CORNERS:
            ix = (i0 + c) % n
            idx.append(np.ravel_multi_index(ix.T, (n, n, n)))
            fa = [f[:, a] if c[a] else 1 - f[:, a] for a in range(3)]
            sg = [1.0 if c[a] else -1.0 for a in range(3)]
            w.append(fa[0] * fa[1] * fa[2])
            dw.append(np.stack([sg[0] * fa[1] * fa[2], sg[1] * fa[0] * fa[2], sg[2] * fa[0] * fa[1]], 1))
        return np.array(idx), np.array(w), np.array(dw)

    def density(self, st):
        n = self.gr.n
        return [np.bincount(st[s][0].ravel(), weights=st[s][1].ravel() / self.N, minlength=n**3).reshape(n, n, n) for s in (0, 1)]

    def state(self, eps):
        st = [self.stencil(s) for s in (0, 1)]
        rho = self.density(st)
        phi = [np.sqrt(rho[s] + eps) for s in (0, 1)]
        L = laplacian(phi)
        Q = [-0.5 * L[s] / phi[s] for s in (0, 1)]
        E = VOL * sum((-0.5 * phi[s] * L[s] + self.gr.V[s] * rho[s]).sum() for s in (0, 1))
        acc = np.zeros_like(self.x)
        for s in (0, 1):
            mu = (Q[s] + self.gr.V[s]).ravel()
            idx, _, dw = st[s]
            acc -= VOL * (dw * mu[idx][..., None]).sum(0)
        return E, rho, acc

    def relax(self, steps, dmax, eps, log_every, tag, t0, beta=0.0):
        E = None
        vel = np.zeros_like(self.x)
        for k in range(steps + 1):
            E, rho, acc = self.state(eps)
            amax = np.abs(acc).max()
            if k % log_every == 0 or k == steps:
                arms = np.sqrt((acc**2).sum(1).mean())
                print(f"[{tag}] step {k:5d} E={E:.7f} |force| rms={arms:.2e} max={amax:.2e} "
                      f"zbar={self.x[:, 2].mean() - self.gr.c[2]:+.4f} wall={time.time() - t0:.0f}s", flush=True)
            if k == steps:
                break
            d = dmax * min(1.0, max(0.02, 2 * (1 - k / steps)))
            step = acc * (d / max(np.sqrt((acc**2).sum(1).mean()), 1e-30))
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            vel = beta * vel + step * np.minimum(1.0, 5 * d / np.maximum(norm, 1e-30))
            self.x += vel
        return E, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--Z", type=float, default=0.25)
    ap.add_argument("--N", type=int, default=100000)
    ap.add_argument("--sigma", type=float, default=1.5, help="initial gaussian width in a0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--steps-field", type=int, default=1000)
    ap.add_argument("--beta", type=float, default=0.0, help="momentum")
    ap.add_argument("--dmax", type=float, default=0.02, help="largest move per step, cells")
    ap.add_argument("--eps-n", type=float, default=1.0, help="density floor in units of one constituent (1/N)")
    ap.add_argument("--F", type=float, default=0.01, help="field in atomic units")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    Z = a.Z
    eu, lu, fu = Z**2, 1 / Z, Z**3
    t0 = time.time()
    gr = Grid(a.n, Z, 0.0)
    print(f"n={a.n} cells={2 * a.n**3} a0={lu:.2f} cells, box=+-{a.n / 2 / lu:.1f} a0, N={a.N}, F={a.F} au "
          f"(lattice {a.F * fu:.3e}), V(nucleus cell)={gr.V0[0][tuple(gr.c.astype(int))] / eu:.3f} au", flush=True)

    E0c, rho0c, x0 = copenhagen(gr)
    print(f"[copenhagen] F=0 E={E0c / eu:.5f} au (exact -0.5) wall={time.time() - t0:.0f}s", flush=True)
    gr.set_field(a.F * fu)
    E1c, rho1c, _ = copenhagen(gr, x0=x0)
    zc = moments(gr, rho1c) / lu
    alpha_c = -zc / a.F
    alpha_ce = -2 * (E1c - E0c) / eu / a.F**2
    print(f"[copenhagen] F={a.F} E={E1c / eu:.6f} au, <z>={zc:+.5f} a0, alpha(dipole)={alpha_c:.3f} alpha(energy)={alpha_ce:.3f} (exact 4.5) "
          f"wall={time.time() - t0:.0f}s", flush=True)

    gr.set_field(0.0)
    sw = Swarm(gr, a.N, a.sigma * lu, a.seed)
    E0s, rho0s = sw.relax(a.steps, a.dmax, a.eps_n / a.N, a.log_every, "swarm F=0", t0, a.beta)
    x_eq = sw.x.copy()
    res = {}
    for sign in (1, -1):
        sw.x = x_eq.copy()
        gr.set_field(sign * a.F * fu)
        res[sign] = sw.relax(a.steps_field, a.dmax, a.eps_n / a.N, a.log_every, f"swarm F={sign * a.F}", t0, a.beta) + (sw.x.copy(),)
    E1s, rho1s, x_field = res[1]
    zp, zm = (res[sg][2][:, 2].mean() - gr.c[2] for sg in (1, -1))
    alpha_s = -(zp - zm) / lu / (2 * a.F)
    alpha_se = -((res[1][0] + res[-1][0]) / 2 - E0s) / eu / a.F**2 * 2
    sw.x = x_field
    l1 = VOL * sum(np.abs(rho0s[s] - rho0c[s]).sum() for s in (0, 1))

    print(f"[swarm] F=0 E={E0s / eu:.5f} au vs copenhagen {E0c / eu:.5f} (diff {(E0s - E0c) / abs(E0c):+.2%}); density L1 vs copenhagen {l1:.4f}")
    print(f"[swarm] alpha(centroid)={alpha_s:.3f} alpha(energy)={alpha_se:.3f} vs copenhagen {alpha_c:.3f}")
    checks = {
        "copenhagen E within 3% of -0.5": abs(E0c / eu + 0.5) < 0.015,
        "copenhagen alpha within 5% of 4.5": abs(alpha_c - 4.5) < 0.225,
        "swarm E within 1% of copenhagen": abs(E0s - E0c) < 0.01 * abs(E0c),
        "swarm density L1 < 0.05": l1 < 0.05,
        "swarm alpha within 10% of copenhagen": abs(alpha_s - alpha_c) < 0.1 * abs(alpha_c),
    }
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print(f"wall={time.time() - t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    c = a.n // 2
    np.savez(a.out, args=np.array(str(vars(a))), x_eq=x_eq, x_field=sw.x, rho0c=rho0c[0][:, c, :], rho1c=rho1c[0][:, c, :],
             rho0s=rho0s[0][:, c, :], rho1s=rho1s[0][:, c, :], rhoms=res[-1][1][0][:, c, :], E=np.array([E0c, E1c, E0s, E1s]) / eu,
             alpha=np.array([alpha_c, alpha_ce, alpha_s, alpha_se]), l1=l1)


if __name__ == "__main__":
    main()
