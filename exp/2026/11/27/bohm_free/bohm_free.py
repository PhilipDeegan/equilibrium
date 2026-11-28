import argparse
import itertools
import time
from pathlib import Path

import numpy as np

W_HEX, W_SQ = 2 / 3, 1 / 3
SQ = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
HEX_S = [np.array(s) for s in itertools.product((0, -1), repeat=3)]
HEX_T = [np.array(t) for t in itertools.product((0, 1), repeat=3)]
OFFS = [s + 0.5 for s in HEX_S] + SQ
WTS = np.array([W_HEX] * 8 + [W_SQ] * 6)


def shifted(f, d):
    return np.roll(f, tuple(-int(x) for x in d), axis=(0, 1, 2))


def neighbours(F, s):
    o = F[1 - s]
    hexes = [shifted(o, h) for h in HEX_S] if s == 0 else [shifted(o, t) for t in HEX_T]
    return hexes + [shifted(F[s], d) for d in SQ]


def hex_vec(s, j):
    return HEX_S[j] + 0.5 if s == 0 else HEX_T[j] - 0.5


def laplacian(F):
    return [sum(w * (n - F[s]) for w, n in zip(WTS, neighbours(F, s))) for s in (0, 1)]


def current(R, I, m):
    out = []
    for s in (0, 1):
        nr, ni = neighbours(R, s), neighbours(I, s)
        J = np.zeros((3,) + R[s].shape)
        for k in range(14):
            c = hex_vec(s, k) if k < 8 else SQ[k - 8]
            im = R[s] * ni[k] - I[s] * nr[k]
            for a in range(3):
                J[a] += WTS[k] * c[a] * im
        out.append(J / (2 * m))
    return out


def quad_basis(x):
    x, y, z = x
    return np.array([1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z])


def quad_grad_eval(coef, r):
    return coef @ quad_basis(r)


class Lattice:
    def __init__(self, n):
        self.n = n
        self.offs = {s: [hex_vec(s, k) for k in range(8)] + SQ for s in (0, 1)}
        self.pinv = {s: np.linalg.pinv(np.array([quad_basis(np.zeros(3))] + [quad_basis(c) for c in self.offs[s]])) for s in (0, 1)}

    def centre(self, s, idx):
        return np.array(idx, float) + 0.5 * s

    def neighbour_index(self, s, idx, k):
        if k < 8:
            d = HEX_S[k] if s == 0 else HEX_T[k]
            return 1 - s, tuple((np.array(idx) + d) % self.n)
        return s, tuple((np.array(idx) + SQ[k - 8].astype(int)) % self.n)

    def gather(self, F, s, idx):
        vals = [F[s][idx]]
        for k in range(14):
            t, j = self.neighbour_index(s, idx, k)
            vals.append(F[t][j])
        return np.array(vals)

    def locate(self, s, idx, r, hop_sum, hops):
        while True:
            d = [np.dot(r - c, r - c) for c in self.offs[s]]
            k = int(np.argmin(d))
            if d[k] >= np.dot(r, r):
                return s, idx, r, hop_sum
            c = self.offs[s][k]
            hops[k >= 8] += 1
            s, idx = self.neighbour_index(s, idx, k)
            r = r - c
            hop_sum = hop_sum + c

    def nearest(self, x):
        a = np.round(x)
        b = np.floor(x) + 0.5
        if np.dot(x - a, x - a) <= np.dot(x - b, x - b):
            return 0, tuple(a.astype(int) % self.n), x - a
        return 1, tuple(np.floor(x).astype(int) % self.n), x - b


class Particle:
    def __init__(self, lat, x):
        self.s, self.idx, self.r = lat.nearest(x)
        self.base = x - self.r
        self.hop_sum = np.zeros(3)
        self.hops = [0, 0]

    def pos(self):
        return self.base + self.hop_sum + self.r


def velocity(lat, J, rho, s, idx, r):
    jv = np.array([quad_grad_eval(lat.pinv[s] @ lat.gather([J[0][a], J[1][a]], s, idx), r) for a in range(3)])
    return jv / quad_grad_eval(lat.pinv[s] @ lat.gather(rho, s, idx), r)


def fields(R, I, m):
    J = current(R, I, m)
    rho = [R[s] ** 2 + I[s] ** 2 for s in (0, 1)]
    return J, rho


def run(n, kn, sigma0, m, dt, t_end, offsets, out, ens, seed):
    lat = Lattice(n)
    k0 = 2 * np.pi * np.array(kn, float) / n
    khat = k0 / np.linalg.norm(k0)
    x0 = n / 2 - 0.22 * n * khat
    g = np.indices((n, n, n)).astype(float)
    pos = [g, g + 0.5]
    psi = []
    for s in (0, 1):
        d = [pos[s][a] - x0[a] for a in range(3)]
        env = np.exp(-(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / (4 * sigma0**2))
        ph = k0[0] * pos[s][0] + k0[1] * pos[s][1] + k0[2] * pos[s][2]
        psi.append((env * np.cos(ph), env * np.sin(ph)))
    R = [p[0] for p in psi]
    I0 = [p[1] for p in psi]
    norm0 = sum((R[s] ** 2 + I0[s] ** 2).sum() for s in (0, 1))
    LR = laplacian(R)
    I_half = [I0[s] + (dt / 2) * LR[s] / (2 * m) for s in (0, 1)]
    I_cur = I0

    cs = [np.dot(c, k0) for c in OFFS]
    vg = sum(WTS[i] * OFFS[i] * np.sin(cs[i]) for i in range(14)) / (2 * m)
    D = sum(WTS[i] * np.outer(OFFS[i], OFFS[i]) * np.cos(cs[i]) for i in range(14)) / (2 * m)
    dev, E = np.linalg.eigh(D)

    perp1 = np.cross(khat, [0.3, 0.7, 0.1])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(khat, perp1)
    frame = np.array([khat, perp1, perp2])
    parts = [Particle(lat, x0 + sigma0 * (np.array(o) @ frame)) for o in offsets]
    p0 = [p.pos().copy() for p in parts]
    xe = x0 + np.random.default_rng(seed).normal(0, sigma0, (ens, 3))
    eparts = [Particle(lat, x) for x in xe]
    rec_em, rec_es = [], []

    def ref_lat(xp, t):
        A = E @ np.diag(np.sqrt(1 + (dev * t / (2 * sigma0**2)) ** 2)) @ E.T
        return x0 + vg * t + A @ (xp - x0)

    def ref_cont(xp, t):
        return x0 + k0 * t / m + (xp - x0) * np.sqrt(1 + (t / (2 * m * sigma0**2)) ** 2)

    steps = int(round(t_end / dt))
    J, rho = fields(R, I_cur, m)
    rec_t, rec_x, rec_X, rec_S = [], [], [], []
    t0 = time.time()
    print(f"k={kn} |k0|={np.linalg.norm(k0):.4f} vg={vg} k0/m={k0 / m} D eig={dev} steps={steps}", flush=True)
    for step in range(steps + 1):
        t = step * dt
        if step % max(1, steps // 20) == 0:
            tot = sum(rho[s].sum() for s in (0, 1))
            X = sum((rho[s][None] * pos[s]).reshape(3, -1).sum(1) for s in (0, 1)) / tot
            S2 = sum((rho[s] * sum((pos[s][a] - X[a]) ** 2 for a in range(3))).sum() for s in (0, 1)) / tot
            xs = np.array([p.pos() for p in parts])
            el = np.max([np.linalg.norm(xs[i] - ref_lat(p0[i], t)) for i in range(len(parts))])
            ec = np.max([np.linalg.norm(xs[i] - ref_cont(p0[i], t)) for i in range(len(parts))])
            rec_t.append(t)
            rec_x.append(xs)
            rec_X.append(X)
            rec_S.append(S2)
            el_x = np.linalg.norm(X - (x0 + vg * t))
            ee = ""
            if eparts:
                xe_t = np.array([p.pos() for p in eparts])
                em, es = xe_t.mean(0), ((xe_t - xe_t.mean(0)) ** 2).sum(1).mean()
                rec_em.append(em)
                rec_es.append(es)
                ee = f" ens|mean-X|={np.linalg.norm(em - X):.3f} ens var/rho var={es / S2:.4f}"
            eta = (time.time() - t0) / max(step, 1) * (steps - step)
            print(
                f"t={t:6.2f} norm drift={tot / norm0 - 1:+.2e} |X-X_lat|={el_x:.3e} "
                f"max|x-ref_lat|={el:.3e} max|x-ref_cont|={ec:.3e}{ee} eta={eta:5.0f}s",
                flush=True,
            )
        if step == steps:
            break
        LI = laplacian(I_half)
        Rn = [R[s] - dt * LI[s] / (2 * m) for s in (0, 1)]
        LRn = laplacian(Rn)
        I_next = [I_half[s] + dt * LRn[s] / (2 * m) for s in (0, 1)]
        In = [(I_half[s] + I_next[s]) / 2 for s in (0, 1)]
        Jn, rhon = fields(Rn, In, m)
        for p in parts + eparts:
            v1 = velocity(lat, J, rho, p.s, p.idx, p.r)
            hs = [0, 0]
            s2, i2, r2, _ = lat.locate(p.s, p.idx, p.r + dt * v1, np.zeros(3), hs)
            v2 = velocity(lat, Jn, rhon, s2, i2, r2)
            p.s, p.idx, p.r, p.hop_sum = lat.locate(p.s, p.idx, p.r + dt * (v1 + v2) / 2, p.hop_sum, p.hops)
        R, I_half, I_cur, J, rho = Rn, I_next, In, Jn, rhon

    print("hops (hex, sq) per particle:", [tuple(p.hops) for p in parts], flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=np.array(rec_t), x=np.array(rec_x), X=np.array(rec_X), S2=np.array(rec_S),
        p0=np.array(p0), x0=x0, k0=k0, vg=vg, D=D, sigma0=sigma0, m=m, dt=dt, n=n,
        hops=np.array([p.hops for p in parts]),
        ens_mean=np.array(rec_em), ens_var=np.array(rec_es), ens_x=np.array([p.pos() for p in eparts]),
        R0=R[0], R1=R[1], I0=I_cur[0], I1=I_cur[1],
        ref_lat=np.array([[ref_lat(p0[i], t) for i in range(len(parts))] for t in rec_t]),
        ref_cont=np.array([[ref_cont(p0[i], t) for i in range(len(parts))] for t in rec_t]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, nargs=3, default=[6, 3, 2])
    ap.add_argument("--sigma0", type=float, default=4.0)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--t-end", type=float, default=40.0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    offsets = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1.5, 1, 0.5), (-0.7, -1.2, 1.1)]
    run(a.n, a.k, a.sigma0, a.m, a.dt, a.t_end, offsets, a.out, a.ens, a.seed)


if __name__ == "__main__":
    main()
