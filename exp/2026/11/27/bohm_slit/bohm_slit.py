import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bohm_free"))
from bohm_free import HEX_S, HEX_T, SQ, WTS, hex_vec, laplacian, neighbours, quad_basis  # noqa: E402

OFFV = np.array([[np.zeros(3)] + [hex_vec(s, k) for k in range(8)] + SQ for s in (0, 1)])
DIDX = np.array([[np.zeros(3)] + [(HEX_S if s == 0 else HEX_T)[k] for k in range(8)] + SQ for s in (0, 1)]).astype(int)
TSUB = np.array([[s] + [1 - s] * 8 + [s] * 6 for s in (0, 1)])
PINV = np.array([np.linalg.pinv(np.array([quad_basis(c) for c in OFFV[s]])) for s in (0, 1)])


def qbasis(r):
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return np.stack([np.ones_like(x), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z], 1)


class Swarm:
    def __init__(self, x, shape):
        self.shape = np.array(shape)
        a = np.round(x)
        b = np.floor(x) + 0.5
        use_a = ((x - a) ** 2).sum(1) <= ((x - b) ** 2).sum(1)
        self.s = np.where(use_a, 0, 1)
        c = np.where(use_a[:, None], a, b)
        self.idx = (np.floor(c).astype(int)) % self.shape
        self.r = x - c
        self.base = c.copy()
        self.hop = np.zeros_like(x)
        self.nhops = np.zeros((len(x), 2), int)

    def pos(self, sel=slice(None)):
        return self.base[sel] + self.hop[sel] + self.r[sel]


def locate(s, idx, r, hop, nhops, shape):
    s, idx, r, hop = s.copy(), idx.copy(), r.copy(), hop.copy()
    while True:
        offs = OFFV[s][:, 1:]
        d2 = ((r[:, None, :] - offs) ** 2).sum(2)
        k = d2.argmin(1)
        move = d2[np.arange(len(k)), k] < (r**2).sum(1)
        if not move.any():
            return s, idx, r, hop
        m = np.nonzero(move)[0]
        kk = k[m] + 1
        c = OFFV[s[m], kk]
        idx[m] = (idx[m] + DIDX[s[m], kk]) % shape
        s[m] = TSUB[s[m], kk]
        r[m] -= c
        hop[m] += c
        if nhops is not None:
            np.add.at(nhops, (m, (kk >= 9).astype(int)), 1)


def gather(F, s, idx, shape):
    t = TSUB[s]
    j = (idx[:, None, :] + DIDX[s]) % shape
    return F[t, j[..., 0], j[..., 1], j[..., 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=192)
    ap.add_argument("--ny", type=int, default=192)
    ap.add_argument("--nz", type=int, default=2)
    ap.add_argument("--kn", type=int, default=19)
    ap.add_argument("--x0", type=float, default=36.0)
    ap.add_argument("--sx", type=float, default=8.0)
    ap.add_argument("--sy", type=float, default=20.0)
    ap.add_argument("--xw", type=float, default=72.0)
    ap.add_argument("--slit-w", type=float, default=4.0)
    ap.add_argument("--slit-d", type=float, default=24.0)
    ap.add_argument("--xs", type=int, default=152)
    ap.add_argument("--sponge", type=float, default=20.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--t-end", type=float, default=300.0)
    ap.add_argument("--ens-x", type=int, default=40)
    ap.add_argument("--ens-y", type=int, default=500)
    ap.add_argument("--n-traj", type=int, default=80)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--traj-from", type=str, default="")
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--nsub-max", type=int, default=1024)
    a = ap.parse_args()

    shape = (a.nx, a.ny, a.nz)
    sh = np.array(shape)
    m, dt = a.m, a.dt
    yc = a.ny // 2
    kx = 2 * np.pi * a.kn / a.nx
    g = np.indices(shape).astype(float)
    pos = [g, g + 0.5]

    wall = []
    gam = []
    for s in (0, 1):
        x, y = pos[s][0], pos[s][1]
        in_wall = np.abs(x - a.xw) < 0.75
        in_slit = (np.abs(np.abs(y - yc) - a.slit_d / 2) < a.slit_w / 2)
        wall.append(in_wall & ~in_slit)
        ex = np.maximum(0, np.maximum(a.sponge - x, x - (a.nx - a.sponge)))
        ey = np.maximum(0, np.maximum(a.sponge - y, y - (a.ny - a.sponge)))
        gam.append(np.exp(-a.gamma * dt * ((ex / a.sponge) ** 2 + (ey / a.sponge) ** 2)))

    R, I0 = [], []
    for s in (0, 1):
        x, y = pos[s][0], pos[s][1]
        env = np.exp(-((x - a.x0) ** 2) / (4 * a.sx**2) - ((y - yc) ** 2) / (4 * a.sy**2))
        R.append(np.where(wall[s], 0, env * np.cos(kx * x)))
        I0.append(np.where(wall[s], 0, env * np.sin(kx * x)))
    norm0 = sum((R[s] ** 2 + I0[s] ** 2).sum() for s in (0, 1))
    LR = laplacian(R)
    I_half = [np.where(wall[s], 0, I0[s] + (dt / 2) * LR[s] / (2 * m)) for s in (0, 1)]

    def fields(R, I):
        G = np.zeros((2, 3, 2) + shape)
        for s in (0, 1):
            for q, F in enumerate((R, I)):
                for k, nb in enumerate(neighbours(F, s)):
                    c = OFFV[s][k + 1]
                    for ax in range(3):
                        if c[ax]:
                            G[q, ax, s] += (WTS[k] * c[ax] / (2 * m)) * nb
        P = np.stack([np.stack(R), np.stack(I)])
        J = P[0][None] * G[1] - P[1][None] * G[0]
        return {"P": P, "G": G, "J": J}, P[0] ** 2 + P[1] ** 2

    ux = (np.arange(a.ens_x) + 0.5) / a.ens_x
    uy = (np.arange(a.ens_y) + 0.5) / a.ens_y
    UX, UY = np.meshgrid(ux, uy, indexing="ij")
    xe = np.stack([a.x0 + a.sx * ndtri(UX.ravel()), yc + a.sy * ndtri(UY.ravel()), np.zeros(UX.size)], 1)
    sw = Swarm(xe, shape)
    N = len(xe)
    side0 = np.sign(xe[:, 1] - yc)
    active = np.ones(N, bool)
    hit_y = np.full(N, np.nan)
    hit_t = np.full(N, np.nan)
    crossed = np.zeros(N, bool)
    traj_ids = np.array(sorted(set(np.ravel_multi_index((np.full(a.n_traj, a.ens_x // 2), np.linspace(0, a.ens_y - 1, a.n_traj).astype(int)), (a.ens_x, a.ens_y)))))
    if a.traj_from:
        h = np.load(a.traj_from)["hit_y"]
        ok = np.nonzero(np.isfinite(h))[0]
        ok = ok[np.argsort(h[ok])]
        traj_ids = ok[np.linspace(0, len(ok) - 1, min(a.n_traj, len(ok))).astype(int)]
    traj = []
    flux = np.zeros(a.ny)
    rho_floor = 1e-10 * norm0 / np.prod(sh)
    steps = int(round(a.t_end / dt))
    J, rho = fields(R, I0)
    nclamp = 0
    pclamp = np.zeros(N, int)
    t0 = time.time()
    print(f"cells={2 * np.prod(sh)} particles={N} kx={kx:.4f} lambda={2 * np.pi / kx:.2f} "
          f"fraunhofer fringe={2 * np.pi / kx * (a.xs - a.xw) / a.slit_d:.2f} steps={steps}", flush=True)
    for step in range(steps):
        t = step * dt
        if step % 20 == 0:
            traj.append(sw.pos(traj_ids).copy())
        if step % (steps // 30) == 0:
            tot = rho.sum()
            ftot = flux.sum() * a.nz * 2 / norm0
            eta = (time.time() - t0) / max(step, 1) * (steps - step)
            print(f"t={t:6.1f} norm={tot / norm0:.4f} screen flux frac={ftot:.4f} particles hit={np.isfinite(hit_y).sum() / N:.4f} "
                  f"active={active.sum()} crossed axis={crossed.sum()} substepped={nclamp} eta={eta:5.0f}s", flush=True)
        LI = laplacian(I_half)
        Rn = [np.where(wall[s], 0, (R[s] - dt * LI[s] / (2 * m)) * gam[s]) for s in (0, 1)]
        LRn = laplacian(Rn)
        I_next = [np.where(wall[s], 0, (I_half[s] + dt * LRn[s] / (2 * m)) * gam[s]) for s in (0, 1)]
        In = [(I_half[s] + I_next[s]) / 2 for s in (0, 1)]
        Jn, rhon = fields(Rn, In)
        flux += 0.5 * (J["J"][0][0][a.xs, :, 0] + Jn["J"][0][0][a.xs, :, 0]) * dt

        ids = np.nonzero(active)[0]
        if len(ids):
            def vblend(s, idx, r, th):
                B = qbasis(r)
                P = PINV[s]

                def fit(F):
                    return np.einsum("ni,nij,nj->n", B, P, gather(F, s, idx, sh))

                def bl(f):
                    return (1 - th) * fit(f(J)) + th * fit(f(Jn))

                pr, pi = bl(lambda F: F["P"][0]), bl(lambda F: F["P"][1])
                gr = np.stack([bl(lambda F, c=c: F["G"][0, c]) for c in range(3)], 1)
                gi = np.stack([bl(lambda F, c=c: F["G"][1, c]) for c in range(3)], 1)
                rr = pr**2 + pi**2
                jr = pr[:, None] * gi - pi[:, None] * gr
                return np.where(rr[:, None] > rho_floor, jr / np.maximum(rr, rho_floor)[:, None], 0.0)

            s_, i_, r_, h_ = sw.s[ids], sw.idx[ids], sw.r[ids], sw.hop[ids]
            v0 = vblend(s_, i_, r_, 0.0)
            sp_, ip_, rp_, _ = locate(s_, i_, r_ + dt * v0, h_, None, sh)
            vp = vblend(sp_, ip_, rp_, 1.0)
            vmx = np.maximum(np.linalg.norm(v0, axis=1), np.linalg.norm(vp, axis=1))
            nsub = np.clip(np.ceil(vmx * dt / a.hmax).astype(int), 1, a.nsub_max)
            nclamp += (nsub > 1).sum()
            pclamp[ids] = np.maximum(pclamp[ids], nsub)
            xold = sw.pos(ids)
            nh = sw.nhops[ids].copy()
            for j in range(nsub.max()):
                q = np.nonzero(j < nsub)[0]
                hq = dt / nsub[q]
                th0, th1 = j / nsub[q], (j + 1) / nsub[q]
                v1 = v0[q] if j == 0 else vblend(s_[q], i_[q], r_[q], th0)
                s2, i2, r2, _ = locate(s_[q], i_[q], r_[q] + hq[:, None] * v1, h_[q], None, sh)
                v2 = vblend(s2, i2, r2, th1)
                nq = nh[q]
                s3, i3, r3, h3 = locate(s_[q], i_[q], r_[q] + hq[:, None] * (v1 + v2) / 2, h_[q], nq, sh)
                s_[q], i_[q], r_[q], h_[q], nh[q] = s3, i3, r3, h3, nq
            sw.s[ids], sw.idx[ids], sw.r[ids], sw.hop[ids], sw.nhops[ids] = s_, i_, r_, h_, nh
            xnew = sw.pos(ids)
            crossed[ids] |= np.sign(xnew[:, 1] - yc) != side0[ids]
            hit = (xold[:, 0] < a.xs) & (xnew[:, 0] >= a.xs)
            f = (a.xs - xold[hit, 0]) / (xnew[hit, 0] - xold[hit, 0])
            hit_y[ids[hit]] = xold[hit, 1] + f * (xnew[hit, 1] - xold[hit, 1])
            hit_t[ids[hit]] = t + f * dt
            gone = hit | (xnew[:, 0] < a.sponge) | (xnew[:, 1] < a.sponge) | (xnew[:, 1] > a.ny - a.sponge)
            active[ids[gone]] = False
        R, I_half, J, rho = Rn, I_next, Jn, rhon

    print(f"done t={a.t_end} screen flux frac={flux.sum() * a.nz * 2 / norm0:.4f} particles hit={np.isfinite(hit_y).sum() / N:.4f} "
          f"crossed axis={crossed.sum()} substepped={nclamp} transmitted substepped={(pclamp[np.isfinite(hit_y)] > 1).sum()} max nsub={pclamp.max()} at cap={(pclamp >= a.nsub_max).sum()} hops hex/sq={sw.nhops.sum(0)} wall={time.time() - t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, flux=flux * a.nz * 2 / norm0, hit_y=hit_y, hit_t=hit_t, x0s=xe, crossed=crossed, side0=side0,
             traj=np.array(traj), traj_ids=traj_ids, rho_A=rho[0][:, :, 0], rho_B=rho[1][:, :, 0], R0=R[0], R1=R[1],
             I0=I_half[0], I1=I_half[1], wall_A=wall[0][:, :, 0], args=np.array(str(vars(a))), yc=yc, kx=kx, nhops=sw.nhops, pclamp=pclamp)


if __name__ == "__main__":
    main()
