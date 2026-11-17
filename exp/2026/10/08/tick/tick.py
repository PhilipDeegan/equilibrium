"""10-08 tick: step() with phi derived from a per-cell periodic quantity (tick phase theta, period 2 pi).
Face offset phi_ik = W_k wrap(theta_j - theta_i) (wrap into (-pi, pi]); same update as step() otherwise:
    u  -= dt sum_k W_k phi_ik          (u_t = -C0SQ div A)
    theta -= dt (u + kappa q),  q_i = sum_k W_k^2 wrap(theta_j - theta_i)^2 / (2 sum W^2)
(phi_t = -grad u  <->  theta_t = -u; q: optional -- a cell's clock shifted by how far out of step its junctions are,
kappa = 0 is plain step()). Loop sums of the wrapped offsets around any closed loop are whole multiples of 2 pi by
construction; defects = elementary triangles (hex a, hex b, square b - a) with a nonzero count.
Modes: --test pulse  (matched Gaussian u bump vs C++ step(), theta 0: must agree to round-off while no wrap engages)
       --test ring   (theta = q Omega / 2 of a ring over the 27 images minus the uniform term (pair_box.py); track the
                      defect triangles: count, centroid rho / z, winding sum)"""
import argparse

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box, solid_angle

p = argparse.ArgumentParser()
p.add_argument("--test", default="ring")
p.add_argument("--P", type=int, default=120)
p.add_argument("--R0", type=float, default=15.0)
p.add_argument("--kappa", type=float, default=0.0)
p.add_argument("--T", type=float, default=20.0)
p.add_argument("--every", type=float, default=2.0)
p.add_argument("--dt_scale", type=float, default=1.0)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
side = args.P / np.sqrt(3)
W, D = C.W, C.D_HAT
W2 = W**2
dt = args.dt_scale * 0.4 / float(np.sum(W2))
TWO_PI = 2 * np.pi


def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


def offsets(th):
    return wrap(th[nb] - th[:, None])


def tick(u, th):
    u = u - dt * (offsets(th) * W2[None, :]).sum(1)
    q = (offsets(th)**2 * W2[None, :]).sum(1) / (2 * W2.sum()) if args.kappa else 0.0
    th = (th - dt * (u + args.kappa * q)) % TWO_PI
    return u, th


# elementary triangles: hex a, hex b with b - a a square offset
hexk = list(range(8))
sqk = list(range(8, 14))
tri = []
for ka in hexk:
    for kb in hexk:
        dd = D[kb] - D[ka]
        for ks in sqk:
            if np.allclose(dd, D[ks]):
                tri.append((ka, kb, ks))
i0 = 0
for ka, kb, ks in tri:
    assert nb[nb[i0, ka], ks] == nb[i0, kb]


def defects(th):
    """per triangle type: integer winding n (N,) of the loop i -> i+a -> i+b -> i; returns (positions, n) of n != 0."""
    P_, Nn = [], []
    for ka, kb, ks in tri:
        j, l = nb[:, ka], nb[:, kb]
        s = wrap(th[j] - th) + wrap(th[l] - th[j]) + wrap(th - th[l])
        n = np.rint(s / TWO_PI).astype(int)
        m = n != 0
        if m.any():
            c = pos[m] + (D[ka] + D[kb]) / 3
            P_.append(c)
            Nn.append(n[m])
    if not P_:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.concatenate(P_), np.concatenate(Nn)


def wrapx(x):
    return (x + side / 2) % side - side / 2


nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
print(f"{args.test}: P {args.P} N {N} dt {dt:.4f} ticks {nst} kappa {args.kappa} triangles/cell {len(tri)}", flush=True)
if args.test == "pulse":
    G = np.exp(-np.sum(wrapx(pos)**2, 1) / (2 * 4.0**2))
    eps = 1e-3
    u, th = eps * G, np.zeros(N)
    ac.u, ac.phi = eps * G, np.zeros((N, 14))
    for s in range(nst):
        u, th = tick(u, th)
        alat.step(0, dt)
        if (s + 1) % ev == 0:
            us = ac.u
            dphi = np.abs(W[None, :] * offsets(th) - ac.phi).max()
            print(f"  t {(s + 1) * dt:6.2f}: max |u - u_step| / max |u| {np.abs(u - us).max() / np.abs(us).max():.2e}; "
                  f"max |phi - phi_step| {dphi:.2e}", flush=True)
else:
    s_ = np.linspace(0, TWO_PI, 400, endpoint=False)
    ring = np.stack([args.R0 * np.cos(s_), args.R0 * np.sin(s_), 0 * s_], 1)
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) * side

    def phase(X):
        return 0.5 * solid_angle(ring, X, shifts=shifts, n_threads=args.threads)

    rng = np.random.default_rng(0)
    jumps = []
    for ax in range(3):
        smp = (rng.random((400, 3)) - 0.5) * side
        smp[:, ax] = -side / 2
        hi = smp.copy()
        hi[:, ax] += side
        jumps.append(np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp))))))
    th = (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3))) % TWO_PI
    u = np.zeros(N)
    x = wrapx(pos)
    for s in range(nst + 1):
        if s % ev == 0:
            Pd, n = defects(th)
            if len(n):
                xd = wrapx(Pd)
                rr = np.hypot(xd[:, 0], xd[:, 1])
                kz = TWO_PI / side
                zc = np.angle(np.mean(np.exp(1j * kz * xd[:, 2]))) / kz
                d_ = f"defect triangles {len(n)} (sum n {n.sum():+d}, |n|>1: {(np.abs(n) > 1).sum()}), rho {rr.mean():6.2f} +- {rr.std():4.2f}, z {zc:+7.3f}"
            else:
                d_ = "no defects"
            E = 0.5 * np.sum(u**2) + 0.25 * np.sum((W[None, :] * offsets(th))**2)
            print(f"  t {s * dt:6.2f}: {d_}; max |u| {np.abs(u).max():.3e}; E {E:.3f}", flush=True)
        if s < nst:
            u, th = tick(u, th)
