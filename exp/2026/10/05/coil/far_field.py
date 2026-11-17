"""Copper-coil test on saved spring states (Philip 10-05: continuous helical winding = electron candidate): outside the
tube, is the phase field that of ONE plain loop (the one toroidal advance), whatever the number of poloidal turns?

Phase of the k = 0 amplitude on shells around the torus centre, global phase removed; compared with q Omega / 2 of a
plain circular loop (Omega = solid angle, exact for any distance), radius rl and height zl fitted per shell.
Reported per shell: fitted loop (rl, zl), rms residual (rad) of the phase about the fitted loop field, rms of the loop
field itself, and the residual of the best plain loop with rl fixed to the core-weighted ring radius rc.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from equilibrium.substrate.walk import rest_state, to_complex

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--t", type=int, action="append", required=True)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--shells", default="0.75,0.85,0.95", help="shell radii as fractions of r_inner")
p.add_argument("--r_inner", type=float, required=True)
p.add_argument("--q", type=int, default=1)
args = p.parse_args()

d = Path(args.dir)
pos = np.load(d / "pos.npy")
v, _ = rest_state(args.m)


def loop_omega(X, rl, zl, n=256):
    """solid angle of the circle radius rl at height zl (axis z, centred on the z axis), seen from X; sign: positive
    seen from +z."""
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.stack([rl * np.cos(s), rl * np.sin(s), zl + 0 * s], 1)
    A, B = P, np.roll(P, -1, 0)
    C = np.array([0.0, 0.0, zl])
    out = np.zeros(len(X))
    for i in range(0, len(X), 4000):
        x_ = X[i : i + 4000]
        a, b, c = C[None, None] - x_[:, None], A[None] - x_[:, None], B[None] - x_[:, None]
        la, lb, lc = (np.linalg.norm(u, axis=2) for u in (a, b, c))
        num = np.einsum("nmk,nmk->nm", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("nmk,nmk->nm", a, b) * lc + np.einsum("nmk,nmk->nm", a, c) * lb
               + np.einsum("nmk,nmk->nm", b, c) * la)
        out[i : i + 4000] = 2 * np.arctan2(num, den).sum(1)
    return out


for t in args.t:
    ck = np.load(d / f"state_t{t:05d}.npz")
    L, R = ck["L"], ck["R"]
    a = np.concatenate([to_complex(L), to_complex(R)], 1) @ v.conj()
    rho = np.sum(L**2, 1) + np.sum(R**2, 1)
    del L, R, ck
    rr = np.linalg.norm(pos, axis=1)
    core = (rho < 0.25) & (rr < args.r_inner)
    wt = (1 - rho)[core]
    rc = np.sum(wt * np.hypot(*pos[core, :2].T)) / wt.sum()
    zc = np.sum(wt * pos[core, 2]) / wt.sum()
    print(f"{d.name} t {t}: core-weighted ring radius rc {rc:.1f}, zc {zc:+.1f}, core cells {core.sum()}", flush=True)
    for f in (float(s) for s in args.shells.split(",")):
        Rs = f * args.r_inner
        sh = np.abs(np.linalg.norm(pos - [0, 0, zc], axis=1) - Rs) < 0.6
        X, as_ = pos[sh], a[sh]
        ok = np.abs(as_) > 0.5
        X, as_ = X[ok], as_[ok]

        def resid(pr, q, rl_fix=None):
            rl, zl, ph = (rl_fix, pr[0], pr[1]) if rl_fix is not None else pr
            return np.angle(as_ * np.exp(-1j * (q * loop_omega(X, rl, zl) / 2 + ph)))

        ph_init = np.angle(as_.mean())
        fits = {q: least_squares(resid, [rc, zc, ph_init], args=(q,), x_scale=[10, 10, 1]) for q in (args.q, -args.q)}
        qbest = min(fits, key=lambda q: np.mean(fits[q].fun ** 2))
        best = fits[qbest]
        fx = least_squares(lambda pr: resid(pr, qbest, rl_fix=rc), [zc, ph_init], x_scale=[10, 1])
        ph0 = np.angle(as_ * np.exp(-1j * best.x[2]))
        print(f"  shell R {Rs:.1f} ({ok.sum()} cells): fitted loop q {qbest:+d} rl {best.x[0]:.1f} zl {best.x[1]:+.1f}; "
              f"rms phase {np.sqrt(np.mean(ph0**2)):.3f}, rms residual {np.sqrt(np.mean(best.fun**2)):.3f}; "
              f"rl = rc fixed: residual {np.sqrt(np.mean(fx.fun**2)):.3f}", flush=True)
