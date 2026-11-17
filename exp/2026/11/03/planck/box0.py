"""A3 step 0 (doc/dst_versus.md): does a closed linear box thermalise?

Periodic BCC box (exp/2026/10/25/defect/bcc.py, discrete step(), fixed per-cell size). Seed: random-phase u with
content only at |k| <= k0, phi 0. Diagnostics: energy per |k| shell (all fields, FFT on the cubic sublattices) and
the temporal power spectrum of u at fixed probe cells in an early and a late window. A thermalising rule moves
content to new frequencies (towards equal energy per mode); a linear fixed rule cannot move it between eigenmodes.
--rough: fixed random cell sizes s in {S-1, S, S+1} (min-cell units, S 10): still linear, scatters in k.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "10/25/defect"))
from bcc import Box  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=32)
p.add_argument("--k0", type=float, default=3.0)
p.add_argument("--dt", type=float, default=0.1)
p.add_argument("--T", type=float, default=2000.0)
p.add_argument("--Tw", type=float, default=200.0)
p.add_argument("--rough", action="store_true")
p.add_argument("--S", type=int, default=10)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--nshell", type=int, default=12)
p.add_argument("--out", default="")
args = p.parse_args()

rng = np.random.default_rng(args.seed)
n = args.n
N3 = (n, n, n)
if args.rough:
    S = args.S
    b = Box(N3, sA=S + rng.integers(-1, 2, N3), sB=S + rng.integers(-1, 2, N3), S=S)
else:
    b = Box(N3)

k1 = np.fft.fftfreq(n) * n
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
K = np.sqrt(KX**2 + KY**2 + KZ**2)
shell = np.minimum(np.round(K).astype(int), 3 * n)
nsh = shell.max() + 1


def lowk(k0):
    f = np.fft.fftn(rng.standard_normal(N3))
    f[K > k0] = 0
    return np.real(np.fft.ifftn(f))


b.uA[:] = lowk(args.k0)
b.uB[:] = lowk(args.k0)
E0 = b.energy()
b.uA /= np.sqrt(E0)
b.uB /= np.sqrt(E0)
E0 = b.energy()


def shells():
    e = np.zeros(nsh)
    for f in [b.uA, b.uB, *b.ph, *b.sqA, *b.sqB]:
        e += np.bincount(shell.ravel(), (np.abs(np.fft.fftn(f)) ** 2).ravel(), minlength=nsh)
    return e / n**3 * 0.5


nsteps = int(round(args.T / args.dt))
nw = int(round(args.Tw / args.dt))
probes = rng.integers(0, n, (64, 3))
pr = tuple(probes.T)
early, late = np.empty((nw, 64)), np.empty((nw, 64))
every = max(1, nsteps // 20)
rec_t, rec_sh, rec_E = [], [], []
t0 = time.time()
for i in range(nsteps + 1):
    if i < nw:
        early[i] = b.uA[pr]
    if i >= nsteps + 1 - nw:
        late[i - (nsteps + 1 - nw)] = b.uA[pr]
    if i % every == 0:
        e = shells()
        rec_t.append(i * args.dt)
        rec_sh.append(e)
        rec_E.append(b.energy())
        hi = e[int(np.ceil(args.k0)) + 1:].sum() / e.sum()
        el = time.time() - t0
        eta = el / max(i, 1) * (nsteps - i)
        print(f"t {i * args.dt:7.1f}  E/E0-1 {b.energy() / E0 - 1:+.2e}  share k>{args.k0 + 1:.0f}: {hi:.2e}  "
              f"shells0-{args.nshell}: " + " ".join(f"{x:.3f}" for x in e[: args.nshell] / e.sum()) +
              f"  eta {eta:.0f}s", flush=True)
    if i < nsteps:
        b.step(args.dt)


def spec(x):
    w = np.hanning(len(x))[:, None]
    P = (np.abs(np.fft.rfft((x - x.mean(0)) * w, axis=0)) ** 2).mean(1)
    return P / P.sum()


om = 2 * np.pi * np.fft.rfftfreq(nw, args.dt)
Pe, Pl = spec(early), spec(late)
ce, cl = np.cumsum(Pe), np.cumsum(Pl)
q = lambda c, f: om[np.searchsorted(c, f)]  # noqa: E731
print("omega quantiles (10/50/90/99 %):  early " + " ".join(f"{q(ce, f):.3f}" for f in (.1, .5, .9, .99)) +
      "   late " + " ".join(f"{q(cl, f):.3f}" for f in (.1, .5, .9, .99)))
print(f"spectrum L1 distance early vs late: {np.abs(Pe - Pl).sum():.3f} (0 = same, 2 = disjoint)")
mode_count = np.bincount(shell.ravel(), minlength=nsh)
eq = mode_count / mode_count.sum()
fin = rec_sh[-1] / rec_sh[-1].sum()
print("final shell share vs equipartition (mode count share) for shells 0-%d:" % args.nshell)
print("  final " + " ".join(f"{x:.3f}" for x in fin[: args.nshell]))
print("  equip " + " ".join(f"{x:.3f}" for x in eq[: args.nshell]))
if args.out:
    np.savez(args.out, t=rec_t, sh=rec_sh, E=rec_E, om=om, Pe=Pe, Pl=Pl, mode_count=mode_count, args=str(vars(args)))
print(f"done {time.time() - t0:.0f}s")
