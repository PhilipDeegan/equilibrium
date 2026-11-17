"""A3 with exchange: does a closed box under step() + Cartan transport drift toward equipartition?

Periodic continuous BCC box (build_bcc_box_periodic, period P), seed = smooth random u and smooth random
circulation (phi = sample_A of a low-k random A, div-free part), only |k| <= kmax. step_cartan(kappa, 3).
Spectrum: cells binned onto a cubic grid of half the cube edge (two interlaced sublattices -> one point each),
FFT of u and the 14 face values, energy per |k| shell (u^2/2 + 1/4 sum phi^2 weighting). Reference: the same
pipeline on white-noise u and phi ("equal per mode" = equipartition in the rule's own dof).
"""

import argparse
import time

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=24)
p.add_argument("--kmax", type=float, default=2.0, help="seed band, in units of 2 pi / side")
p.add_argument("--amp", type=float, default=0.3, help="rms of the seeded phi")
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--T", type=float, default=300.0)
p.add_argument("--every", type=float, default=10.0)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="")
args = p.parse_args()

t0 = time.time()
alat = _lat.continuous.AdaptiveLattice()
alat.set_weights(C.WEIGHTS_NAME)
_lat.continuous.build_bcc_box_periodic(alat[0], args.P, args.P, args.P, d1=1.0)
alat.rebuild(0)
alat.set_n_threads(args.threads)
ac = C.AnchorCluster(alat, 0)
pos = ac.pos
assert (ac.nb >= 0).all(), "periodic box should have no missing faces"
lo = pos.min(0)
dmin = np.min(np.abs(np.diff(np.unique(np.round(pos[:, 0] - lo[0], 9)))))
G = np.round((pos - lo) / dmin).astype(int)
n = G.max(0) + 1
side = n * dmin
assert len({tuple(g) for g in G}) == ac.N
rng = np.random.default_rng(args.seed)

par = G.sum(1) % 2
assert all(((G[par == q] % 2) == q).all() for q in (0, 1)), "sublattices should be all-even / all-odd on the half grid"
subs = [np.nonzero(par == q)[0] for q in (0, 1)]
subG = [G[ix] // 2 for ix in subs]
ns = tuple(int(m) for m in subG[0].max(0) + 1)
assert all(len(ix) == int(np.prod(ns)) for ix in subs)
k1 = [np.fft.fftfreq(m) * m for m in ns]
KX, KY, KZ = np.meshgrid(*k1, indexing="ij")
K = np.sqrt(KX**2 + KY**2 + KZ**2)
shell = np.round(K).astype(int)
nsh = shell.max() + 1
k1f = [np.fft.fftfreq(m) * m for m in n]
KF = np.sqrt(sum(x**2 for x in np.meshgrid(*k1f, indexing="ij")))


def smooth_field(shape_extra=()):
    """random field on the cells with only |k| <= kmax (k in units of 2 pi / side)."""
    out = []
    for _ in range(int(np.prod(shape_extra)) if shape_extra else 1):
        f = np.fft.fftn(rng.standard_normal(tuple(n)))
        f[KF > args.kmax] = 0
        g = np.real(np.fft.ifftn(f))
        out.append(g[tuple(G.T)])
    return np.stack(out, -1) if shape_extra else out[0]


def spectrum(u, phi):
    e = np.zeros(nsh)
    grid = np.zeros(ns)
    for ix, g in zip(subs, subG):
        for f, w in [(u, 0.5)] + [(phi[:, k], 0.25) for k in range(14)]:
            grid[:] = 0
            grid[tuple(g.T)] = f[ix]
            e += w * np.bincount(shell.ravel(), (np.abs(np.fft.fftn(grid)) ** 2).ravel(), minlength=nsh)
    return e / grid.size


A = smooth_field((3,))
mid = C.face_midpoints(ac).reshape(-1, 3)
# A at face midpoints: trilinear-free approximation, A of the cell (smooth on the cell scale)
A_faces = np.repeat(A, 14, axis=0)
phi = C.W[None, :] * np.einsum("nkd,kd->nk", A_faces.reshape(ac.N, 14, 3), C.D_HAT)
phi = C._antisym(ac, phi)
phi = C.project_div_free(ac, phi)
phi *= args.amp / np.sqrt(np.mean(phi**2))
u = smooth_field()
u *= args.amp / np.sqrt(np.mean(u**2))
ac.u, ac.phi = u, phi

ref = spectrum(rng.standard_normal(ac.N), C._antisym(ac, rng.standard_normal((ac.N, 14))))
ref /= ref.sum()


def energy():
    return float(np.sum(C.energy_density(ac)))


dt = C.stable_dt(ac)
nsteps = int(round(args.T / dt))
every = max(1, int(round(args.every / dt)))
E0 = energy()
kseed = int(np.ceil(args.kmax))
print(f"N {ac.N} grid {tuple(n)} side {side[0]:.2f} dt {dt:.4f} steps {nsteps} E0 {E0:.4e} setup {time.time() - t0:.1f}s",
      flush=True)
print("equip ref shell share: " + " ".join(f"{x:.3f}" for x in ref[:14]))
rec = []
t1 = time.time()
for i in range(nsteps + 1):
    if i % every == 0:
        s = spectrum(ac.u, ac.phi)
        sh = s / s.sum()
        hi = sh[kseed + 1:].sum()
        top = sh[nsh // 2:].sum()
        el = time.time() - t1
        print(f"t {i * dt:7.1f} E/E0-1 {energy() / E0 - 1:+.2e}  above seed {hi:.3f} (equip {ref[kseed + 1:].sum():.3f})  "
              f"top half k {top:.3f} (equip {ref[nsh // 2:].sum():.3f})  shells " +
              " ".join(f"{x:.3f}" for x in sh[:10]) + f"  eta {el / max(i, 1) * (nsteps - i):.0f}s", flush=True)
        rec.append((i * dt, s))
    if i < nsteps:
        alat.step_cartan(0, dt, args.kappa, 3)
if args.out:
    np.savez(args.out, t=[r[0] for r in rec], s=np.array([r[1] for r in rec]), ref=ref, args=str(vars(args)))
print(f"done {time.time() - t0:.0f}s")
