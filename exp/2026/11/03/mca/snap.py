import argparse
import sys
from pathlib import Path

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["save", "compare"])
p.add_argument("--file", default=str(Path(__file__).parent / "snap_baseline.npz"))
p.add_argument("--steps", type=int, default=20)
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()


def state(ac, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(ac.N), C._antisym(ac, rng.standard_normal((ac.N, 14)))


def discrete_step(refine=False):
    rng = np.random.default_rng(1)
    alat = _lat.discrete.AdaptiveLattice()
    if refine:
        alat[0].info.max_level = 1
    _lat.discrete.build_bcc_sphere(alat[0], 6.0, d1=1.0)
    alat.rebuild(0)
    if refine:
        cl = alat[0]
        j = int(np.argmin((np.asarray(cl.leaf_pos) ** 2).sum(1)))
        alat.split(0, int(cl.leaf_global_indices[j]))
        assert int(np.max(alat[0].leaf_level)) == 1
    alat.set_n_threads(args.threads)
    cl = alat[0]
    N = cl.n_leaves
    cl.set_leaf_u(rng.standard_normal(N))
    cl.set_leaf_lam(rng.uniform(0.0, 2.0, N))
    cl.set_leaf_phi(np.ascontiguousarray(0.5 * rng.standard_normal((14, N))))
    for _ in range(args.steps):
        alat.step(0, 0.004)
    return np.array(cl.leaf_u), np.array(cl.leaf_phi)


def continuous(seed, setup, stepper):
    ac = C.anchor_cluster(5.0, 0.5, n_threads=args.threads)
    u, phi = state(ac, seed)
    ac.u, ac.phi = 0.3 * u, 0.3 * phi
    setup(ac, phi)
    dt = 0.3 * C.stable_dt(ac)
    for _ in range(args.steps):
        stepper(ac, dt)
    return ac.u, ac.phi


def step(ac, dt):
    ac.alat.step(ac.i, dt)


def cartan(ac, dt):
    ac.alat.step_cartan(ac.i, dt, 0.7, 3)


def lam(ac, phi):
    ac.cl.set_leaf_lam(np.random.default_rng(2).uniform(0.0, 2.0, ac.N))


def medium(ac, phi):
    fm = C.face_midpoints(ac)
    ac.set_face_medium(0.4 * np.sin(fm[..., 0]) * np.cos(0.7 * fm[..., 1]), 0.0)


def sigma(ac, phi):
    medium(ac, phi)
    ac.alat.set_cartan_sigma(True)


def hold(ac, phi):
    ac.set_cartan_hold(np.random.default_rng(20).random(ac.N))


def vcap(ac, phi):
    ac.alat.set_cartan_vcap(0.3 * np.abs(0.3 * phi / C.W).max())


def cap(ac, phi):
    S = 0.25 * np.sum(np.where(ac.nb >= 0, 0.3 * phi, 0.0) ** 2, 1)
    ac.alat.set_cell_capacity(0.3 * S.max(), 2)


def none(ac, phi):
    pass


CASES = {
    "discrete_step_lam": discrete_step,
    "discrete_step_lam_amr": lambda: discrete_step(True),
    "continuous_step_lam": lambda: continuous(11, lam, step),
    "continuous_step_medium": lambda: continuous(12, medium, step),
    "cartan_plain": lambda: continuous(13, none, cartan),
    "cartan_sigma": lambda: continuous(14, sigma, cartan),
    "cartan_hold": lambda: continuous(15, hold, cartan),
    "cartan_vcap": lambda: continuous(16, vcap, cartan),
    "cartan_cap": lambda: continuous(17, cap, cartan),
}


def ulps(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ia, ib = a.view(np.int64), b.view(np.int64)
    ia = np.where(ia < 0, np.int64(-(2**63)) - ia, ia)
    ib = np.where(ib < 0, np.int64(-(2**63)) - ib, ib)
    return np.abs(ia - ib).max()


out = {}
for name, fn in CASES.items():
    u, phi = fn()
    out[f"{name}.u"], out[f"{name}.phi"] = np.asarray(u), np.asarray(phi)
    print(f"ran {name}", flush=True)

if args.mode == "save":
    np.savez(args.file, **out)
    print(f"saved {args.file}")
    sys.exit(0)

ref = np.load(args.file)
bad = 0
for k in sorted(out):
    a, b = ref[k], out[k]
    same = np.array_equal(a, b)
    rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
    print(f"{k:28s} {'IDENTICAL' if same else 'DIFFERS'}  max_ulp={ulps(a, b):<6d} max_rel={rel:.3e}")
    bad += not same
sys.exit(1 if bad else 0)
