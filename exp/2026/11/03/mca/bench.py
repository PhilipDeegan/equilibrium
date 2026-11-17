import argparse
import time

import numpy as np

from equilibrium.substrate import continuous as C

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=16.0)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--only", default="")
args = p.parse_args()

ac = C.anchor_cluster(args.R, 0.5, n_threads=args.threads)
rng = np.random.default_rng(3)
ac.u = 0.3 * rng.standard_normal(ac.N)
ac.phi = 0.3 * C._antisym(ac, rng.standard_normal((ac.N, 14)))
dt = 0.3 * C.stable_dt(ac)
lam = rng.uniform(0.0, 2.0, ac.N)


def best(fn, n):
    fn()
    t = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t.append((time.perf_counter() - t0) / n)
    return min(t)


res = {}
if args.only in ("", "step"):
    res["step"] = best(lambda: ac.alat.step(ac.i, dt), 50)
if args.only in ("", "step_cartan"):
    res["step_cartan"] = best(lambda: ac.alat.step_cartan(ac.i, dt, 1.0, 2), 10)
if not args.only:
    ac.cl.set_leaf_lam(lam)
    res["step_lam"] = best(lambda: ac.alat.step(ac.i, dt), 50)
    res["step_cartan_lam"] = best(lambda: ac.alat.step_cartan(ac.i, dt, 1.0, 2), 10)
print(f"N={ac.N} threads={args.threads}")
for k, v in res.items():
    print(f"{k:18s} {v * 1e3:8.3f} ms   {v / ac.N * 1e9:7.2f} ns/cell")
