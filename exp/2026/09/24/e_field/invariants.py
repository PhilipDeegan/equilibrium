"""Invariants of continuous.AE: noise IC, track H_T = |e|^2 + beta phi.M phi
(M = antisym(K^T K), beta = C0SQ^2 = de/dt coefficient) and total |u|^2."""
import sys
import numpy as np
from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)
from equilibrium.substrate.lattice import SubstrateLattice

lat = SubstrateLattice(radius_cells=6.0, spacing=0.5)
rng = np.random.default_rng(0)
lat.u = rng.standard_normal(lat.N) * 0.1
lat.phi = 0.5 * C._antisym(lat, C.sample_A(lat, lambda x: rng.standard_normal(x.shape)))
ae = C.AE(lat)
dt = C.stable_dt(lat) * float(sys.argv[1])
beta = C.C0SQ
m = ae.e_mask


def HT():
    Mphi = C.curl_curl_sym(lat)
    return float((ae.e**2).sum() + beta * (lat.phi * np.where(m, Mphi, 0.0)).sum())


def HT_full():
    return float((ae.e**2).sum() + beta * (lat.phi * C.curl_curl_sym(lat)).sum())


h0, hf0 = HT(), HT_full()
print(f"|u|0={np.linalg.norm(lat.u):.4f}")
for i in range(1, 3601):
    ae.step(dt)
    if i % 600 == 0:
        print(f"step {i}: H_T/H_T0={HT()/h0:.6f}  (unmasked form {HT_full()/hf0:.6f})  |u|={np.linalg.norm(lat.u):.4f}", flush=True)
print("control: plain step() on the same IC")
rng = np.random.default_rng(0)
lat.u = rng.standard_normal(lat.N) * 0.1
lat.phi = 0.5 * C._antisym(lat, C.sample_A(lat, lambda x: rng.standard_normal(x.shape)))
E0 = C.energy_density(lat).sum()
C.advance(lat, dt, 3600)
print(f"step(): E/E0={C.energy_density(lat).sum()/E0:.6f} |u|={np.linalg.norm(lat.u):.4f}")
