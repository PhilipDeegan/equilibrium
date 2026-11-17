"""Poynting balance for the EXPERIMENTAL A-E extension (continuous.AE).
IC: u=0, A=0, E = curl(z G), G gaussian (div E = 0). Energy inside r<Rs,
U = sum 1/2 (|E|^2 + c^2 |B|^2) V, vs time-integrated outward flux of
S = c^2 E x B through the shell Rs (shell-averaged surface integral)."""
import sys, time
import numpy as np
from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
R, s0, Rs, T = 10.0, 1.0, 4.0, 1.2
lat = SubstrateLattice(radius_cells=R, spacing=h)
c2 = C.C0SQ
x = lat.pos
G = np.exp(-0.5 * (x**2).sum(1) / s0**2)

def E0(p):
    g = np.exp(-0.5 * (p**2).sum(1) / s0**2)
    return np.stack([-p[:, 1] / s0**2 * g, p[:, 0] / s0**2 * g, 0 * g], 1)

ae = C.AE(lat, C.sample_A(lat, E0))
dt = C.stable_dt(lat); n = int(np.ceil(T / dt)); dt = T / n
V = (h * 2 / 3**0.5) ** 3 / 2
r = np.linalg.norm(x, axis=1)
inside = r < Rs
dr = 2 * h
shell = np.abs(r - Rs) < dr / 2
rhat = x / np.maximum(r, 1e-12)[:, None]

def measure():
    E = ae.E_cell(); B = C.curl_A(lat)
    U = 0.5 * ((E**2).sum(1) + c2 * (B**2).sum(1)) * V
    S = c2 * np.cross(E, B)
    F = float((S[shell] * rhat[shell]).sum(1).sum() * V / dr)
    uE = 0.5 * (lat.u**2 + 0.25 * (lat.phi**2).sum(1)) * V
    return float(U[inside].sum()), F, float(U.sum()), float(np.abs(lat.u).max())

U0, F_prev, Utot0, _ = measure()
flux_int = 0.0
print(f"h={h} N={lat.N} n={n} dt={dt:.5f} U_in0={U0:.5e} U_tot0={Utot0:.5e}", flush=True)
t0 = time.time()
for i in range(1, n + 1):
    ae.step(dt)
    if i % max(1, n // 12) == 0 or i == n:
        U, F, Utot, umax = measure()
        # trapezoid on the report grid is too coarse; accumulate every step instead
    else:
        U = None
    if U is None:
        E = ae.E_cell(); B = C.curl_A(lat)
        S = c2 * np.cross(E, B)
        F = float((S[shell] * rhat[shell]).sum(1).sum() * V / dr)
    flux_int += 0.5 * (F + F_prev) * dt
    F_prev = F
    if U is not None:
        print(f"  t={i*dt:.3f} U_in/U0={U/U0:.4f} (U0-U_in)/U0={(U0-U)/U0:+.4f} "
              f"flux_out/U0={flux_int/U0:+.4f} balance_err={(U0-U-flux_int)/U0:+.2e} "
              f"U_tot/U_tot0={Utot/Utot0:.4f} |u|max={umax:.2e} ({time.time()-t0:.0f}s)", flush=True)
