"""Discrete boundary design, numpy step() (current weights).

Wave part: a rate sponge that keeps faces consistent. After each step()
    u   *= exp(-g_cell dt),  phi *= exp(-g_face dt),  g_face = mean of the two cells' g,
g = gmax * t^p over a layer of width w outside a fixed interior radius Ri. Outgoing gaussian u
pulse; reflection = interior energy (r < Ri) in a window after the pulse has left, vs a
reference domain big enough that nothing returns in the window.

Static part: fixed u = 1 at the centre cell with u = 0 held on the outermost shell (Dirichlet),
with and without the sponge; compare with the finite-domain solution a (1/r - 1/b).
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice, wave_speed

out = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--Ri", type=float, default=9.6)
p.add_argument("--width", type=float, default=1.5)
p.add_argument("--static-steps", type=int, default=40000)
args = p.parse_args()

c = wave_speed()
Ri = args.Ri


class RateSponge(SubstrateLattice):
    def __init__(self, R, w=0.0, gmax=0.0, power=2):
        super().__init__(radius_cells=R)
        self.r = np.linalg.norm(self.pos, axis=1)
        t = np.clip((self.r - Ri) / w, 0.0, 1.0) if w > 0 else np.zeros(self.N)
        self.g = gmax * t**power
        valid = self.nb >= 0
        gn = np.where(valid, self.g[np.where(valid, self.nb, 0)], self.g[:, None])
        self.gf = 0.5 * (self.g[:, None] + gn)

    def step(self, dt):
        self.step_cpp(dt)
        if self.g.any():
            u = self.u * np.exp(-self.g * dt)
            self.u = np.where(self.fixed_u, self.u, u)
            self.phi = self.phi * np.exp(-self.gf * dt)


def energy_in(lat, m):
    valid = lat.nb >= 0
    return 0.5 * np.sum(lat.u[m] ** 2) + 0.25 * np.sum((lat.phi[m] * valid[m]) ** 2)


t_leave = (Ri + 3 * args.width) / c
t_win = (t_leave, t_leave + 2 * Ri / c)


def pulse(lat):
    lat.u[:] = np.exp(-(lat.r**2) / (2 * args.width**2))
    m = lat.r < Ri
    e0 = energy_in(lat, m)
    dt = lat.stable_dt()
    n = int(np.ceil(t_win[1] / dt))
    vals = []
    for s in range(n):
        lat.step(dt)
        t = (s + 1) * dt
        if t >= t_win[0] and s % 10 == 0:
            vals.append(energy_in(lat, m) / e0)
    return float(np.mean(vals)), float(np.max(vals))


ref = RateSponge(Ri + c * t_win[1] / 2 + 2.0)
print(f"c={c:.3f} window t={t_win[0]:.2f}..{t_win[1]:.2f}; reference mean/max {pulse(ref)}", flush=True)

rows = []
for w, power, k in itertools.product((2.4, 4.8, 9.6), (2, 3), (1.0, 3.0, 10.0, 30.0)):
    gmax = k * c / w
    lat = RateSponge(Ri + w, w, gmax, power)
    mean, mx = pulse(lat)
    rows.append((w, power, k, gmax, mean, mx))
    print(f"w={w:4.1f} p={power} gmax={k:4.0f} c/w={gmax:7.3f}  refl mean {mean:.3e} max {mx:.3e}", flush=True)
np.save(out / "sweep_rows.npy", np.array(rows))

# production sponge for comparison at the same geometry (per call, per cell row)
from equilibrium.substrate.lattice import DampedLattice

for w in (2.4, 4.8, 9.6):
    R = Ri + w
    lat = DampedLattice(radius_cells=R, sponge_fraction=w / R, damp_max=0.02)
    lat.r = np.linalg.norm(lat.pos, axis=1)
    mean, mx = pulse(lat)
    print(f"production DampedLattice w={w} damp 0.02: refl mean {mean:.3e} max {mx:.3e}", flush=True)


# ---------------- static
def static(lat, label):
    b = lat.r.max()
    src = int(np.argmin(lat.r))
    lat.fixed_u[src] = True
    lat.u[src] = 1.0
    shell = lat.r > b - 0.6
    lat.fixed_u[shell] = True
    lat.u[shell] = 0.0
    dt = lat.stable_dt()
    snaps = []
    for s in range(1, args.static_steps + 1):
        lat.step(dt)
        if s % (args.static_steps // 4) == 0:
            snaps.append(lat.u.copy())
    late = float(np.abs(snaps[-1] - snaps[-2]).max())
    m = (lat.r > 2.0) & (lat.r < Ri - 1.0)
    X = 1.0 / lat.r[m] - 1.0 / b
    a = float(np.sum(X * lat.u[m]) / np.sum(X * X))
    resid = float(np.sqrt(np.mean((lat.u[m] - a * X) ** 2)) / np.abs(lat.u[m]).max())
    slope = float(np.polyfit(np.log(lat.r[m]), np.log(np.abs(lat.u[m]) + 1e-300), 1)[0])
    print(f"static {label}: late change {late:.2e}  fit a(1/r-1/b): a={a:.4f} rel rms resid {resid:.2e}  log slope {slope:.3f}", flush=True)
    return lat


static(RateSponge(Ri + 4.8), "Dirichlet shell only")
best = min(rows, key=lambda r: r[4])
static(RateSponge(Ri + best[0], best[0], best[3], best[1]), f"Dirichlet + best sponge w={best[0]} p={best[1]} k={best[2]}")
