"""
Probe: flux tube in Saha vacuum with noise injection.

Builds a zero-proton domain under the reciprocal BC (vacuum conserves total u,
boundary injects Gaussian noise as a thermal bath).  Drops a toroidal phi
winding at the centre and runs forward, reporting every REPORT steps:

  step      — tick counter
  u_mean    — domain mean of u (should stay near U_BG)
  u_std     — spatial std of u (thermal width)
  u_ring    — mean u at ring cells (does flux tube attract/repel u?)
  phi_circ  — total phi circulation around the ring (winding persistence)
  phi_rms   — rms phi across the whole domain
  winding   — discrete winding number (should stay ±1 if tube persists)
"""

import numpy as np

from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    phi_ring_circulation,
    winding_number,
)
from equilibrium.substrate.lattice import DampedLattice

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R = 10.0
DT = 0.005
U_BG = 1.0
FLUX_AMP = 1.0
NOISE_AMP = 0.1
DAMP_MAX = 0.005  # sponge absorbs outgoing waves; reciprocal BC restores mean
N_STEPS = 4000
REPORT = 400
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Reciprocal BC — inline so probe is self-contained
# ---------------------------------------------------------------------------


class ReciprocalBC:
    def __init__(self, lat, u_target, rng):
        nb = lat.nb
        boundary = np.any(nb < 0, axis=1) & ~lat.fixed_u
        self._idx = np.where(boundary)[0]
        self._target = float(u_target)
        self._rng = rng

    def step(self, lat, noise_amp=0.0):
        deficit = self._target - float(lat.u.sum())
        n = len(self._idx)
        if n == 0:
            return
        weights = self._rng.exponential(size=n)
        weights /= weights.sum()
        injection = deficit * weights
        if noise_amp > 0:
            noise = noise_amp * self._rng.standard_normal(size=n)
            noise -= noise.mean()
            injection += noise
        lat.u[self._idx] += injection


def main():
    # ---------------------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    # DampedLattice: sponge absorbs outgoing waves and damps u/phi toward U_BG.
    # ReciprocalBC: restores the mean u the sponge removed + injects thermal noise.
    # Together: stable thermal equilibrium at U_BG with variance ~ NOISE_AMP / DAMP_MAX.
    lat = DampedLattice(radius_cells=R, u_bg=U_BG, damp_max=DAMP_MAX)
    lat.u[:] = U_BG

    bc = ReciprocalBC(lat, u_target=float(lat.u.sum()), rng=rng)

    ring_tiles, ring_faces = find_toroidal_ring(lat, axis="z")
    if ring_tiles is None:
        raise RuntimeError("toroidal ring not found — increase R")

    centre = lat.pos[ring_tiles].mean(axis=0)
    init_toroidal_winding(lat, ring_tiles, ring_faces, amplitude=FLUX_AMP, frozen=False)

    print(f"N={lat.N}  ring={len(ring_tiles)} tiles  centre={centre.round(2)}")
    print(f"DT={DT}  noise={NOISE_AMP}  flux_amp={FLUX_AMP}")
    print()
    print(
        f"{'step':>6}  {'u_mean':>8}  {'u_std':>7}  {'u_ring':>8}  "
        f"{'phi_circ':>10}  {'phi_rms':>8}  {'winding':>8}"
    )
    print("-" * 72)

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------

    def _report(step):
        u = lat.u
        phi = lat.phi
        u_ring = float(u[ring_tiles].mean()) if ring_tiles else float("nan")
        circ = phi_ring_circulation(lat, ring_tiles, centre)
        w = winding_number(lat, ring_tiles, axis="z")
        phi_rms = float(np.sqrt(np.mean(phi**2)))
        print(
            f"{step:>6}  {float(u.mean()):>8.4f}  {float(u.std()):>7.4f}  "
            f"{u_ring:>8.4f}  {circ:>10.4f}  {phi_rms:>8.4f}  {w:>8.3f}"
        )

    _report(0)
    for s in range(1, N_STEPS + 1):
        lat.step(DT)
        bc.step(lat, noise_amp=NOISE_AMP)
        if s % REPORT == 0:
            _report(s)


if __name__ == "__main__":
    main()
