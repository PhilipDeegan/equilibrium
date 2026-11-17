"""
Static (no time-stepping) check: for a fixed toroidal ring winding number
(topologically pinned circulation Gamma), does the frozen-loop field energy
H = 0.5*sum(phi**2) scale like 1/radius as the ring is scaled up via
find_toroidal_ring(mode="square", hops=N) while Gamma is held exactly fixed
(amplitude = Gamma / N_edges, N_edges = 8*hops)?

Motivation (2026-09-02 session): candidate first-principles stability
mechanism for a circulation-carrying knot -- if circulation is exactly
conserved under step() (already confirmed this session, see now.md's
"ring lean-drift" entry) and per-edge phi-energy is quadratic, then
compressing a fixed-circulation loop to smaller radius costs energy
diverging as 1/r -- a real barrier against collapse to a point, with no
extra confinement mechanism (no lam profile) required. This script checks
only the algebraic/geometric half of that claim; it is NOT a dynamical
test that real stepping resists compression -- see now.md caveat.

Run:
  PYTHONPATH=$PWD python3 -u exp/2026/09/02/18_28_circulation_quantization/probe_circulation_energy_barrier.py
"""

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    set_flux_loop,
    winding_number,
    _sq_hop_dist,
)

AMP0 = 8.0


def main():
    for hops in [1, 2, 3, 4, 5, 6, 8, 10]:
        lat = SubstrateLattice(radius_cells=max(20, hops * 3 + 8))
        tiles, faces = find_toroidal_ring(lat, axis="z", mode="square", hops=hops)
        if tiles is None:
            print(f"hops={hops}: ring not found")
            continue
        w = winding_number(lat, tiles, axis="z")
        a = _sq_hop_dist(lat)
        n_edges = len(tiles)
        r_phys = a * hops
        amplitude = AMP0 / hops
        set_flux_loop(lat, tiles, faces, flux=amplitude)
        gamma = amplitude * n_edges
        h = 0.5 * np.sum(lat.phi**2)
        print(
            f"hops={hops:2d}  N={n_edges:3d}  r~{r_phys:6.3f}  w={w:+.4f}  "
            f"Gamma={gamma:8.4f}  amp={amplitude:8.4f}  H={h:10.5f}  "
            f"H*r={h * r_phys:8.4f}"
        )


if __name__ == "__main__":
    main()
