# Bohmian double slit on the Kelvin (BCC) lattice

Question: if particles are guided by a wave that obeys the lattice's own rule, do they reproduce the double slit?
Expected: fringes where the wave puts them, arrival statistics matching |psi|^2 (Born rule), and no trajectory
crossing the symmetry axis.

Scope: this is QM's idealised double slit, a signpost. The wall is a rigid psi = 0 boundary, not matter. A real slit
is a hole in material whose charges interact with the particle. The run checks that the rule plus guidance gives the
textbook answer for the textbook setup. It says nothing about real trajectories: no-crossing, and which slit being
set by the starting side, belong to the idealisation (see `exp/2026/11/28/status.md`).

## Two things are simulated

1. **The wave**. A free Schrodinger packet stored as real pairs of arrows (R, I) on the two BCC sublattices (corner
   and body cells). Each step updates R from I and then I from R, using the 14-face Laplacian (8 hex faces weighted
   2/3, 6 square faces 1/3). The wave does not know about the particles.
2. **The particles**. These are test points that follow the wave and do not act back on it. Each one is a cell index
   plus a 3D offset from that cell's centre. When the offset leaves the cell's Voronoi region, the particle hops to
   the neighbour across that face, hex or square.

### Guidance

The wave's local flux defines the velocity: v = J / rho, with J = Im(psi* G), rho = |psi|^2, and
G = (1/2m) sum_k w_k c_k psi_k. G is built from the same 14 face terms as the Laplacian, so J is the rule's own face
current. psi and G are fit with a local quadratic over the cell and its 14 neighbours, then evaluated at the
particle's offset. The fields are taken as linear in time across a step. Each step is a Heun step. A fast particle
(near a node of psi, where rho -> 0) is split into substeps of at most 0.05 cells each, capped at 1024. There is no
speed clamp.

## Setup (defaults)

| | |
|---|---|
| box | 192 x 192 x 2 cells per sublattice (147456 cells in all), uniform in z, so effectively 2D |
| packet | starts at x = 36, sigma x/y = 8/20, wavelength 10.1 (k a ~ 0.62), moving +x |
| wall | psi = 0 on 3 BCC layers at x = 72; two slits 4 wide, 24 apart (centre to centre) |
| screen | the line x = 152, distance L = 80 behind the wall |
| sponge | 20 cells on the x and y borders, psi damped (exp(-gamma dt (e/W)^2)) |
| time | dt = 0.05, t_end = 300 (6000 steps) |

## How many particles

**20000 by default** (`--ens-x 40 --ens-y 500`), taken without randomness from |psi0|^2. A 40 x 500 midpoint grid on
(0, 1)^2 is mapped through the inverse normal CDF to x and y. All 20000 are traced the whole run; they leave when they
cross the screen or enter the sponge.

- About 5.8% get through the slits, roughly 1160 hits. The rest reflect off the wall and are absorbed by the sponge.
- Only particles from the front of the packet get through. Which slit a particle uses is fixed by the side of the axis
  it starts on.
- The full path is saved for only `--n-traj` particles (80 by default), every 20 steps. These are for the trajectory
  plot. The others keep only their screen hit (y, t). `--traj-from run.npz` picks the saved particles from those that
  reached the screen, spread across it. The run is deterministic, so re-tracing gives identical hits.

The cost is in the particles: with 20000 particles a run takes ~19 min on 1 core, while the wave alone takes ~1 min.

## What is measured (`analyse.py run.npz [traj.npz]` -> `bohm_slit.png` next to run.npz)

- **Fringes**: peaks of the time-integrated flux J_x across the screen. Fraunhofer predicts a spacing of
  lambda L / d = 33.7.
- **Born rule**: the particles' arrival histogram on the screen (2-cell bins) against the time-integrated flux, as
  chi2/dof with sqrt(N) errors. The flux depends only on the wave, so it is the same for any particle count.
- **Transmitted fraction**: particles hit / N against the flux through the screen.
- **No crossing**: the number of particles that ever change sides of y = yc. Should be 0, since the setup is mirror
  symmetric about y = yc.
- **Substeps**: the largest per-particle substep count, and how many particles hit the cap (should be 0).

## Results

- `run_g.npz` (20000 particles, current code): peaks at 0 and +-34, chi2/dof 62/76, transmitted 0.0579 (particles)
  vs 0.0599 (flux), 0 crossings, max 28 substeps, 0 at the cap.
- Small sample, `exp/2026/11/28/bohm_slit_sample/` (`--ens-x 10 --ens-y 200`, 2000 particles, 263 s): the same
  flux (it does not depend on the particles), 110 hits, transmitted 0.0550 vs 0.0599 (1 sigma), 0 crossings, max 14
  substeps, 0 at the cap. Its chi2/dof of 31/76 is not a test: with about 1.5 hits per bin, the sqrt(N) errors
  overstate the scatter.
- The default `--n-traj` paths come from the packet's centre column. That column reflects, so no saved path reaches
  the screen. Use `--traj-from`.
- `run.npz`, `run_clamp.npz` and `run_traj.npz` come from the older code: J and rho fit separately and a speed clamp
  |v| <= 2. Its trajectories are still the ones in the left panel of `bohm_slit.png`. See `../status.md`.
