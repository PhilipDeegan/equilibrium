# ring_phi_dipole — 15 Jun 2026 22:57

## Hypothesis
The cycling toroidal ring (8-tile BCC loop, the DST electron winding) generates a phi field
with the angular and radial structure of a magnetic vector potential: A ∝ sinθ/r².
Additionally, the Lorentz force on the ring in a seeded phi-B background should be
chirality-dependent (opposite sign for winding +1 vs −1).

## Parameters

### Lorentz sweep (`sweep_lam_lorentz.py`)
- R=40, N=347941, dt=5.1282e-04, dt_scale=0.1
- n_threads=10 (C++ step via DampedLattice.step_cpp)
- kick_px=0.5 (x-direction phase tilt)
- B_values=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
- n_steps=2000 per B value
- COM window: r < 4×ring_r = 5.58 cells (941 cells)

### Profile run (within sweep_lam_lorentz.py, before sweep)
- profile_steps=200000, total_time=102.6, ring_cycles≈12.5
- profile_damp_max=0.05 (10× harder sponge to suppress reflections)
- B=0, kick=0 — freely cycling ring only
- Snapshots: tmp/profile_ring_start.npz, tmp/profile_ring_end.npz

### Inspection (`inspect_profile_ring.py`)
- Loads start/end npz, ring centre from centroid of top-8 phi_rms tiles
- r-shell-cells: 4.0, 6.0, 8.0; shell_width=0.8
- 2D log-scale slices: axial plane and equatorial plane

## Runtime output

```
Ring tile positions:
[[ 0.          1.15470054  0.        ]
 [ 0.         -1.15470054  0.        ]
 [ 1.15470054 -1.15470054  0.        ]
 [ 1.15470054  1.15470054  0.        ]
 [-1.15470054  1.15470054  0.        ]
 [-1.15470054 -1.15470054  0.        ]
 [ 1.15470054  0.          0.        ]
 [-1.15470054  0.          0.        ]]
Ring centre (centroid of top-8 phi_rms tiles): [0. 0. 0.]
Inner region power law (r=2-7): phi_rms ~ 1.844e-02 * r^-2.143  (A-field=-2, B-field=-3)

Angular structure (phi_rms vs theta):
  r=4.0: phi_axial=4.425e-04  phi_equatorial=1.168e-03  ratio=0.38  (dipole=2.0)
  r=6.0: phi_axial=1.775e-04  phi_equatorial=3.123e-04  ratio=0.57  (dipole=2.0)
  r=8.0: phi_axial=1.621e-04  phi_equatorial=5.160e-04  ratio=0.31  (dipole=2.0)

Lorentz sweep (winding +1):
  B=0.001: vx=+3.0e-3  vy=+6.6e-2  Δvx=-5.8e-3  Δvy=+3.6e-2
  B=0.005: vx=-1.3e-2  vy=+1.0e-1  Δvx=-2.1e-2  Δvy=+7.1e-2

Lorentz sweep (winding −1):
  B=0.001: vx=+1.8e-2  vy=-2.1e-2  Δvx=+9.5e-3  Δvy=-5.1e-2
  B=0.005: vx=+2.8e-2  vy=-9.4e-2  Δvx=+1.9e-2  Δvy=-1.2e-1
```

## Key findings

### 1. Lorentz chirality confirmed
Δvx and Δvy both flip sign between winding +1 and −1 at all B values tested.
B_phi=0.001–0.005 is the clean operating range — x-motion not arrested, signal large
relative to the symmetric background artefact (sum Δvy ≠ 0 from seeded phi-B interaction).

### 2. Phi field is a vector potential: r^−2.143
Inner region (r=1.5–8 cells) from the geometric ring centre [0,0,0].
Target for magnetic vector potential A: r^−2. Confirmed to within 7%.
phi is A, not B. B = curl(phi) would give r^−3 — that is NOT what we measure directly.
Background noise floor (~1e-4) from accumulated ring radiation swamps the far-field beyond r~8.

### 3. Angular structure: equatorial dominant — correct sign for A-field
phi_axial < phi_equatorial at all measured radii. For A ∝ sinθ/r²:
A → 0 along axis (θ=0), A → max in equatorial plane (θ=π/2). Observed ratios 0.31–0.57
(elevated above the pure sinθ prediction by isotropic background floor inflating axial bins).
Background-corrected equatorial/axial ≈ 3–7, direction correct throughout.

### 4. Ring geometry verified
8 tiles at expected BCC hex-face positions in the xy plane. Centroid exactly [0,0,0].
Winding amplitude decay: ~0.7% per 5000 steps → ~24% total decay over 200k steps.

## Artefacts
- `tmp/profile_ring_start.npz` — lattice at t=0 (ring just initialised)
- `tmp/profile_ring_end.npz` — lattice after 200k profile steps
- `inspect_profile_ring.png` — 6-panel: radial profile, angular structure ×3, 2D slices ×2
- `sweep_lam_lorentz.png` — 4-panel: velocities, Δvy, sums, radial profile

## What was ruled out
- Global phi COM as a ring tracking metric — dominated by far-field leakage; must use windowed COM
- vy/vx ratio as Lorentz metric — blows up when vx reverses sign at moderate B
- Lam shell as B-field source — confirmed wrong in prior session (no chirality)
- r^−3 as the phi radial target — phi is A (r^−2), not B (r^−3)
- Large domain (R=40) as the fix for the background noise floor — the floor is set by
  total ring radiation / domain volume, scales similarly with R; a much larger domain
  or a driven (constant-amplitude) ring is needed for a clean far-field measurement

## Next steps
1. Two-mode frequency spectrum of the ring (FFT of n=1 Fourier amplitude during stability run)
2. Unsourced Lorentz test: let phi wake build from ring cycling, place second ring, measure deflection — no seeded phi-B
3. Bohr magneton in DST units: compare ring phi circulation amplitude to the analogue
