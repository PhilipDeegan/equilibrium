# Electron translation gate — pressure-tilt kick on the double-ring dipole

Tests whether external pressure (a pinned static u tilt) can translate the
lam-free electron dipole, and whether coupling is gated by the interior
crossing channel (step_cross). Inertia framing: (1) omnibalance — the
zero-net-force IC is the discrete omnibalance state; (2) cost of translation —
motion should require the interior crossing.

## Parameters

- `probe_electron_translation.py --report 500` (all defaults)
- R = 20, scale = 4.0 (same knot as the R=80 runs), N ≈ 43k
- gradient g = 5e-4 per cell, pinned at a battery ring just inside the sponge
- phases: 500 base / 3000 kick / 2000 coast; dt = 5.128e-3
- configs: cross ∈ {0, 0.01, 0.02} × kick axis ∈ {x (equatorial), z (dipole axis)}

## Results

| cross | axis | v_kick | mobility v/g | pol_survival |
|-------|------|--------|--------------|--------------|
| 0.000 | x | +9.4e-17 | +1.9e-13 | 1.000 |
| 0.000 | z | -2.2e-17 | -4.4e-14 | 1.062 |
| 0.010 | x | +1.7e-03 | +3.4 | 0.166 |
| 0.010 | z | -5.2e-03 | -10.3 | 0.219 |
| 0.020 | x | -4.8e-03 | -9.6 | 0.028 |
| 0.020 | z | -1.3e-02 | -25.1 | 0.099 |

## Findings

1. **cross = 0: coupling is EXACTLY zero** (mobility ~1e-13, machine
   precision, both axes; winding perfectly preserved). Confirms the
   superposition argument: with lam = 0 the operator set is linear, the
   tilt adds an independent background solution, and the winding cannot
   feel it. Omnibalance is untouchable by pressure in the linear sector.
2. **cross > 0: contact but NO transport.** COM wander is sub-cell
   (±0.05), flips sign between configs (x: +3.4 → -9.6 going 0.01 → 0.02),
   does not scale with cross or with drive direction, and coast
   "velocities" are comparable to kick velocities with random signs. The
   cross=0.02/z coast v = -0.12 coincides with circ_pol jumping 6.6 → 24
   as prms_max falls to 0.03 — the COM tracker latching onto background,
   not motion. Meanwhile the winding erodes (pol_survival 0.17 → 0.03).
3. **Conclusion: the current operator set has no channel by which pressure
   translates a lam-free winding.** step_cross as implemented provides
   erosion + noise, never coherent displacement. "Inertia = cost of
   translation" is not mis-calibrated — it is unimplemented.

## Implications for step_cross v2

The redesign must supply both halves:
- coherent face transit passes free (no erosion of a winding at rest);
- an interior-crossing pathway that converts pressure asymmetry into
  coherent winding displacement — the actual cost of translation.
Candidate gating: fire only on transverse co-located flux (linkage,
reconnection) and on translation drive, not on any inward face-phi.

## Artefacts

- `probe_electron_translation.py` — script copy at run time
- `run_output.txt` — full stdout (both this run and per-step COM traces)
- `etrans.png` — displacement + winding-survival curves
- `etrans.npz` — summary array (cross, axis, v_base, v_kick, v_coast,
  d_kick, pol_frac), gradient
