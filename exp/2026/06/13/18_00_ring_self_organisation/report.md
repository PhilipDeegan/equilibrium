# Ring Self-Organisation from PHARE IC

## Parameters

- R = 100 BCC cells, uniform (max_level=0), N = 5,441,209 cells
- phare-ic: `/home/p/git/phare/master/ic_b.npz`, scale=10.0
  - Ring fits in r < 37 BCC cells; vacuum gap r=37–90 before sponge
- cross-scale = 0.0001
- sponge-frac = 0.1, damp-max = 0.05
- dt = 5.1282e-03, lam-scale = 0.05
- n-threads = 10

## Key finding

**phi=0 IC with u ∝ |B| and lam ∝ |B| self-organises into a proton+moat ring structure by ~1200 steps.**

### phi_max growth (RMS vs PHARE IC baseline)

| Steps | RMS |
|-------|-----|
| 0 | 0 |
| 100 | 6.98e-04 |
| 200 | 1.31e-03 |
| 400 | 2.41e-03 |
| 600 | 3.35e-03 |
| 800 | 4.14e-03 |
| 1000 | 4.75e-03 |
| ~1200 | plateau |

### Sequence

1. **0–1000 steps**: phi builds up from zero, driven by u gradients via pass 2. Spatial structure is a smooth spherically-symmetric blob centred on origin. u field is uniformly positive. RMS growth decelerating.
2. **~1200 steps**: phi_max plateaus — toroidal circulation is self-sustaining. Local equilibrium reached.
3. **1000→1500 steps**: u field spontaneously develops a **ring of positive u with a negative-u centre (moat)**. Visible in all three cross-section planes (XY, XZ, YZ) as a red torus with blue hole.

## Interpretation

The phi plateau marks the moment the toroidal circulation becomes self-sustaining. The back-reaction of circulating phi on u (pass 1) carves out the ring/moat structure. This is the DST atom: proton knot = ring of positive u, electron moat = negative u at centre. No topology was imposed — it emerged from the lattice update rule alone.

## What failed before (and why)

- **phi projection from PHARE B field**: discrete divergence ratio = 3.7 (divergence 3.7× larger than signal). Helmholtz cleaning did not work cleanly due to non-leaf neighbour indices and face anti-symmetry issues.
- **cross-scale = 0**: without step_cross, inward phi piles up at centre as u oscillations (breathing mode) rather than redirecting into ring circulation.
- **scale=20**: ring filled entire active zone (r < 75 BCC = sponge boundary). No vacuum gap. scale=10 solved this.

## Artefacts

- `shell_profile_R100_s10_cs_1000.h5` — snapshot at 1000 steps (pre-ring)
- `shell_profile_R100_s10_cs_1500.h5` — snapshot at 1500 steps (ring visible)
- `shell_profile_R100_s10_cs_1000.png` — smooth u blob, no ring
- `shell_profile_R100_s10_cs_1500.png` — clear ring+moat in u field

## Next steps

- Run to 2000 steps, inspect whether ring sharpens further
- Measure shell circulation profile once ring is stable
- Check if far-field phi falls as r^(-2) (magnetic dipole) or r^(-3)
- Log the ring's natural oscillation frequency (the DST analogue of the Compton period)
