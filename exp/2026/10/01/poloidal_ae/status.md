# Poloidal coil on step_ae (10-01 evening) -- CONTROL, not a candidate

`run.py --pitch P` (C++ step_ae, DEC curl, iso weights, h = 1, R56, 955k cells), torus r0 12, w 8/3 (16 cells across
the tube), no lam (step_ae refuses lam != 0). Seed: A = f(d) [(r0/rho) psi_hat + P theta_hat], f = (d/w) Gaussian
with a smooth cutoff, div A = 0 exactly in the continuum; u = E = 0 (coil at rest). Pitch 0: closed poloidal loops of
phi, B toroidal and confined to the tube; pitch 1: helical. Sponge: C++ quadratic on u/phi + E damped in Python.
3 crossings (t 168), 15-18 min per run. Plots: `coil_rec.png` (plot_rec.py), `coil_p*_{ic,stepNNNNNNN,final}.png`.
Snapshots /bk/dst/2026/10/01/poloidal_ae/.

| | t 13.8 | t 27.6 | t 41 | end (t 168) |
|---|---|---|---|---|
| p0 E_struct | 0.30 | 1.8e-3 | 2.6e-3 | 1.0e-4 |
| p0 E_tube | 0.084 | 9e-4 | 6e-4 | 2.4e-5 |
| p1 E_struct | 0.31 | 3.3e-3 | 6.5e-3 | 1.1e-4 |

- Both radiate away within ~half a crossing (t ~ 20-28): the seed projection a_psi 1 -> 0.05 by t 14, energy
  half in E during the transit (A <-> E exchange), then out at c into the sponge. Nothing holds; pitch makes no
  difference. Energy conserved to ~1e-4 before the sponge.
- Remnant ~1e-3: a square-symmetric E pattern focused at the centre (lattice anisotropy of the inward-radiated
  part), decaying slowly; not structure.
- Expected for linear vacuum Maxwell (no finite-energy localised non-radiating solution). Philip 10-01: magnetism
  is a consequence, not a thing in itself -> step_ae persistence would make the E/B sector constitutive; this run
  is only the control of what E/B alone does.
