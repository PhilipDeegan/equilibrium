# Electron candidates under self-sourced lam, 2(a) — 2026-09-29

**INVALID as structure evidence (Philip 10-01): lump widths 1-2 cells. An electron below R20 (radius 20 cells) is totally invalid, R20 maybe too small (mem feedback_structure_resolution). Survival / retention / response numbers below are records of the setup only.**

`self_lam.py`: discrete step() (sigma at the face on both legs), DampedLattice rate sponge
(0.25), R = 12, each step lam = eps + alpha (u^2 + 1/2 sum phi^2), eps = 0. 6 crossings (t = 62).
H = 1/2 sum u^2 + 1/2 sum phi^2 is conserved for any sigma(t) with face sigma, so the
nonlinearity only redistributes energy. Log `self_lam.log`, plot `self_lam.png`, end states
`end_*.npz`. Core = r < 4.5. Split = u / gradient phi / divergence-free phi (frozen).

| IC | alpha | amp | core late/initial | lam_max end | core split u/grad/sol |
|---|---|---|---|---|---|
| gauss | 0 | any | 8e-12 | 0 | (tail) |
| gauss | 1 | 1 | 1e-5 | 4e-6 | 0.30/0.39/0.31 |
| gauss | 1 | 3 | 0.098 | 1.0 | 0.11/0.34/0.54 |
| gauss | 4 | 1 | 4e-4 | 6e-3 | 0.02/0.02/0.97 |
| gauss | 4 | 3 | 0.385 | 64 | 0.42/0.43/0.16 |
| dipole | 0 | any | 2e-9 | 0 | frozen tail |
| dipole | 1 | 3 | 0.015 | 0.31 | 0.00/0.00/1.00 |
| dipole | 4 | 3 | 0.364 | 10 | 0.19/0.34/0.47 |
| ring (control) | 0..4 | 1,3 | 0.95..0.999 | .. | ~all frozen circulation |

Reading:
- Weak nonlinearity: a small plateau that is mostly divergence-free phi: varying sigma makes
  phi_t = -sigma_f W du non-gradient, and that part freezes -- manufactured circulation, not a
  trapped wave.
- Strong nonlinearity (alpha amp^2 ~ 36): 36-38% of the core energy stays, mostly propagating
  content (u + gradient phi), but lam_max = 10..64 (sigma 0.09..0.015): the lump survives because
  its own energy has nearly stopped its clock -- self-trapping by time dilation (DST's sigma -> 0
  limit), not a dynamical balance.
- Retention rises smoothly with alpha amp^2: no threshold, no preferred amplitude or size in this
  scan -- nothing yet that would give one e/m (selection criterion 3).
Next: longer runs (does the trapped lump keep leaking at a rate ~ sigma_core, or stop?), fine
amplitude scan for a threshold / preferred state, translation under a push (criterion 2), and
eps > 0 (the floor).
