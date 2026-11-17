# Two flux sources: interaction sign vs phase, step() level 0 (2026-09-25)

`phase_interaction.py`. Sources inject at a prescribed rate
(u_t = -div phi + q_i(t) g_i, gaussian s = 1.5): each holds its own flux
divergence (the Kirchhoff charge), u not pinned. F = -dE_int/dd,
E_int = <H_both> - <H_1> - <H_2>, H = step()'s energy over r < 25.6 (sponge
beyond 32), R = 40, h = 1. Amplitude-driven = pinned: the SIGN is the result,
not evidence for a particle. Philip: physics has no "how" for charge ("it
does"), so this takes charge as given and tests what follows.

## DC (w = 0)
| d | E_int like | E_int unlike |
|---|---|---|
| 4 | +1.412e-3 | -1.412e-3 |
| 6 | +8.711e-4 | -8.711e-4 |
| 8 | +5.561e-4 | -5.561e-4 |
| 12 | +2.416e-4 | -2.416e-4 |
Like REPEL, unlike ATTRACT, exactly antisymmetric. E = a/d + b fits to 2%;
b = finite measurement ball (-a/b = 21.6 vs radius 25.6). a = 0.90 x
q1 q2/(4 pi c^2 V) -- Coulomb with c^2 = C0SQ.
Predicted before running: positive phi^2 energy + flux-fixed sources =>
cross term +q1 q2/d (overlap-cost row: each structure holds its distortion).
So step()'s statics carry the electrostatic sign once charge = conserved net
flux divergence; says nothing about where charge comes from.

## Oscillating
- w = 0.456 (lambda 50 > R 40): INVALID, cavity -- single-source H depends on
  position (1.47e-3 -> 2.71e-3). Files renamed *_cavity_INVALID.
- w = 1.911 (lambda 12): energy E_int follows radiation-in-transit
  interference (~ sin(kd)/kd), H1 still +-15% with position. Energy in a ball
  is not a force for radiating sources -> replaced by the force proxy.

## Force proxy (`force_proxy.py`): F_2 = <q_2(t) g_2 . A_1(t)>, single-source runs
Consistent with the DC energy (F = -dE/dd = q_2 A_1). Positive = like repel.
| d | kd | F DC | ref 1/(4 pi c^2 d^2) | F w=1.911 | continuum ref (form factor) |
|---|---|---|---|---|---|
| 3 | 1.57 | 2.76e-4 | 6.63e-4 | +1.08e-4 | +2.81e-4 |
| 4.5 | 2.36 | 2.28e-4 | 2.95e-4 | +2.24e-5 | +2.86e-5 |
| 6 | 3.14 | 1.56e-4 | 1.66e-4 | -6.78e-5 | -8.27e-5 |
| 9 | 4.71 | 7.09e-5 | 7.37e-5 | -9.02e-5 | -9.38e-5 |
| 12 | 6.28 | 3.81e-5 | 4.14e-5 | +2.23e-5 | +2.05e-5 |
(d=3: the two s=1.5 gaussians overlap, force softened.)
- DC: like repel, -> Coulomb 1/d^2 (0.92-0.96 of continuum at d >= 6).
- Oscillating: sign pattern repel/repel/ATTRACT/ATTRACT/repel matches the
  retarded continuum [cos kd + kd sin kd]/d^2 exactly; magnitudes 0.8-1.1.
  NOT inverted Bjerknes: in-phase repel at small kd, then the retardation
  sign alternation any classical wave interaction has.

## What it means
- step() with flux-fixed sources is classical retarded scalar-potential
  interaction: electrostatic sign in the quasi-static limit.
- Phase alone is not charge: for kd >> 1 the time-averaged force oscillates
  in d (~ sin(kd)/d) and averages away over a spread of separations. An
  electron-scale internal frequency (wavelength ~ Compton) would give kd >> 1
  at atomic distances -> no Coulomb. Coulomb needs a nonzero MEAN flux
  divergence (a DC charge), consistent with 09-24 (periodic flux cannot make
  charge linearly).
- Open: the coupling. q.A follows from the DC energy with flux-fixed sources.
  A structure responding to grad u instead (pressure, fluid-Bjerknes type)
  feels nothing at DC (u uniform) and a different, underived sign when
  oscillating. Which one an emergent structure responds to
  needs a free structure -- none exists yet.
