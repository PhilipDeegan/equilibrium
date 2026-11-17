# 2026-10-19 grad: a held ring (step() + Cartan) in a throughput gradient (Philip: without a receiver there is no
# gradient; a lone body may cycle energy locally, like dipole field lines, but not offload far)

zoo.py --tilt g: face sigma = 1 + g sin(2 pi x / side) (set_face_medium, step() only; Cartan transport unchanged).
In a uniform sigma a div-free ring cannot feed u; with sigma varying, the sigma-weighted divergence of the ring's flow
is nonzero -> u (waves) is driven in proportion to the gradient. Diagnostics: Eu = sum u^2 (energy that has left the
vortical sector), energy in shells r < 0.5 R / R / 1.5 R / 2 R / beyond around the ring centroid (bounded oscillation =
cyclic local exchange; steady outward drift = offload). E stays exact (step() symmetric in sigma, Cartan orthogonal).
R 9 smoke (tilt 0.1, T 6): dE/E 3e-7; Eu rises to 34 (1.2 % of E) by t 5 and levels off.
R 20 sweep g 0 / 0.01 / 0.03 / 0.1, T 30 (r20/) running.

## R 20 result (T 30; dE/E -1.7e-8 in all)
| tilt g | Eu t 5 / 10 / 15 / 20 / 25 / 30 | sum|G| t 30 - (g 0) | outer shell (> 2 R) t 0 -> 30 | rms2r |
| 0 | 12.00 / 15.64 / 16.09 / 13.11 / 13.34 / 10.22 | 0 | 172.4 -> 178.3 | +1.72 % |
| 0.01 | 12.00 / 15.64 / 16.09 / 13.11 / 13.35 / 10.22 | +0.82 | 172.4 -> 178.3 | +1.72 % |
| 0.03 | 12.03 / 15.66 / 16.11 / 13.08 / 13.36 / 10.26 | +2.65 | 172.4 -> 178.3 | +1.77 % |
| 0.1 | 12.36 / 15.90 / 16.35 / 13.03 / 13.51 / 10.61 | +9.69 | 172.4 -> 178.2 | +1.73 % |
(E 6456; shells r < 10 / 20 / 30 / 40 / beyond.)
1. The LONE ring (g 0) already exchanges energy with the wave (u) sector: Eu rises from 0 to ~16 (0.25 % of E) and
   oscillates (16 -> 10), bounded; shells hold to ~1 % with no outward drift. Source: the Cartan transport's exact
   (gradient) part feeds u. = Philip's "recurrent cyclic" local exchange, not offload, in the lone body.
2. The gradient adds only a smooth perturbation: extra Eu at t 30 = 0.003 / 0.034 / 0.39 for g 0.01 / 0.03 / 0.1
   (~ g^2: x11 for x3.3), extra sum|G| ~ linear in g (deformation of the flow, not loss). No threshold, no offload
   in T 30 at any tilt. Shells identical to 0.1 %.
Reading: in step() + Cartan a throughput gradient barely reaches the structure: sigma enters step() (the u sector)
only, while the vortical part is carried by Cartan, which ignores the medium. A receiver's gradient must act on the
transport itself (how fast circulation is carried in slower cells) to matter. 10-12: a naive sigma x transport broke
energy -> needs an energy-consistent form (sigma-weighted metric in i_v da, both legs).
