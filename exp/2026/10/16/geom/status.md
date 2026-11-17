# 2026-10-16 geom -- geometry under fixed transport (step() + Cartan, kappa 1), next item 1

geom.py = 10-14 vill.py + --geom plain | twist | tokamak | hopf, --wind (core-loop circulation in turns), rms2 (|G|^2
weight = core-weighted). sweep.sh, summ.py -> summary.txt. Seeds /bk/dst/2026/10/16/geom/.
- twist: whole-turn smoothed core + toroidal swirl v0 exp(-d^2/2 wsw^2) through the core (helical vorticity lines)
- tokamak: whole-turn core + toroidal flow inside a shell d < rshell (poloidal vorticity sheet = windings around the core)
- hopf: a = Hopf-fibre tangent field at scale R0, normalised to linked 1; a || da, Lamb vector 0 (steady Euler)

## Rings (R 9 P 72 T 20; R 20 P 140 T 30; wsw 3 / rshell 6 / wshell 1.5 at R 20, x 0.45 at R 9)
R 9: every ring leaks the same (~+45 % rms, +34-42 % sum|G| by t 20). Windings 0.5 / 1 / 2 turns change it by a few %
(slow it in rms, mostly because they slow travel; sum|G| slightly worse at wind 2). No geometry selects at R 9.
R 20, matched travel (0.06 / 0.10 R), rms | sum|G| growth:
  plain   +0.28 / +0.98 %   -0.08 / +0.16 %
  twist   +0.36 / +1.22 %   +0.01 / +0.42 %
  tokamak +0.62 / +2.03 %   +0.27 / +1.04 %
R 9 -> R 20 every ring drops ~30x: still the discretisation residual. Windings add a little more residual (the shell is
the sharpest feature: 1.5 cells at R 20); none cycles better than the plain ring. Expect it to shrink at R 40; not run.
Travel: windings slow the ring (R 9 wind 2 tokamak 0.214 R vs plain 0.281 R) -- classical swirling-ring behaviour.

## Hopf
Does not travel (0.03 R at R 20 vs 0.11 R), Eu ~10x lower than rings, E exact (-5e-10). Slow spread (+2 % rms,
+2 % sum|G|) the same at R 9 and R 20 -> box truncation, not resolution: R 9 in P 140 (box / R 9 vs 4.6) rms -0.02 %,
sum|G| +0.72 %. So Hopf is the one geometry that cycles in place: flow runs along closed linked fibres, the structure
stays put. But it is steady for the classical reason (a || da -> -i_v(da) = 0); continuum Euler says the same. Nothing
selects it, and its circulation is still continuous (any normalisation is held).

## Verdict
Under step() + Cartan, geometry decides travel vs in-place cycling (Hopf / Beltrami-type a || da cycles in place;
rings with a Lamb vector travel), but no geometry leaks distinctively once resolved: the leaks are discretisation
residual. Leak-vs-cycle cannot select a quantum here; consistent with 10-14 (needs the missing relation, item 2).
