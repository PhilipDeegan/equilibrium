# 2026-10-22 depression: paper design (Philip: structure = substrate depression, maybe the cell gradient;
# a translating structure must be spongey AND prefer its default state)

Rule: step() + Cartan (+ store). H = sum u^2 / 2 + sum_j G(S_j). step() moves only the gradient part of phi
(phi_t = -sigma W grad u); Cartan is Lie-Poisson (10-21) -> the circulation content is frozen-in and transported.

## Route A: rate depression, sigma_f = 1 - d g(S) in the bracket -- dropped on paper
- sigma scales J, not H: dH/dt = 0 for any sigma(state) (skew), so the set of preferred (minimum-energy) states is
  unchanged. A rate can steer, not prefer.
- sigma uniform over a body = a local time rescale -> no length is selected.
- Already seen in runs: 10-18 saturating-throughput lumps held by self-slowdown but leak, scale-free, lose momentum;
  10-19 csig = refraction toward slower cells only.

## Route B: energy depression, G concave then saturating (cheapest content per cell at some S*) -- blocked by frozen-in
- Q-ball-like on paper (E / content minimal at a finite size), BUT in step() + Cartan the vorticity distribution is a
  Casimir (Lie-Poisson): core area, circulation and impulse are fixed by the IC. H can choose the shape on that leaf
  (Arnold), never the core density or the ring size. A different G only reshapes the stable state, cannot select size.
- Consequence: in this rule a "default" exists only for shape relative to IC-fixed invariants; size and content are
  whatever was seeded. Same root as 10-21 (valve needs smooth frozen-in breaking) and 10-14 (off-quantum held exactly).

## What would give spongey + default (candidate, not built)
Taylor-type relaxation: frozen-in broken smoothly at small scale while a robust invariant (helicity / linked
circulation) is kept -> relaxes to the minimum-energy state for that invariant (Beltrami / spheromak lineage, steady
at rest: omega x v = 0 makes the Cartan term vanish and div-free phi with u = 0 is static under step()). The released
energy must go somewhere in a conservative rule -> into u (offload). A free minimum-energy state at fixed helicity
expands (E ~ K / R) -> it needs an outside agent to set size (mem force-free-needs-external-stress) = the depression /
background pressure. So: default = relaxation (needs frozen-in breaking) + size = depression (needs a confining
pressure). Both missing pieces are the same two already flagged separately.

## Open design questions for Philip
1. Frozen-in breaking that keeps helicity and sends the released energy into u (conservative), smooth at structure
   scale -- what in the substrate does it (cell gradient / L averaging?) rather than a picked diffusion.
2. Depression as pressure (energy, outside the body) vs as rate (sigma) -- route A says rate cannot hold a size.
3. Velocity as a free DOF: a Kelvin ring's speed is slaved to its size and it cannot sit at rest; an electron can.
   A Beltrami / spheromak body is static at rest -> better electron candidate class than rings / coils.
