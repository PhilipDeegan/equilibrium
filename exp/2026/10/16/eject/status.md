# 2026-10-16 eject -- accept-and-eject relay (Philip: cells prefer no local perturbation), void context

eject.py: per-face port amplitudes, every cell ejects everything each tick, nothing stored. P 96 (221k cells).
- straight (pass to the opposite face): exact, but pencil beams along the 14 lattice axes only (anisotropic).
- huygens (TLM shunt junction b = 2 sum w a / sum w - a): energy exact (1e-12 after 60 ticks).
  Smooth pulse (gaussian width 3): clean outgoing spherical shell, speed 0.578 cells / tick (lap weights; 0.59
  equal weights), isotropic to <1 % (square / hex / edge directions at t 40: 24.16 / 24.38 / 24.33), shell
  width constant (3.6), nothing left behind. Wave speed < hop speed (like TLM / checkerboard).
  Point pulse (one cell): 65 % of the energy stays within t / 3 -- lattice-scale content has ~zero group velocity:
  a cell-sized perturbation is NOT ejected cleanly; it lingers.
Scalar (one amplitude per face) = one polarisation. Maxwell would need two per face (TLM condensed node analogue).
Next: context (rate / cell-size gradient -> bending), two-polarisation node, closed ejection loop.

## Throughput from cell conditions (bend.py)
Philip: relay is axiomatic; throughput set by local cell conditions.
- Accumulator version (cell ejects only when its accumulator fills at rate sigma; arrivals held meanwhile): energy NOT
  conserved (x2.7 by t 60) -- held arrivals superpose on one port. Dropped.
- TLM stub version (each cell also ejects into an internal port of admittance y that returns next tick; slower where y
  is larger): energy exact (1e-15). Gradient run (dy/dx 0.018): forward-lobe drifts toward the slow side (+0.28 vs
  +0.07 at g 0), right sign; launch too messy (scalar pulse splits both ways) to read a magnitude.
- Uniform y, spherical pulse: y 0 shell 0.52 cells / tick; y 1 / 3: a large share stays at the origin oscillating
  between ports and stub (E within r < 5: 0.34 / 0.71 at t 40; stub energy 30-60 %). Wider pulse (width 6, P 140, y 1):
  trapped share smaller (0.13 at t 60) but growing, shell still leaves at ~0.5. Not clean: the internal port creates a
  slow / local branch. Either lattice-scale (TLM stub spurious mode) or a real "held" branch -- open. Needs the dispersion
  relation of node + stub on the 14-face cell before more runs.

## Dispersion (disp.py, modes.py) -- source-free
15 branches per cell (14 ports + stub):
- 2 travelling: speed 0.603 cells / tick, isotropic to 3 decimals ([100] [111] [110]), dispersion 0.595 at q 1;
  with stub y: 0.426 / 0.302 at y 1 / 3 = exactly 1 / sqrt(1 + y). The mode is V (pressure) + LONGITUDINAL net flux
  -> a scalar, sound-like wave (1 polarisation, longitudinal).
- 6 static (omega 0 at every q): V 0, flux purely TRANSVERSE, a ~ -a_mirror (straight-through flow). Steady
  divergence-free circulations that never move or radiate, at every scale: the held branch is real, not lattice-only.
- 7 at omega pi (flip every tick, a ~ +a_mirror, V 0): flat lattice artefacts. Most of the 'trapped' energy in the
  sphere runs (all ports equal = mirror-symmetric IC) went into these + the stub.
Real EM is the reverse: transverse waves, longitudinal (Coulomb) static part. The scalar junction gives a longitudinal
wave + static transverse circulation. Maxwell from one rule needs the transverse sector to propagate too (a node that
couples transverse port flux, cf TLM condensed node) -- that is where E and M from the same rule must come from.
