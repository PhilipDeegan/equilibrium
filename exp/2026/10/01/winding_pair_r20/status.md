# Winding pair at R20 — verification runs, 2026-10-01

Successor of `exp/2026/10/01/winding_pair` (invalid: tube 1.5 cells, and its helical seed was discontinuous on the
centre line, see below). Philip 10-01: structure ~40 cells across inside an R50-60 domain; first verify one case with
IC plot -> 10-100 steps -> HDF5 snapshot + plot, then decide the battery.

Setup (`run.py`, `plot_snap.py`): pair_opp (n, +1) + (n, -1), n = 3, r0 = 12, w = 8/3 (structure radius r0 + 3w = 20,
6w = 16 cells across the tube), R56 domain (955k cells), sponge from r = 42. Rule lam = alpha 2 e, C++ implicit
midpoint (`equilibrium/substrate/self_lam.py`, `tst/substrate/test_self_lam.py`), sponge factors in place.
Snapshots: ~/tmp/dst/2026/10/01/winding_pair_r20/<name>/state_*.h5 (no /bk on this machine). ~0.13 s/step at 8
threads = ~8 min per crossing.

## Fixes found by the verification
1. Seed discontinuity (v1 -> v2): chi = n theta + q psi winds q times around the centre circle, but the envelope was a
   plain Gaussian (maximal there) -> cos(q psi) jumps across the centre line (pair_opp u = 2 env cos(n theta) cos psi
   flips sign). Fixed: env_q = (d/w)^|q| Gaussian (d e^{i psi} = (rho - r0) + i z, smooth). The invalid w = 1.5
   battery had the same defect.
2. Plots: slabs |.| < 0.6-1.2 mix lattice planes (1/sqrt3 apart) at different phases -> false speckle once the pattern
   rotates. Now one plane (|.| < 0.3), linear griddata.

## Result (v3, 1000 steps, t = 13.3 = 0.27 crossings, ~1.4 wave periods)

| | alpha 8 | alpha 0 (linear) |
|---|---|---|
| grid-scale fraction of u in tube (residual vs 8 hex-neighbour mean) | 4.2e-3 -> 0.75-0.83, monotone | 4.2e-3 -> 7.8e-3 |
| k_eff^2 of u (1/2 sum W^2 du^2 / sum u^2) | 1.81 -> 22.6 (x12.5) | 1.81 -> 2.44, oscillating |
| winding mode |a_(1,1)| | 0.354 -> 0.005 | 0.354 -> 0.036 (dispersing) |
| peak energy density | 0.37 -> 1.9 | falls |

Checks: H conserved to 2e-16 (snapshots); midpoint 7-8 iterations; dt/2 reproduces dt to 4 digits over the first
100 steps (grid fraction 1.49e-2 both); travel direction and rate as seeded (phase rate ~ sigma omega).
Final state (`pair_opp_a8_R56_r12_w2.67_v3_final.png`): the torus has become two grid-width rings at the two lobes of
the (d/w) Gaussian (rho ~ 9.5, 15), u grid-scale noise, no winding pattern left.

## Reading (CONDITIONAL: this rule, cold vacuum)
Self-sourced lam with sigma = 1/(1+lam) is self-focusing with an unbounded index 1 + lam: energy slows its own region,
which bends waves into it, which raises the energy. A resolved structure collapses to the lattice scale within ~1
wave period at alpha 8 (Kerr-type 3D collapse; no saturation, so only the grid arrests it). The planned battery would
measure collapse, not pairing. The 09-29/30 "frozen circulation" survivors may be this collapse's end state on
structures that started near grid scale.

Open (Philip's call): short alpha scan (dispersal vs collapse, any window between?); saturating lam (bounded index);
or move to option (a) (two components in the rule, Bloch-symbol analysis first).

## alpha scan (`--tag _scan`, 2000 steps, t = 26.7 = 0.55 crossings; `plot_scan.py`, `alpha_scan.png`)

| alpha | sigma_tube(0) | grid frac > 0.1 at t (tau) | E_tube/E0 end | winding amp end | reading |
|---|---|---|---|---|---|
| 0.5 | 0.88 | 17.1 (16.1)* | 0.002 | 0.001 | disperses (grid frac ~1% while energy present) |
| 1 | 0.80 | 16.5 (14.3)* | 0.011 | 0.001 | disperses (grid frac ~7% plateau) |
| 2 | 0.69 | 6.9 (4.8) | 0.064 | 0.004 | both: ~20% grid scale, then disperses |
| 4 | 0.56 | 4.8 (2.6) | 0.42 | 0.024 | collapses; retained energy is grid scale |
| 8 | 0.42 | 4.0 (1.6) | 0.64 | 0.098 | collapses |
| 16 | 0.30 | 3.5 (1.0) | 0.83 | 0.079 | collapses |

\* after the tube has emptied: the grid fraction of a near-empty remnant is noise.
No window: weak coupling disperses, strong coupling collapses to the lattice scale within 1-3 proper time units
(well under one wave period, 2 pi / omega ~ 9); retention rises with alpha only because the collapsed grid-scale
remnant is slowed. The shoulder in E_tube and |a| at t ~ 8-13 for alpha <= 2 is most likely the inward-radiated part
crossing the torus hole (~24 cells at c ~ 2.3 ~ 10 time units) before leaving. Conclusion for this rule family
(instantaneous, local, unbounded lam = alpha 2 e): no resolved winding structure at any coupling.

## Fixed centred depletion battery, first 4 (10-01 evening; `--tag _bat`, lam0 4, 3 crossings, `plot_battery.py`)

Linear step_cpp, sigma on both legs (speed only). Plots: `battery.png` (E_struct, E_tube, |a|, grid fraction vs t and
proper time), `battery_phase.png` (winding phase), `*_bat_{ic,final}.png`; snapshots /bk/dst/2026/10/01/winding_pair_r20/.
3.6 min per run (0.9 min/crossing at 8 threads; the 25 min/run estimate in `next` was ~7x high).

| run | sigma_tube(0) | E_struct end | E_tube end | |a_(1,+1)| | phase rate early/late | grid frac end |
|---|---|---|---|---|---|---|
| tube helix | 0.50 | 0.050 | 0.045 | 0.50 -> 0.047 | -0.54 / -0.52 | 0.22 |
| tube pair_opp | 0.50 | 0.021 | 0.020 | 0.35 -> 0.017 | -0.55 / -0.56 | 0.53 |
| shell helix | 0.39 | 0.51 | 0.062 | 0.50 -> 0.053 | -0.45 / -0.31 | 0.13 |
| shell pair_opp | 0.45 | 0.20 | 0.026 | 0.35 -> 0.020 | -0.42 / -0.48 | 0.56 |

Reading (CONDITIONAL, R20 screening):
- No run traps a winding. All windings keep travelling (steady phase slope, never frozen) but their amplitude falls
  ~10-20x in 3 crossings; nothing levels off.
- tube (static analogue of the self-sourced rule): steady leakage, E_struct 0.05 / 0.02. The slow tube is a weak lens:
  no collapse (linear), no hold.
- shell: E_struct levels at ~0.5 (helix) / ~0.2 (pair), but the energy leaves the tube: it is guided by the slow
  spherical layer (a waveguide / SOFAR channel) and gathers at the poles (xz lobes at |z| ~ 10), with an E_tube
  revival at t ~ 100 (refocusing). Geometric-optics guiding of whatever enters the layer, not winding retention.
- Pair vs control: the pair loses MORE than the single winding in both shapes (no pairing benefit at equal energy).
- Grid-scale fraction grows to 0.1-0.6 (xy speckle in the slow region). The seeded wavelength is already ~4.7 cells
  (k_eff^2 1.8) and sigma 0.2 slows it further -> under-resolved in the depleted region; treat the late states as
  resolution-limited. lam0 1 (sigma 0.5) or longer seeded wavelengths would be cleaner before lam0 16.
