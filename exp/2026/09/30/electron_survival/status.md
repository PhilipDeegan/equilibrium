# Electron survival under self-sourced lam, 1a / 1b — 2026-09-30

**INVALID as structure evidence (Philip 10-01): lump widths 1-2 cells. Still valid (resolution-independent): the exact (alpha, amp) -> g reduction, the uniform-floor relabelling, and the Euler-vs-midpoint energy-error finding. An electron below R20 (radius 20 cells) is totally invalid, R20 maybe too small (mem feedback_structure_resolution). Survival / retention / response numbers below are records of the setup only.**

Follow-up of `exp/2026/09/29/electron_survival`. Rule: lam = eps + alpha (u^2 + 1/2 sum phi^2), discrete
step() with face sigma = mean cell sigma, DampedLattice rate sponge (0.25), R = 12. Script `self_lam_long.py`
(modes long / scan / scaling / energy; `--integrator euler|midpoint`, `--dt_frac`). Snapshots (full state)
in /bk/dst/2026/09/30/electron_survival/. Plot `long.png` (`plot_long.py`).

## Exact reductions
- (alpha, amp) -> (alpha/s^2, s amp) is exact (step() linear at fixed lam): only g = alpha amp^2 matters
  (--mode scaling: identical to 12 digits).
- A uniform additive floor is exact relabelling: (eps, alpha, dt) == (0, alpha/(1+eps), dt/(1+eps))
  (`floor_equivalence.py`, 3e-16). The Earth-Moon L1 background (lam ~ Phi/c^2: E+M 1.45e-11, Sun 9.87e-9,
  galaxy 5.4e-7) is therefore the void run with g x (1 - 5.5e-7). Low-not-void only matters if the floor is
  non-additive, non-uniform, or acts through something other than sigma.

## Integrator: the 09-29 / 1b numbers carry a first-order energy error
Symplectic Euler is not symplectic once sigma depends on the state. Continuous-time H is conserved exactly
(skew J(x), quadratic H), so implicit midpoint (sigma at the midpoint) conserves it identically:

| integrator | dt | max abs(H/H0 - 1), t < 100, dipole g = 100, R = 8, no sponge |
|---|---|---|
| euler | 1, 1/2, 1/4 | 4.8e-2, 2.3e-2, 1.2e-2 |
| midpoint | 1, 1/2, 1/4 | 3e-13, 3e-12, 8e-14 (5-7 fixed-point iterations) |

## 1a: long runs (200 crossings, t = 2079), width 1.5 — core energy / initial (r < 4.5)

| case | integrator, dt | t = 207 | 519 | 1039 | 2079 | end split u/grad/sol |
|---|---|---|---|---|---|---|
| gauss g = 36 | euler 1 | 0.200 | 0.198 | 0.172 | 0.148 | 0.22/0.01/0.77 |
| dipole g = 36 | euler 1 | 0.168 | 0.166 | 0.166 | 0.166 (frozen) | 0.00/0.03/0.98 |
| gauss g = 100 | euler 1, 1/2, 1/4 | 0.60, 0.57, 0.59 | 0.54, 0.48, 0.47 | 0.48, 0.42, 0.42 | 0.444, 0.384, 0.333 | (1/4) 0.32/0.05/0.64 |
| dipole g = 100 | euler 1 | 0.568 | 0.469 | 0.472 | 0.528 (energy growing) | 0.13/0.27/0.61 |
| dipole g = 100 | euler 1/2, 1/4, 1/8 | 0.56, 0.53, 0.56 | 0.44, 0.36, 0.44 | 0.43, 0.27, 0.32 | 0.398, 0.227, 0.277 | (1/8) 0.01/0.05/0.95 |
| dipole g = 100 | midpoint 1, 1/2 | 0.556, 0.556 | 0.338, 0.365 | 0.299, 0.256 | 0.282, 0.210 | 0.10/0.05/0.86, 0.08/0.03/0.89 |

Reading:
- The dt = 1 Euler rise (dipole g = 100) is integration error (gone at dt/2 and with midpoint).
- Even with exact energy the late value is not dt-converged (midpoint 0.28 vs 0.21): trajectories agree to
  t ~ 200 and separate after ~500 -> sensitive dynamics. Only a band can be quoted: 0.2-0.3 at g = 100.
- In every long run the survivor is >= 86-98% divergence-free phi = circulation manufactured by the varying
  sigma and then frozen (step() cannot move it). By criterion 1e this is NOT survival of a structure.
- Core sigma stays 0.1-0.3 (self time-dilation); the propagating part keeps leaking.

## 1b: fine scan in g (12 crossings, Euler dt = 1 -> magnitudes biased ~10-20%, see above)
gauss / dipole, width 1, 1.5, 2, g = 2..200 (`scan_*.log`, `scan_rows_*.npy`):
- No threshold, no preferred g or size: retention < 1e-3 below g ~ 6 (all frozen circulation), rises smoothly
  to 0.8-0.87 at g = 200; core r_rms follows the initial width.
- Retention is a function of core sigma (w = 2 gauss: sigma 0.26 -> 0.18, 0.10 -> 0.60, 0.04 -> 0.87):
  clock slowdown, not a balance. Width 1 (about one cell) is under-resolved and non-monotonic.
- Not rerun with midpoint (numpy midpoint ~25-40 min per long run; C++ port needed for sweeps).

## Verdict (conditional test: cold vacuum, absorbing sponge, void lam)
In a cold, absorbing (sponge) vacuum, self-sourced lam gives no surviving electron candidate: slow leakage
under self time-dilation, ending as frozen circulation, with no threshold or preferred e/m.

## Scope (Philip 09-30)
- This background has zero inward pressure by construction. Decay may mean the background is wrong
  (particles held by focusing pressure from the background), not that the structure fails.
- Formation may need a proton's landscape (Coulomb field extent). Free-electron persistence at low
  (not void) lam remains a criterion.
- H2 being the default (over H1) suggests a p/e mismatch; net neutrality ~1e-21 e -> the mismatch is in field
  shape/extent.

## Next
Thermal ambient background (ambient.thermal_state + reciprocal BC, T scan, 2-3 realisations, midpoint):
the bath sets its own lam floor, and the low-sigma core lenses ambient waves inward. Retention vs T,
measured as excess over the bath mean. Port the midpoint step to C++ if it becomes a sweep.

## Proper-time bookkeeping (next 10-01 item 2; `proper_time.py`, `proper_time.png`, from saved records, no reruns)
tau_core = int sigma_core dt, sigma_core = core-energy-weighted 1/(1+lam). Ecore/E0 at tau = 0.5 / 1 / 2 / 3 proper
crossings (w 1.5 scans, 12 crossings; long runs, 200 crossings):
- gauss: g 8 -> 0.20 / 0.16 / 0.10 / 0.09; g 25 -> 0.42 / 0.19 / 0.15 / 0.15; g 70 -> 0.54 / 0.43 / - ; long g 100 ->
  0.71 / 0.63 / 0.59 / 0.56, 0.33-0.44 at tau ~ 21 (dt-dependent); long g 36 -> 0.20 flat to tau 18, 0.15 at tau 45.
- dipole: g 8 -> 0.13 / 0.016 / 0.007; g 25 -> 0.40 / 0.33 / 0.24 / 0.21; long g 100 -> 0.73 / 0.66 / 0.58 / 0.55,
  0.21-0.53 at tau 31-49 (dt-dependent); long g 36 -> 0.17 flat from tau ~ 5 to 90.
Reading:
- NO collapse onto one curve in proper time: at fixed tau, retention still rises with g. So the fixed-t retention is not
  just a slowed clock (that reading is confined to the gauss a1000 potential case, sigma 0.036, ambient_flux_lam).
- The long-run plateaus are flat in proper time (dipole g 36: 0.166 over ~85 proper crossings). A plateau that does not
  age with its own clock is a zero-frequency component, consistent with the 1a split (86-98% frozen divergence-free
  phi): a static circulation left in place, not a bound oscillating structure. Still 1e: manufactured circulation.
- Non-monotonic in g at small g (gauss g 8 > g 10-15, dipole g 12 > g 15-18) and dt-dependent late values: sensitive
  dynamics, the plateau level is not a property of g.
Verdict (conditional, cold vacuum, self-sourced lam): proper time removes the "frozen clock" loophole here; what is
left is frozen circulation, already excluded by 1e.
