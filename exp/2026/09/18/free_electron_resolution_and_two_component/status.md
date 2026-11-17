# Two-component IC + resolution check — status (2026-09-18)

## Context

Follow-up to `exp/2026/09/17/free_electron_soliton_alpha_scan/status.md`'s
queued next steps: (1) a resolution check before trusting any alpha* claim,
(2) a genuinely non-degenerate two-component IC, (3) rerun (6,14) at larger
domain clearance.

## Scripts

- `v7_two_component.py` — queued step (2a): +n/-n companion-phi branches
  summed with a PHASE OFFSET (not v5's phase=0, which is proven and
  reverified here to reduce exactly to v1/v5's degenerate phi=0 IC --
  `_verify_degenerate_limit()` asserts max|phi|<1e-10 at phase=0 before
  every run). (n=2,r=10), 3 phases (pi/8, pi/2, pi) x 9 alphas.
- `v6_resolution_check.py` — queued step (1)+(3) combined: swept the ring
  envelope's `width` (2.0 vs 4.0) at (2,10) and (6,14), with `R_domain`
  matched to keep ~50% relative clearance at every (config,width) --
  simultaneously testing whether wider domain clearance fixes (6,14)'s
  anomaly. **Turned out NOT to be a real numerical-resolution test** --
  see Interpretation below.

Both self-contained (same physics/IC pattern as v4/v5), `N_WORKERS=6`.
Results: `v7_two_component_results.npz`, `v6_resolution_check_results.npz`.
Full console output in `v7.log`/`v6.log`.

## Results

### v7 (two-component, phase-offset)

| phase/pi | alpha=0 grow% | alpha* | slope behavior |
|---|---|---|---|
| 0.125 | +51.96% | 0.858 | **crosses into negative alongside grow% near alpha~0.9-1.1** (first-ever diagnostic agreement) |
| 0.500 | +54.45% | 0.792 | stays positive at every alpha (v4's problem pattern) |
| 1.000 | +7.70% | 0.725 | noisy/non-monotonic (u amplitude is exactly 0 at this phase) |

**phase=0.125*pi is the first config across v1-v7 where `spread_growth` and
`slope_frac_per_period` agree in sign** -- around alpha~0.9-1.1 both go
slightly negative together (grow -2.5% to -15%, slope -0.35% to -0.8%).
Not a clean flat plateau (slope overshoots negative), and windowed decay
is still large (84-86%, not reverified against whole-domain energy
conservation). Larger phase offsets (0.5*pi, pi) don't show this -- the
effect is specific to being close to, but not exactly at, the degenerate
limit.

### v6 (width sweep, matched-clearance domain)

| n | r | width | R_dom | alpha=0 grow% | alpha=0 slope% |
|---|---|---|---|---|---|
| 2 | 10 | 2.0 | 36 | +7.10 | +1.48 |
| 2 | 10 | 4.0 | 52 | -13.60 | +0.30 |
| 6 | 14 | 2.0 | 44 | -22.79 | +5.19 |
| 6 | 14 | 4.0 | 60 | +18.14 | +1.71 |

(full alpha=0.9/1.3 rows in `v6.log`)

**(6,14) boundary-proximity guess weakened**: even at matched ~50%
clearance (up from v4's tight ~19%), width=2 still shows the grow/slope
sign disagreement v4 first flagged (grow=-22.8%, slope=+5.2%). Only at
width=4 do they agree (both positive). Giving it proper clearance did not
fix the anomaly -- it is NOT simply a too-close-to-the-sponge-edge effect.

## Interpretation

**v6 is not a valid numerical-resolution test, and does not answer the
09/17 session's queued question.** Two separate corrections, both from
Philip:

1. Comparing FRACTIONAL `spread_growth` across different `width` at a
   FIXED number of periods is confounded by ordinary wavepacket
   dispersion physics: `T_pred` depends only on K=n/r, not width, so a
   narrower envelope (broader k-bandwidth, uncertainty-principle
   group-velocity spread) is expected to disperse faster fractionally
   than a wider one over the same absolute time -- independent of
   numerical resolution. The width=2 -> width=4 sign flip at (2,10) is at
   least partly (maybe entirely) real physics, not evidence of
   under-resolution.
2. **Deeper issue**: `width` is an IC-shape parameter, not a mesh-density
   parameter at all. A real mesh-refinement check needs to vary `spacing`
   (`SubstrateLattice`'s real cell-center-distance parameter,
   `equilibrium/substrate/lattice.py:181`) while holding the ring's
   PHYSICAL size fixed (not cell-index size) and scaling `radius_cells`
   up to compensate -- v4 and v6 both always ran at the default
   `spacing=1.0`, so "cell count" and "physical size" were accidentally
   identical throughout and nothing ever tested cell density independent
   of object size. This is buildable now (no blocker) -- just not what v6
   did. See [[project_L_ell_not_wired]] for the related (but distinct,
   initially over-scoped) finding that doc/dst.md's `L`/`self.ell`
   per-cell coarse-graining variable is a stub with zero effect on
   `step()`/`stable_dt()` anywhere -- that specifically blocks variable
   (AMR-style) local coarse-graining, not uniform mesh refinement, which
   the `spacing` route above sidesteps entirely.

One number from v6 that ISN'T fully explained by the dispersion-rate
confound: slopes at width=4 sit much closer to zero (+0.3%, -0.28%,
-1.04%) across every alpha than width=2 or v4's original baseline (+1.3%
to +2.3%, always positive) -- worth another look once a genuine
spacing-based check exists.

## Next steps (queued for 2026-09-19+)

1. **Finer phase scan on the two-component IC**, 0 to 0.25*pi, around
   (2,10) -- the 0.125*pi result is the most promising single data point
   in this whole thread (v1-v7) and hasn't been localized/refined at all.
2. **Build a genuine mesh-refinement convergence check**: fix `r`/
   `width`/`R_domain` as physical (Planck-length) targets, sweep `spacing`
   (0.5, 0.25, ...), scale `radius_cells = R_domain_physical/spacing` to
   match. Reuse v4/v6's diagnostic pair (spread_growth + slope) to see
   whether either the single-component or two-component IC's numbers
   actually converge as spacing shrinks -- this is the resolution check
   the 09/17 session originally asked for and v6 didn't deliver.
3. **(6,14) anomaly is back to fully open** -- neither WKB-breakdown
   (ruled out 09/17) nor boundary proximity (ruled out here) explains the
   grow/slope sign disagreement. No new hypothesis yet.
4. Three ideas discussed this session, only two written to memory (Jellium
   was conversational only, not yet its own memory file -- write one if
   picked up): a Jellium/statistical-electron framing (parked, a different
   research track than single-soliton stability -- treat "electron" as a
   collective excitation of a diffuse background rather than an isolated
   knot); [[project_schwinger_offload_hypothesis]] (missing pair-creation/
   radiative offload channel as the real missing ingredient, not more
   alpha-tuning); [[project_dst_virtual_photon_bookkeeping_idea]] (leans
   "continuum QFT artifact," template is casimir.py's effects-not-mechanism
   approach if ever picked up). None started.
5. Separately, still open, unblocked by any of the above: the
   magnetic-moment k-eigenvalue question
   ([[project_dipole_moment_phi_convention_gap]]), and the electron u/phi
   convention conflict ([[project_electron_u_phi_convention_conflict]]) --
   worth resolving before trusting either the dipole-IC or ring-soliton-IC
   family's fixed points as "the electron."
