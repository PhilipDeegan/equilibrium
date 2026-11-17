# L8 AMR safety check — is the deep telescope safe for the real refined spheromak run?

Goal: decide whether it's safe to run the real, calibrated `refined_neutron_decay.py`
production configuration (R=100, max-level=8 — the actual Bohr-radius/neutron-diameter
target) without burning a lot of wall-clock time finding out the hard way.

Context: [[project_odd27_amr_junction_leak]] (Part 2) had already found a severe,
depth-accelerating H-energy growth at L6/L8 in the Odd27 AMR telescope, and — the
same day, later — a recursive/interleaved Berger-Oliger subcycle rewrite
(`_step_subcycle_recurse`/`_step_subcycle_one` in `adaptive_lattice.hpp`) landed,
fixing the "coarse-first double-stepping" mechanism that had been the leading
hypothesis for the cause. This session's question: does that fix (plus the two
earlier multiplicity fixes) actually make L8 safe now?

## Method

Direct full L8 production settings (R=100) are too expensive for a quick check —
a single L8 step at R=100 (5.58M leaves) took over 8 minutes without finishing.
Instead, used the matched-footprint technique from Part 2: `--scale-override
1.3606e-3 --scale-ref-level 8` holds the spheromak at the same physical (sim-unit)
size regardless of `--max-level`, and a much smaller R=20 domain (~144k-180k
leaves total) keeps the base/telescope mesh cheap while still testing the same
physics and the same coarse-fine junction structure. `--report 1` prints every
step's H/H0 (not just first/last) so the trajectory *shape* is visible, not just
an endpoint.

## Pitfall hit first (kept as a methodology note)

First attempt used R=20 *without* `--scale-override`. The script's default
calibration ties the neutron's sim-unit `scale` to `R` when uncalibrated
(`scale = NEUTRON_RADIUS_M / (R_DOMAIN_M / R)`), so shrinking R shrinks the
object in sim-units too — at R=20 this left only 27 leaf cells (~2 cells across
the diameter) resolving the spheromak. The resulting H/H0 swings (130-700x
across just 5 steps, `L8_INVALID_no_scale_override.log`) are IC-under-resolution
noise, not a real AMR signal — a reminder that comparing across R without
locking the physical footprint invalidates the test. Discarded; not used in the
verdict below.

## Results (valid, matched footprint, R=20, scale locked to L8=1.3606e-3 sim-units)

L6 (`L6_matched_footprint.log`), 10 steps:

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| H/H0 | 3.65 | 9.62 | 15.40 | **18.58** | 18.26 | 15.04 | 10.46 | 6.73 | 5.39 | 6.20 |

Single early peak (step 4), then settles into a ~5-10x band. Matches the
qualitative shape of the earlier 40-step L6 trace in the Part 2 investigation
(peak1 ~23.7x, settling band ~11-19x) — same character, this run's absolute
numbers differ somewhat (different R, dt, step count) but the "bounded
oscillation, not runaway" signature is consistent.

L8 (`L8_matched_footprint.log`), 10 steps:

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| H/H0 | **130.8** | 127.6 | 40.7 | 79.0 | 77.8 | 91.0 | 111.9 | 106.1 | 100.4 | 100.8 |

Peak arrives immediately (step 1 — no build-up the way L6 has), then oscillates
in a persistently elevated ~77-131x band, still ~100x at step 10 with no sign
of decaying within this window.

## Verdict

**L8 is NOT safe for the real refined spheromak production run as of
2026-07-20.** The bounded-oscillation qualitative signature holds at both
depths (this is not a divergence-to-infinity bug), but the band amplitude is
~7-10x worse at L8 than L6 for just 2 more refinement levels — consistent with
Part 2's "accelerating faster than linear with depth" characterization, and in
the same range as that investigation's earlier ×96-98/20-step sanity probe.
None of the fixes landed so far (`face_multiplicity`, `ancestor_group_multiplicity`,
`.lam` restriction, the recursive Berger-Oliger subcycle rewrite) close this
gap — the recursive-subcycle fix in particular was verified to have only a
~1.7e-8 relative effect on this magnitude, confirming it real but not dominant.

Practical guidance: L2/L4 remain safe. L6 is a real, substantial (~5-19x) but
*bounded* energy inflation — usable only if that magnitude is acceptable for
whatever's being measured. **Do not attempt the calibrated R=100/L8 production
run** until the still-open structural cause (localized to same-level flux at
coarse-fine boundary cells specifically, per Part 2's
`disable_same_level_flux_at_boundary_only` finding) is actually found and fixed.

## Side question raised during this check: no Fourier/von-Neumann counterpart to the CFL

Prompted by "we have a CFL, but do we have a resistive Fourier counterpart?" —
checked the code directly. `TimeStepper`'s `by_level`/`cfl` presets give the
wave/hyperbolic part (`step_subcycle`'s u/phi pass1/pass2 system) a real,
structurally-enforced CFL (level `l` gets `ratio^l` substeps of `dt_base/ratio^l`).
No equivalent exists for the resistive/diffusive channels:

- `step_cross` (`adaptive_lattice.hpp:930`, explicitly called a "resistive
  throttle" in-code) is applied **once per outer coarse step**, not through
  `step_subcycle`'s per-level recursion at all. Its only level-awareness is a
  **linear**-in-`cell.L` amplitude correction (`delta = -dt * cell.L * scale *
  si2 * inward`) intended to "reproduce the same per-level fraction" as the
  wave CFL — but a genuinely diffusive process's stability bound is quadratic
  in cell size (Fourier/von Neumann number `D*dt/L^2 <= const`), not linear.
- `step_lam` has **no level-dependence at all** — flat outer `dt` applied to
  every leaf regardless of `L`. This is a strictly worse gap than
  `step_cross`'s: `step_cross` at least has a (wrong-power-law) attempt at
  per-level scaling, `step_lam` has none — a level-8 leaf and a level-0 leaf
  currently relax at the exact same rate per outer step, even though `sigma()
  = 1/(1+lam)` feeds directly back into the wave part's flux weighting
  (`_step_pass1_lev`/`_step_pass2_lev`) at every level. If lam is meant to
  represent a per-cell physical relaxation/capacity (per `project_lam_dual_role`
  — mass/tension AND phi-holding capacity) that should scale with cell size the
  same way the other per-cell dynamics do, this is a straightforward,
  unambiguous bug (not just a missing-refinement judgment call like
  `step_cross`'s linear-vs-quadratic question) — a level-8 leaf is ~6561x
  smaller than a level-0 cell yet integrates `lam` at an identical rate.

Caveat: this is NOT implicated in the L6/L8 numbers above — Part 2's isolation
(disabling `step_cross`/`step_lam`/sponge one at a time) found the runaway
persisted identically with all three off, pinning it to `evolver.step()` alone.
So this gap doesn't explain today's result. But it's a real, separate,
unexamined risk that would bite hardest exactly where depth is now being
pushed (L8's `L^2` is ~9x smaller than L6's, ~600x smaller than L4's) — worth
deriving the actual diffusion-number bound for `step_cross` (using
`sigma^2 * cross_scale` as the effective diffusivity) and checking whether
current defaults already violate it at L6/L8, before trusting `cross_scale`/
`lam_scale` dynamics at those depths even after the wave-part issue is fixed.
Not investigated further this session — flagged for next time.

## Artefacts

- `refined_neutron_decay.py` — copy of the script at time of run (includes a
  small addition made this session: per-report-step `H`/`H/H0` print inside
  the main step loop, so the full trajectory is visible in the log, not just
  first/last).
- `run.sh` — exact commands (including the discarded invalid attempt).
- `L6_matched_footprint.log`, `L8_matched_footprint.log` — the two valid runs.
- `L8_INVALID_no_scale_override.log` — the discarded under-resolved run, kept
  for the methodology note above.

## Next steps

1. L2/L4 through this same matched-R=20 harness, for a full 4-point depth
   curve in one consistent methodology (previous L2/L4 numbers came from
   earlier, differently-configured tests) — useful as a clean before/after
   baseline once the structural fix is attempted.
2. Root-cause the same-level-flux-at-boundary anomaly (Part 2's localized but
   not-yet-explained mechanism).
3. Derive and check the diffusion-number (Fourier) stability bound for
   `step_cross`/`step_lam`, independent of the wave-CFL work above.
