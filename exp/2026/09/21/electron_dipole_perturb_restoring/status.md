# Perturb-and-check-restoring-force test on the double-ring dipole electron IC

Queued since 09-17 (`project_chirality_mixing_two_component_mass`,
`project_dst_electron_dipole`): every free-electron soliton probe to date
(v1-v8b, exp/2026/09/16-19) used a single-ring IC, already flagged as the
wrong structure for the Dirac chirality-mixing/"two coupled components
needed for mass" argument. The double-ring dipole (`init_electron_dipole_ic
_vecpot`) is the right structure, but it's already an EXACT static fixed
point at lam=0 -- nothing to test by just running it. This session built
and ran the actual perturb-and-check design.

## Result: negative, but for an interesting, diagnosable reason -- not "the
dipole is inert," but "this test never actually looked where confinement
would show up"

Three real methodological bugs found and fixed in sequence, each one worth
keeping as a record for any future perturbation test on this IC family:

**Bug 1 -- degenerate perturbation.** First attempt: `phi *= 1+eps*tanh(z/w)`,
a per-CELL multiplicative reweighting between the two lobes. This does NOT
break divergence-freeness: `div_i(phi) = sum_k W_k*phi[i,k]` is linear in
phi with one scalar-per-cell multiplier, so the multiplier factors straight
out of the per-cell sum regardless of how it varies in space --
`div_i(f_i*phi) = f_i*div_i(phi) = 0` for ANY f, not just a uniform
rescale. Confirmed directly: the "linear control" run was frozen
bit-for-bit (H unchanged to printed precision, 3000 steps) -- the state
never left the fixed-point family. Fixed by switching to a localized
additive bump on u instead (`u += eps*exp(-0.5*|pos-center|^2/w^2)`),
which breaks `du=0` unambiguously.

**Bug 2 -- background contamination.** A single-world "energy retained in a
window" diagnostic read ~99.97% retained regardless of regime -- but the
window (r<=4.05, centered on the bump) almost entirely overlapped the
dipole's OWN static field energy (~154.7, itself frozen since it's the
still-present fixed point), which completely swamped the tiny bump's own
energy (~0.044). Fixed by running each regime as a PAIR (with-bump state A,
reference no-bump state B, identical operators) and diffing the fields --
isolates exactly the bump-induced excess energy. Confirmed the underlying
wave dynamics work normally in bare vacuum first (bump fully disperses
E_bump: 0.044 -> 1.4e-12 by step 500) before trusting this on the dipole.

**Bug 3 -- boundary-reflection contamination at late times.** With bugs 1-2
fixed, E_bump(t) in the paired-diff read drops cleanly toward a floor by
t~10 (matches the vacuum-only dispersal timescale) but then RISES again
for every regime including the linear control, by the run's end (t=15.38).
Round-trip light-crossing time for this domain (R=30, source at r~2.75) is
~17 time units -- the "rise" is the wave's own reflection off the
(partially absorbing, damp_max=0.003) sponge coming back into the window,
a pure domain-size artifact unrelated to the dipole. The MINIMUM of
E_bump(t), read before this contamination arrives (t~10-12.5), is the
correct "how much genuinely dispersed away" diagnostic, not the endpoint.

## The result that looked promising, and why it didn't survive a spatial check

At the minimum (t~11-12.5), the floor is monotonic and clean across regimes:
linear=1.6e-12, step_cross=4.7e-7, lam_relax(a=0.5)=8.3e-7, (a=1.0)=3.4e-6,
(a=2.0)=1.4e-5 -- a 5+ order of magnitude, alpha-monotonic separation from
the linear control. This LOOKED like exactly the restoring/confinement
signature the whole free-electron thread has never found.

**It doesn't hold up.** Rerun at the lam_relax(a=2.0) minimum (step 2450),
saving the full spatial A-B diff field and binning excess energy by radius
from the bump: total excess energy across the WHOLE domain is 0.0123 (a
sizeable fraction of the original 0.044 bump, as expected -- energy hasn't
reached the sponge yet). Of that, only ~1.5e-5 sits within r<4 (near the
dipole/bump); the overwhelming majority (0.0123, essentially all of it) is
out at r=10-25 -- the ordinary outgoing wavefront, exactly where a freely
dispersing wave should be at that time, nothing localized near the source.

**Conclusion**: the window-level "signal" comparing regimes was not
localized confinement of the bump by the dipole structure. It's broadband
nonlinear divergence between the two paired trajectories (bump-present vs.
bump-absent) as `step_lam_relax`'s `lam_target=alpha*|phi|` coupling
responds slightly differently to the dipole's OWN large background field in
each branch once nonlinearity is on -- a generic sensitivity effect, not
evidence of a restoring force specific to the perturbation's location. Per
[[feedback_stability_not_reality]]/[[feedback_mechanism_not_description]]:
a clean, monotonic-looking number is not itself evidence of a real
mechanism until checked against where the effect actually lives spatially
-- this is exactly that check, and it failed.

**This is a negative result about THIS test design (small local bump,
`step_lam_relax` nonlinearity, R=30 domain), not evidence against the
double-ring dipole / chirality-mixing hypothesis itself** -- same framing
already established for the single-ring v5 degenerate-IC negative result.

## v2 (same day, Philip's correction): static/invariant lam instead of step_lam_relax -- real, spatially-confirmed partial-trapping signal

Philip's objection to v1: for an electron "at rest," lam shouldn't be a
dynamically-relaxing field chasing `alpha*|phi|` every step -- that's
exactly what made v1 uninterpretable (see the memory above: at
alpha=0.5-2.0, `step_lam_relax` was found to drive sigma down 15-70%
across the WHOLE dipole envelope within ~10 time units, so the two paired
worlds were diverging from independently rewriting their own propagation
speed, not from anything localized). `init_electron_dipole_ic_vecpot`
already has a `lam_target` parameter that sets a STATIC lam profile once
from `|B|=|curl(phi)|` at IC construction and never touches it again --
this is the "lam invariant" construction. Rebuilt the test around it
(`perturb_restoring_v2_static_lam.py`): same paired bump/no-bump diff, same
bump, but now plain `step()+sponge` only (no `step_cross`, no
`step_lam_relax`) and BOTH paired worlds share the identical static
sigma(pos) field -- any difference between them can only come from the
bump itself.

**Result: real this time, confirmed by the spatial check that killed v1's
apparent signal.** At t=2.56 (step 500, well inside the clean window,
nowhere near the ~11-17 time-unit boundary-reflection contamination), the
fraction of the bump's excess energy still within r<4 of the structure is
monotonic in lam_target: 0.00% / 0.50% / 2.49% / 19.21% for lam_target =
0 / 0.5 / 1.0 / 2.0. The full E_bump(t) window trace is qualitatively
different in shape from v1's false positive: lam_target=2.0 drops from
0.044 to ~3.2e-4 by t~7 then genuinely PLATEAUS (3.20e-4 -> 3.20e-4 ->
3.21e-4, t=7.2-9.2) instead of continuing to decay toward zero the way the
lam=0 baseline does (reaches 1e-12 by t=10); lam_target=1.0 shows the same
shape, smaller and shorter-lived (flat ~4.5-4.8e-5 over t~5.6-7.2, then
resumes decaying by t~9-10); lam_target=0.5 shows a faint version of the
same.

**Honest scope**: this is ordinary wave-in-a-medium physics (a region of
reduced local propagation speed, sigma_min down to 0.33 at lam_target=2,
partially reflects/delays a wave crossing it) -- a real, structure-shaped
effect, not evidence specific to any electron/mass-mechanism hypothesis by
itself. It is also PARTIAL and, in the data available (limited to t<10-11
by the domain's own boundary-reflection artifact), still leaking rather
than settling to a permanent nonzero floor -- "temporarily/partially
confining," not "bound." A larger domain (to push the reflection-
contamination window further out) is needed to see whether the plateau
eventually decays fully away (leaky/metastable) or truly saturates
(a real bound state) -- not yet done.

Files: `perturb_restoring_v2_static_lam.py`, `run_v2.log`,
`results_static_lam.npz`.

## v3 (same day): free (unpinned) u-deficit -- charge does NOT persist in any tested config, including with static lam

Following the ontology discussion: doc/dst.md's canon is proton=high-u
open knot (charge), electron=u-deficit -- never implemented in any tested
IC (the dipole's u is exactly flat, confirmed). Rather than PINNING a u
source cell (yesterday's Coulomb test's method, flagged by
[[feedback_pinning_signals_aphysicality]] as a red flag), seeded a FREE,
unpinned local u-deficit at the dipole's own center and let it evolve
(`electron_charge_test.py`). Four configs: vacuum, dipole phi + lam=0,
dipole phi + static lam_target=1.0/2.0 (today's v2 partial-confinement
regime).

**First read of the endpoint (t=15.38) looked informative -- and was
wrong, same mistake as v1.** The raw summary ("retained" fraction) is
dominated by the SAME boundary-reflection contamination flagged in v1/v2
(round-trip ~11-17 time units on this R=30 domain) -- pulling the full
time series confirmed it: ALL FOUR configs show the deficit oscillating
and decaying toward the noise floor through t~8-11 (vacuum/dipole_lam0
reach ~1e-5, a ~1e6x reduction from the initial 9.73, by t~8), then the
signal swings back up to O(1-3) purely from the reflected wave returning
by t~14-15 -- not from anything physical persisting near the structure.

**Also confirmed exactly**: vacuum and dipole_lam0 (phi present, lam=0)
give BIT-FOR-BIT IDENTICAL deficit traces at every single time point --
direct numerical confirmation that at lam=0 (uniform sigma), the linear
PDE gives EXACT superposition: the dipole's own exactly-divergence-free
phi has zero effect on how a nearby u-perturbation evolves. The dipole's
structure only starts to matter once lam is nonzero (dipole_lam1/lam2):
the deficit's oscillation amplitude and timing genuinely change (bigger
swings, slower apparent frequency, consistent with locally reduced sigma)
but STILL decays toward the noise floor within the clean window (before
t~10-11) in every case -- no config shows a persistent, nonzero,
non-oscillating plateau the way the earlier bump-in-phi (v2) test did.

**Conclusion, stated as plainly as the result warrants**: a free,
unpinned local u-deficit does not persist as a charge-like feature in this
construction -- with or without the dipole's phi structure, with or
without a static, structure-shaped lam profile. This is itself evidence,
not a failed test: the double-ring dipole IC, as currently built, gives no
mechanism for genuine self-sustained charge. Getting a persistent u
signature would need either (a) a genuine sourcing/feedback loop tying u's
sign to something in the structure (not yet identified/built), or (b) an
explicit pin (already known, from yesterday, to work -- but flagged as
non-physical by [[feedback_pinning_signals_aphysicality]], not a real
demonstration).

Files: `electron_charge_test.py`, `run_charge_test.log`,
`results_charge_test.npz`.

## Not yet done / possible next steps

1. A genuinely localized-confinement check would need either a much larger
   domain (so the reflection-contamination window opens later, giving more
   clean observation time) or an explicit "where is the excess energy"
   radial-profile diagnostic tracked over TIME (not just one late snapshot)
   to see whether ANY nonlinear regime ever produces a spatially-peaked
   (not delocalized) retention near the source -- not attempted this
   session for any regime besides the one spot-check above.
2. A cleaner perturbation that couples more directly to the lobe-balance
   question (the actual chirality-mixing motivation) rather than a generic
   local u-bump -- still not built; the per-cell multiplicative approach
   (bug 1) was the natural first idea for that and is now known not to work
   for reasons unrelated to physics (pure discrete-divergence algebra).
   A genuinely non-divergence-free lobe perturbation would need a per-FACE
   (not per-cell) reweighting, or a direct phi ADDITIVE term shaped like
   the antisymmetric-in-z part of a DIFFERENT k/mode than the IC's own.
3. `step_cross`'s own effect (0.77% "retained" in the first, buggy
   window-only read) was not spatially checked either -- given the
   background-contamination and delocalization findings above, that number
   should be treated with the same skepticism until similarly checked.

Files: `perturb_restoring_v1.py` (the final, bug-fixed paired-diff version;
git history has the two earlier broken attempts if ever needed),
`run_v1.log`, `results_bump.npz` (time series per regime).
