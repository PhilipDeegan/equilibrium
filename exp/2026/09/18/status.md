# Entries for 2026-09-18 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-18 — non-degenerate two-component IC (v7) gives the first real stability-diagnostic agreement in the whole v1-v7 thread; IC-width "resolution check" (v6) turns out NOT to be a valid numerical-convergence test (confounded by wavepacket dispersion physics, and `width` was never a mesh-density knob); (6,14)'s boundary-proximity explanation is weakened; four new ideas/corrections logged to memory

Continuation of 09/17's alpha-scan thread
(`exp/2026/09/17/free_electron_soliton_alpha_scan/`). Full results, tables,
and next-steps checklist: `exp/2026/09/18/free_electron_resolution_and_two_component/status.md`
(read that file before continuing this thread).

**v7 (two-component, phase-offset IC, `v7_two_component.py`)**: built the
queued non-degenerate two-component IC (+n/-n companion branches summed
with a PHASE offset, not v5's phase=0 which is reverified here to reduce
exactly to v1/v5's degenerate phi=0 case). At (n=2,r=10), phase=0.125*pi is
**the first config across v1-v7 where `spread_growth` and
`slope_frac_per_period` agree in sign** -- both go slightly negative
together near alpha~0.9-1.1, unlike every single-component config where
slope stayed positive regardless. Larger phase offsets (0.5*pi, pi) don't
show this. Most promising single result in the whole thread; not yet
localized/refined.

**v6 (IC-width sweep, `v6_resolution_check.py`)**: intended as the queued
resolution check + (6,14) boundary-clearance retest. Neither goal was
actually met: (a) `width` is a Gaussian-envelope shape parameter, not a
mesh-density parameter -- comparing fractional spread growth across
different widths at a fixed period count is confounded by ordinary
wavepacket dispersion physics (narrower envelope = broader k-bandwidth =
faster fractional dispersion, independent of resolution); (b) even at
proper ~50% domain clearance (up from v4's tight ~19%), (6,14) still shows
the same grow/slope sign disagreement v4 first flagged -- boundary
proximity is weakened as an explanation, not confirmed. **A real
mesh-refinement check needs `spacing` swept with `r`/`width`/`R_domain`
held fixed in PHYSICAL units and `radius_cells` scaled to compensate --
buildable now, not yet done.**

**Infrastructure finding + same-day correction**: doc/dst.md's `L`
(coarse-graining cell-size, "sigma=L" in vacuum) is declared/serialized
but has zero effect on `step()`/`stable_dt()` anywhere in the codebase,
Python or C++ (`inc/eq/substrate.hpp`: "ell unused", "ignored: no ell
update in discrete model") -- new memory [[project_L_ell_not_wired]].
Philip corrected the scope: this only blocks variable/AMR-style local
coarse-graining, NOT uniform mesh-refinement convergence in general, which
is independently testable via `spacing` (see v6 interpretation above) --
memory updated same day to reflect this.

**Four other memories written this session** (mostly from
conversation, not experiments): [[feedback_pinning_signals_aphysicality]]
(a field held externally fixed to force a construction into a fixed point
is a red flag, not a real demonstration -- sharpens
[[feedback_stability_not_reality]]); [[project_electron_u_phi_convention_conflict]]
(the dipole-electron IC pins u/lets phi carry structure, the ring-soliton
IC pins phi/lets u carry structure, and doc/dst.md's own canonical model
says u-deficit -- three disagree, still unresolved); [[project_schwinger_offload_hypothesis]]
(flagged: free-electron ICs may be failing to find a fixed point because
they lack any pair-creation/radiative offload channel, not because alpha
is mistuned); [[project_dst_virtual_photon_bookkeeping_idea]] (flagged,
leans "continuum-QFT perturbation-theory artifact," not pursued). A
Jellium/statistical-electron framing was also discussed (treat "electron"
as a collective excitation of a diffuse background, not an isolated
persistent knot) but not yet written to its own memory file.

**Queued next** (see status.md for full detail): (1) finer phase scan
0-0.25*pi on the two-component IC around (2,10) -- highest priority, most
promising open thread; (2) build the actual spacing-based mesh-refinement
convergence check described above; (3) (6,14)'s grow/slope disagreement is
back to fully unexplained -- neither WKB breakdown nor boundary proximity
account for it; (4) separately unblocked: the magnetic-moment
k-eigenvalue question ([[project_dipole_moment_phi_convention_gap]]) and
the u/phi convention conflict, both untouched this session.

---
