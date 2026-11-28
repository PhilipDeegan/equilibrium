# 2026-11-17

Repo housekeeping, no simulation runs.

- vis/: notebooks deduped into importable modules with thin notebooks; buk/, buk/srt/ and res/sort/ dissolved into physical/, math/, perf/.
- Agent memory moved out of the repo (was doc/mems). Conventions extracted to CONTRIBUTING.md and doc/principles.md.
- Repo recreated on GitHub as a single `init` commit (old one renamed equilibriumed, private, kept).
- CI: format gates (clang-format, black/ruff on changed code, iwyu, clang-tidy warning-only), drafts skipped, build split into build + 6 python shards + cpp (test_lattice now runs). ~13 min vs ~1 h. Floor: TestStepCrossUniformVsJunctionMismatch.test_uniform_level2_no_junction_stable (~10 min).
- mkn: -K for static deps, `mode: none` on final binaries/libs.

## Next
- Thermoelectrics / energodynamics: temperature and heat as bookkeeping for the diffused share of energy; Kelvin relation Pi = S*T as an identity target. Nernst (vortex version) as a later target. Prereqs: mobile charge structures, a coarse T from the rule. First feasible check: does a resolved winding drift down an emergent depletion gradient with a transverse response.
