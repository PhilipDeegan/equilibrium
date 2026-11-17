# Entries for 2026-09-17 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-17 — fine alpha scan DISCONFIRMS the 09/16 self-sourced-lam "crossover" (still growing under a real stability check, not a fixed point); QM (Dirac chirality-mixing) independently confirms this project's own double-ring dipole requirement (two components, not one) for E=mc²; naive "two-component" IC built to test that reduces to the already-known-bad v1 IC and performs worse

Continuation of 09/16's free-electron ring-closure/soliton thread
(`exp/2026/09/16/20_50_free_electron_ring_closure/`). Full results,
tables, and next-steps checklist:
`exp/2026/09/17/free_electron_soliton_alpha_scan/status.md` (this entry is
a summary — read that file for details/numbers before continuing this
thread). Two new memories updated:
[[project_free_electron_soliton_needs_nonlinearity]] (v4 update) and the
new [[project_chirality_mixing_two_component_mass]].

**v4 (fine alpha scan, TRAVELING one-component IC, `alpha_scan_v4.py`)**:
reran 09/16's v3 mechanism at 9 alpha values x 3 configs ((2,6), (2,10)
[v3 baseline], (6,14)), adding a slope-over-the-latter-half diagnostic
(not just start-vs-end). **Result: the v3 "crossover" at (2,10) alpha~1.0
is NOT a genuine fixed point** — the slope diagnostic stays positive
(+1.3% to +2.3%/period) at EVERY alpha tested, including right at the
endpoint-growth zero-crossing; the ring is still net-growing throughout, a
transient wobble happens to cancel at the two sampled instants. alpha* also
isn't universal — (2,6) needs ~3x more alpha than (2,10) for the same
balance point. (6,14) is non-monotonic/sign-disagreeing between
diagnostics, originally guessed as a WKB-companion-phi validity issue (see
below, reinterpreted).

**Parallel thread, independent of the alpha scan**: Philip asked whether a
SINGLE ring/flux-tube is even the right structure, recalling that E=mc²
was thought to need at least two components "traveling at c." Traced this
to something more specific than spin's 720-degree/SU(2) behavior: the
Dirac bispinor splits into two coupled 2-component Weyl (chirality)
fields, each ALONE massless (structurally like a photon) — **the mass
term in the Dirac equation IS the coupling between left/right chirality**,
and zitterbewegung is the beat between them at 2mc²/ħ. This converges,
independently, with two things already in DST-project memory: the
established double-ring dipole electron model
([[project_dst_electron_dipole]], required for "omnibalance"/zero-net-force,
which a single ring lacks) and the zitterbewegung/chirality framing
([[project_zitterbewegung_spin_chirality]]). Three separately-motivated
arguments now converge on the same structural requirement — recorded in
[[project_chirality_mixing_two_component_mass]]. (Prompted by an external
LLM Q&A saved to `CLAWD` in the repo root — used only as a reference
pointer; the chirality-mixing mechanism was verified independently, not
just quoted.)

**v5 (STANDING/phi=0 IC test of the two-component idea,
`alpha_scan_v5_standing.py`)**: before building anything, worked out by
hand that summing an `n_prop=+n` and `n_prop=-n` companion-phi wave at
equal amplitude/matched time cancels phi EXACTLY (`Phi_k_amp` flips sign
under `K_local -> -K_local`) — this is algebraically identical to 09/16's
original v1 phi=0 "shocked" IC, not a new construction. Ran the same
alpha-scan/diagnostics on it anyway for a direct comparison. **Result:
worse across the board, not better** — far more severe windowed-energy
decay at every config/alpha (up to 99.99% at (2,6), where the reported
spread numbers are almost certainly measuring noise in a near-empty
remnant), the slope-vs-growth disagreement recurs and is worse, and
(2,10)'s growth is non-monotonic in alpha (rises before falling), unlike
v4's clean trend for the same config. **This is a negative result about
one specific degenerate construction, not evidence against the underlying
QM argument** — a real test needs either a phase-offset (non-degenerate)
pairing or the actual double-ring dipole IC (`init_electron_dipole_ic`),
neither built yet.

**(6,14)'s anomaly, reinterpreted**: v5 uses zero WKB/companion-phi
machinery and STILL shows the same anomaly flavor as v4 (sign
disagreements, wild non-monotonicity) — weakens the original "WKB
breakdown" guess. Better-fitting explanation: (6,14) sits closest to the
domain/sponge edge of the three configs tested (~19% radial clearance vs
34-48% for the other two) — boundary proximity, not WKB or n-specific
physics, is now the leading suspect.

**Also still open, not yet addressed by v4/v5**: the ring envelope width
(`width=2.0` cells, only ~2 lattice cells per Gaussian sigma against the
lattice's own unit-1.0 hop spacing) has never been resolution-tested — a
real, independent risk that under-resolution is behind some of the
observed spreading/noisy alpha-dependence in ALL of v1-v5, unrelated to
the topology question above.

**Queued next** (none started, see status.md for full detail): (1)
resolution check (2-3x width, matched domain) on (2,10)/(6,14) BEFORE any
more alpha scans — should come first; (2) a genuinely non-degenerate
two-component IC — either a phase-offset +n/-n pairing, or porting the
real `init_electron_dipole_ic` double-ring dipole (currently built for the
AMR cluster API, and already an EXACT static fixed point under the plain
reversible step — testing lam-nonlinearity on it meaningfully would need a
perturb-and-check-restoring-force design, not an anti-dispersion one);
(3) rerun (6,14) at larger domain clearance to test the boundary-proximity
hypothesis. Separately, unblocked: the magnetic-moment k-eigenvalue
question ([[project_dipole_moment_phi_convention_gap]]) remains open.

---
