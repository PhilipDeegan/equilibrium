# Entries for 2026-08-20 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## 2026-08-20 (compressed): EM chirality verifiers built — still missing whatever makes magnetism sign-dependent

Built two empirical verifiers for classical EM claims DST never actually
tested: (A) moving charge -> magnetic-like `curl(phi)` field, (A2) cycling
`phi` ring-mode -> same (doc/dst.md's actual prescription). Both fail the
chirality test (wake doesn't distinguish charge/winding sign) — see
`doc/dst/failures/moving-charge-verifier-fails-chirality.txt` and
`doc/dst/failures/cycling-ring-verifier-fails-chirality.txt`. A sponge-BC
methodology gotcha found along the way is in
`doc/dst/failures/sponge-ablation-inert-but-damps-near-walls.txt`.

Encouraging partial result inside the A2 negative: cycling-ring's radial
`phi`-falloff fits slope -2.645 (W=+1) / -2.780 (W=-1) vs the classical
dipole target -3 — much closer than Test A's construction got to anything —
and orientation becomes correctly azimuthal-dominant once coasting.

**Open confound, not yet resolved (do this check before anything else in
this area)**: A2's chirality washout may be caused by `step_cross`'s
known-broken v1 gate corrupting the ring's circulation throughout the run
(active even at "rest" since A2's phi is not uniformly zero there), not a
genuine absence of cycling-driven chirality. Concrete next check: rerun
A2's REST phase alone with `cross_scale=0` and see whether
`cos_sim(diff_p,diff_m)` moves toward -1. Until that's done, A2 is NOT a
settled negative — see the failure file for full reasoning.

Standing hypothesis (Philip, not yet tested): DST may be missing an
explicit `lam`-interaction term for magnetism — `step()` only lets `lam`
set propagation speed (`σ=1/(1+lam)`), with no term coupling a
moving/winding `lam` object into a `phi` source; the only existing
lam<->phi coupling lives in the separate MHD/frozen-in extension
(`step_hall`, `dφ/dt=curl(v×B)` in `TestFrozenInDynamics`), which requires
a pre-existing B, so it can't be the source of B either. **Next actual
step** (after the step_cross confound check above): Test A3 — add an
explicit lam-velocity-driven `phi` source term and rerun the same
chirality/orientation check.

Conceptual framing to carry forward:
- `phi` (not `u`) is DST's own established vector potential (`doc/dst.md:
  371`, `doc/mems/project_overview.md:25` — "phi (face 1-form) = A,
  plaquette curl of phi = B"); `u` plays the scalar-potential role (E/B
  convention #3, `E:=-∇u-∂φ/∂t`, already proven ≡0 under free `step()`).
  Naming collision with the classical scalar potential also called φ —
  flag again next time it recurs.
- Magnetism-from-motion should be read as **offloading, not generation**
  (same field content redistributing under a frame change, classical SR:
  B is what E looks like from a moving frame — no new field created). Sharp
  pass criterion for any future verifier: did `curl(phi)` appear at the
  expense of an equal loss elsewhere in the `u²`/`φ²` bins of `H` — real
  conversion, not new energy from nowhere. [[project_dst_em_field_convention_crisis]]
  is the standing reference this all builds on (reused its `curl(phi)`
  and `E:=-∂φ/∂t` formulas rather than re-deriving).
- Test B (moving `phi`-circulation -> E-like/Faraday response) not yet
  started. Only convention #2 (`E:=-∂φ/∂t`) is worth building a real Test B
  around, reusing `face_vec`/`b_vec_vectorized` from
  `equilibrium/substrate/mhd.py` +
  `exp/2026/08/18/18_45_poynting_far_zone_rerun/probe_poynting_dipole_compare.py`.
