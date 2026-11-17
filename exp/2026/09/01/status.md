# Entries for 2026-09-01 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION END (2026-09-01, night) — ring lean-drift clean; `stern_gerlach.py` fixed; chirality null result

Ring lean-drift (`init_toroidal_winding`, no boundary/kick mechanism, plain `step()` only; `exp/2026/09/01/19_50_step_cross_necessity/probe_ring_lean_drift.py`), confirmed twice (R=40 and R=30, with/without seeded `u`): a closed ring's own internal "lean" produces zero drift — mean flux vector and centroid displacement both at the 1e-15/1e-16 floating-point noise floor. A genuine external static `lam` gradient (no kick, no `step_cross`, no `step_lam_force`) DOES produce a real (~9-10 orders above noise) but bounded centroid shift, settling by period 8-10 and flat through period 16 (`--lam-grad 0.05 --n-periods 16`) — a genuine equilibrium-settling response to an inhomogeneous medium, not sustained translation.
Sign-flip check methodology gotcha: see doc/dst/failures/amplitude-sign-flip-cos-sim-not-chirality-test.txt.
`stern_gerlach.py` fixed — no longer crashes on fresh checkout (builds its own spheromak IC when the snapshot file is missing; default filename derived from `--R`/`--amplitude`/`--lam-rings`). Bug found + chirality null result: see doc/dst/failures/stern-gerlach-bg-injection-bug-and-chirality-null.txt.

## SESSION END (2026-09-01) — punch list for next time

Today in one line: found DST's real local energy-flux law (replaces the broken E×B Poynting convention) and confirmed DST's curl(phi)=B correctly reproduces the classical straight-wire B-field shape (zero-center/linear-ramp/1-over-r) once measured independently of `mhd.py`. Both are genuinely new, verified, keepable results — full detail in the entries before this one in the file, don't re-derive either from scratch.

**Concrete next ideas, roughly in the order they came up:**

1. **Is `step_cross` even necessary?** (Philip's own live question — highest-value thread.) Take `init_toroidal_winding`, run under plain `step()` only, check whether its own face-population asymmetry ("lean") produces real energy-centroid drift, using `equilibrium.substrate.energy_flux.local_energy_flux_vector`'s full 3-vector projected/summed over the ring. If real, months of step_cross-gate work may have been solving the wrong problem for translation. (Addressed above, 2026-09-01 night entry: lean itself gives zero drift; an external lam gradient does give a bounded settling shift.)
2. **Poynting-ring radial slope no longer matches classical -2** (equator -3.08, polar -1.58, under the corrected `S`) — open, not chased. Could be genuine (source-too-small-vs-wavelength, unresolved BCC anisotropy) or could mean `probe_poynting_step_only.py` needs a bigger/longer run. Next step: push R and settle time further.
3. **Direction + time-evolution of `S`**, not just radial magnitude — never checked whether `S` points outward the way classic dipole radiation does, or whether there's a genuine outward-traveling wavefront over time. Only a late-snapshot magnitude-vs-radius slope exists so far. Would need `S·r̂` tracked at fixed radius across many steps.
4. **Ampère/Lorentz-force term, parked mid-derivation** — missing ingredient for magnetism chirality (A/A2 tests, 2026-08-20, still fail chirality): a term coupling a moving/winding `lam`/current into sourcing `phi` circulation (Maxwell's "idle-wheel" mechanical-vortex picture was the inspiration). The original sketch was derived against `mhd.py`'s loop machinery, which is off-limits — needs re-deriving against `equilibrium.substrate.curl`'s independent, bug-checked plaquette loops instead. Not started.
5. `mhd.py`'s confirmed `SQ_PLAQUETTES["y"]` sign bug (see entry elsewhere in file) — deliberately left unpatched. If any `mhd.py`-based work (step_hall, step_frozen_in, the A/A2 chirality tests) resumes, remember `B_y` has been flipped this whole time; may explain some past chirality/orientation anomalies as a mundane bug rather than missing physics.

## RESUME HERE NEXT (2026-09-01, later): DST's discrete curl(phi)=B reproduces the EXACT classical straight-wire magnetostatic profile

New module `equilibrium/substrate/curl.py`, deliberately NOT built on `equilibrium.substrate.mhd` (peripheral MHD/instability extension, not core substrate). Face-index-to-physical-direction determined empirically from real `lat.pos`/`lat.nb` geometry; loop orientation uses the correct cyclic rule (`u=next(axis)`, `v=prev(axis)`, x->y->z->x).

**Result: exact shape match** to the analytic straight-wire `A_z(r)`/`B_phi(r)` (zero at axis, linear ramp, 1/r falloff) at a clean constant ratio 4/3 = (2/sqrt(3))^2 (sq-face hop-distance-squared calibration factor), across the full range inside and outside the wire — no anisotropy artifact at this resolution. Along the way, found and fixed (in the new module only) a real curl-sign bug — see `doc/dst/failures/mhd-py-curl-y-sign-bug.txt` for the bug itself (left unpatched in mhd.py per Philip).

Tests: `tst/substrate/test_curl.py` (2 tests, passing).

**Caveat, [[feedback_stability_not_reality]]**: validates the GEOMETRY of curl reconstruction against a hand-set, analytically-known `phi` — does not validate any dynamical mechanism generating that `phi` from a real circulating `lam` current (no `occ`-like current-to-`phi` source term in core `step()` — open, parked pending mhd.py redirect).

## PREVIOUS (2026-09-01, earlier): DST's local "Poynting vector" is a scalar per-face flux, not E×B — confirmed and re-verified

Derived and verified directly against real `SubstrateLattice.step()` (nonuniform sigma too): `S_i,k = W_k*phi[i,k]*ubar_i,k` (scalar flux per face, face-average u), residual against real per-step energy change scales as clean O(dt^2) from dt=1e-3 to 1e-5. New module `equilibrium/substrate/energy_flux.py`, new test `tst/substrate/test_energy_flux.py` (3 tests, passing). Full derivation: `doc/mems/project_dst_em_field_convention_crisis.md`'s new RESOLVED section — the crisis is closed.

Applied to `probe_poynting_step_only.py` (R=30/hops=4/8000-steps, properly settled): equatorial slope=-3.081, polar slope=-1.575, ratio=3.52 — direction still classically sane (equator>pole), neither slope cleanly -2, open/not chased further. See `doc/dst/failures/poynting-vector-e-cross-b-convention-broken.txt` for why the old E×B convention structurally cannot work, and for the 3 other Poynting scripts still left broken.

**Open thread, not yet built**: does `step_cross` even need to exist for translation, or can a closed circulation (e.g. `init_toroidal_winding`) drift under plain `step()` alone via its own face-population asymmetry (`W_HEX=3` vs `W_SQ=1`, [[project_bcc_dispersion_anisotropy]])? Test tool ready: `energy_flux.local_energy_flux_vector` (real 3-vector). Not yet run — chosen subject: `init_toroidal_winding` ring.
