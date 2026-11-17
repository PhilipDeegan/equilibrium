# Entries for 2026-08-27 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## PREVIOUS (2026-08-27, end of day): code observation + Gauss-law/Coulomb validation

`init_electron_dipole_ic` sources no `u`-gradient — see `doc/dst/failures/electron-dipole-ic-zero-charge-u-gradient.txt`.

**Foundational Gauss's-law/Coulomb mechanism: validated under the right BC.** `probe_grounded_sphere.py` (exact BC, `u=0` at outer shell, damp_max=0.003): R^2=0.973->0.992, interior `|Q_i|` dropped ~10x, residuals mostly under ~9% — real, close match to the exact classical `u(r)=q(1/r-1/R)` solution. Global conservation (`Q_enc(r)` radius-independent to 1-3%) holds robustly under any BC tested. Shape-sensitive results are BC-dependent — see `doc/dst/failures/sponge-bc-distorts-poisson-shape.txt` and `doc/dst/failures/zero-damping-symplectic-never-settles.txt`.

**`step()`'s sign convention confirmed correct** for the bare-point-source mechanism: `du_i=-dt*sigma*Sum_k W_k*phi[i,k]` (continuity structure), `dphi[i,k]=dt*sigma*W_k*(u_i-u_nb)` (flux flows high-to-low u correctly). Still open: winding-direction convention (`W=+-1`) on actual circulation-bearing structures not yet checked.

Kick/coast energy-partition and knot-in-lam-gradient tests: both inconclusive — see `doc/dst/failures/kick-coast-and-knot-star-gradient-inconclusive.txt`.

**Conceptual/vocabulary correction, now in [[feedback_conventions]]**: no "the electron's own field"/"generates a magnetic field" — DST is monistic, prefer "effect radius"/"a disturbance in the substrate."

**Parked ideas, not built**: (1) directed `phi`-flux-between-objects diagnostic (closer to "virtual photon exchange"). (2) "displacement gradient"/"substrate protrusion" charge hypothesis — no existing mechanism moves anything via bare `lam`-gradients alone; real test is the knot-in-star-gradient result once its measurement is fixed. (3) `discrete_curl.py`'s exact `circulation()` primitive as a better `lam_target` than `alpha*|phi|` — still parked.

## PREVIOUS (2026-08-27, earlier): comprehensive gate sweep + e/p pair diagnostics

Comprehensive gate-invalidation sweep: no step_cross v2 gate passes both R1 and R2 — see `doc/dst/failures/step-cross-v2-no-gate-passes-both-r1-r2.txt`.

**A2 rest-phase chirality confound theory CONFIRMED.** Rerunning `probe_cycling_ring_curl.py` rest-phase-only with `cross_scale=0`: `cos_sim(diff_p,diff_m) = -1.0000` exactly (vs original run's +0.035 with `step_cross` active) — `step_cross` was genuinely corrupting the cycling ring's own circulation even at rest; with it off, "cycling alone produces a chirality-correct dipole" (`doc/dst.md`'s original prescription) is confirmed true. **Correction (Philip)**: says nothing about the disequilibrium/clock-divergence magnetism hypothesis (proton cycling phase-locked to local `sigma=1/(1+lam)`) — that only has content once something genuinely translates through non-uniform `sigma`; untested here (`v_coast=0` confirmed). Orientation problem (`along_v` still exceeds `along_phi`) untouched. Data: `exp/2026/08/27/16_40_a2_restonly_cs0/`.

Single-cell triangle-plaquette "cycling on most faces" construction explored and abandoned — see `doc/dst/failures/single-cell-triangle-plaquette-cycling-abandoned.txt`. Redirected to the existing electron double-ring dipole (validated fixed point, R1).

**`electron_positron_collision.py` extended**: `--gate`/`--ref`, `--same-sign`, genuine energy-weighted centroid tracking (`dz_up`/`dz_dn`).

**RESULT — sign-independence rules out step_cross u-asymmetry as real attraction** — see `doc/dst/failures/step-cross-u-asymmetry-sign-independence-not-attraction.txt`. Next, if this thread continues: track centroid drift for an ISOLATED single object as a relaxation baseline before trusting any future pair `dz` measurement.

**Also raised, not yet built**: (1) `circulation()` primitive as better `lam_target` (parked). (2) three-way energy-partition diagnostic (translation/heat/radiation) for step_cross perturbations, motivated by Larmor-radiation analogy — would resolve A2's unexplained `H_kick`->`H_coast` drop. (3) "majority-lean" quantized gate (per-cell coherence fraction `L_i=|v_i|/Sum_k W_k|phi_i,k|`, fires directionally like `barrier`) — a genuinely new gate family distinct from the invalidated roster, motivated by the traveling-wave reproducer already showing coherent pattern translation under bare `step()` with no `step_cross`.
