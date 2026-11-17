# Entries for 2026-08-08 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## eta_h L5 calibration, re-run against the properly-resolved checkpoint (2026-08-08 night)

Re-swept L5 directly against the real `/bk/l5_core_diameter9_cells_
across_220.h5` checkpoint (`exp/2026/08/08/20_09_eta_h_l5_resolved_
calibration/sweep_eta_h_l5_resolved.py`, L0-L4 held at the old re-earned
ladder, only L5 varied, 3 steps each, 4 threads, ~740s/step):

| eta5 | step1 | step2 | step3 |
|---|---|---|---|
| 0 | 1.003582 | 1.013292 | 1.026412 |
| 1e-10 | 1.003580 | 1.012841 | 1.024836 |
| 1e-9 | 1.003611 | 1.011893 | 1.021735 |
| 1e-8 | 1.003646 | 1.011273 | 1.020440 |
| 1e-7 | 1.003652 | 1.011163 | 1.020309 |
| 1e-6 | 1.003653 | 1.011151 | 1.020296 |

Monotonic improvement 0->1e-8 (step3 growth 2.64%->2.04%, ~23% reduction),
then saturates — 1e-7/1e-6 give essentially the same step3 value as 1e-8
(within 0.02%), no instability seen out to 1e-6. See
doc/dst/failures/eta-h-negligible-old-telescope.txt for why the earlier
"eta_h is negligible" reading no longer holds. Both scripts + all 6 logs
live in the exp dir above.

**Bisected 2026-08-09** (`exp/2026/08/09/13_42_eta_h_l5_fine_bisect/`):
eta5=3.3e-9 and 6.6e-9 added between the 1e-9/1e-8 endpoints (same
checkpoint, same held L0-L4 ladder, 3 steps). Step3 growth: 3.3e-9 ->
2.0760%, 6.6e-9 -> 2.0519% (vs 1e-9 -> 2.1735%, 1e-8 -> 2.0440%). Knee sits
close to **eta5 ~ 3e-9** — most of the total 0->1e-8 improvement is
already captured by 3.3e-9, each further doubling toward 1e-8 buys a
rapidly shrinking increment (-0.098, -0.024, -0.008pp per step). Consistent
with (same order of magnitude as) the already-re-earned **L4 value of
3e-9** — a defensible default for L5 too if a single shared value across
levels is preferred, though not re-derived from first principles here. No
instability at any bisection point.

**Next**: still only tested at 3 steps / L5 in isolation (L0-L4 fixed) —
not yet re-earned at L0-L4 against this checkpoint, not yet tried at longer
step counts on this checkpoint, not tried at L6+ (blocked on the RAM wall
above regardless). Remaining deltas (<0.03pp) are close to where run-to-run
noise could matter — not checked for reproducibility at fixed eta5.

## Done (2026-08-08): HDF5 checkpoint + multiplicity-table memory optimization

Two changes to `AdaptiveCellCluster`/`cluster_io.hpp`, implemented, tested
(C++ round-trip test bit-identical at small scale + real 23M-leaf checkpoint
round-trip bit-identical: `H`/`u`/`phi`/`leaf_indices` all exact matches), and
verified with no regressions against the existing `test_lattice` suite (same
single pre-existing unrelated failure, `test_curvature_corrected_secant_
formula_isolated` — see "Recently resolved" below, still open, unaffected):

1. **`MultiplicityTable`** (`cluster.hpp`) replaces the dense
   `std::vector<double> face_multiplicity`/`ancestor_group_multiplicity`
   (previously `14×leaves` doubles each, ~2.6GB apiece at 23M leaves) with a
   collapse-to-empty-when-all-1.0 wrapper over a `float` (not `double`)
   backing array. On the real checkpoint, `face_multiplicity` is 0% deviated
   → collapses to empty; `ancestor_group_multiplicity` is 47.8% deviated →
   stays a dense float array (half the old double cost). Values verified
   float32-safe (largest deviation ~4e5, float32 relative error ~1e-7,
   negligible against a one-shot flux-correction divisor).
2. **Streaming `save_cluster`/`load_cluster`** (`cluster_io.hpp`) — replaced
   the old approach (build one O(n_slabs·CAPACITY) flat buffer per field,
   holding the whole cluster twice in memory at once) with per-slab HDF5
   hyperslab `select().read_raw()`/`write_raw()` straight into/out of each
   `CellSlab`'s own storage. Save peak memory unchanged now (previously
   +5.47GB); save time for the 23M-leaf checkpoint: 1.6s.

Peak RSS after `AdaptiveLattice.load()` on the real checkpoint: 20.58GB (orig)
→ 18.22GB (after both changes) in isolated measurement; **~17GB observed in
the actual `load_and_step.py` production run tonight** (threading was also
fixed in that script — `set_n_threads(8)` wasn't being called on the
load-from-snapshot path, only on the build-from-scratch path).

**Not done / smaller remaining levers**: the peak is now set by
`rebuild_ancestor_group_multiplicity`'s own transient (builds a full dense
`double` scratch buffer before compressing to float, ~2.6GB momentarily) —
could be streamed too but wasn't attempted. `CellSlab` byte-shrinking
(`parent`-array-for-L0 removal, further float32 downcasts of `u`/`phi`) was
explored in the old log, designed but not implemented — see `doc/dst_amr.md`
"Memory layout" section if resuming; won't reach the 8x needed for large-R
work regardless (that needs multi-scale-L0, see above).

## IN PROGRESS: QED-style demo experiments (unrelated to the AMR/L8 work above)

Separate initiative: DST analogues of standard QED demonstrations, next to
the existing `equilibrium/substrate/ab.py` (Aharonov-Bohm) and `casimir.py`
(parallel-plate Casimir, pure dispersion-relation mode sum, no lattice
stepping). All live in `run/substrate/` (flat `SubstrateLattice`/
`DampedLattice`, no AMR needed).

**Done tonight**: `equilibrium/substrate/double_slit.py` +
`tst/substrate/test_double_slit.py` (7 tests, ~28s) +
`run/substrate/double_slit_demo.py`. Confirmed design (after mid-session
corrections -- see doc/dst/failures/double-slit-fixed-u-boundary-wall.txt,
doc/dst/failures/double-slit-evanescent-drive-frequency.txt,
doc/dst/failures/double-slit-plain-lattice-edge-blowup.txt):
- Wall/backplate = high-`lam` (dense-material) regions in a low-`lam`
  background gas (`gas_lam`, real atmosphere analogue). Front-face
  reflection, slit-edge diffraction, and partial wall-body leakage all fall
  out of the same `sigma`-gradient update rule.
- Recorded pattern is accumulated dose (`∫u² dt`) at a solid backplate, like
  a real photographic plate integrating exposure.
- Source is a continuously driven, quasi-monochromatic oscillator. Current
  default `drive_period_steps=250` (omega~4.9, wavelength~4-5 cells) is
  safely inside the lattice's propagating band (omega_max~12.5 for
  W_HEX=3/W_SQ=1).
- Domain is `DampedLattice` (sponge boundary) for the continuously-driven
  ~1600-step run.
- Fringe contrast at this small/fast demo scale (radius=18, N~32k leaves,
  ~3s/run) is real but modest -- two-slit gives a taller, structured central
  lobe with a shoulder feature the one-slit envelope doesn't have, not yet a
  clean many-fringe textbook picture. A bigger domain would sharpen it
  further -- not chased tonight, queued as a next step.

**Done 2026-08-09**: Casimir-Polder-style point-scatterer ("knot-knot")
vacuum force -- `equilibrium/substrate/casimir_polder.py` +
`tst/substrate/test_casimir_polder.py` (9 tests, ~5.6s). Confirmed this does
NOT need a `SubstrateLattice` instance at all -- reuses `casimir.py`'s exact
dispersion relation `_dispersion(Kx,Ky,Kz,a)` directly, extended to a
real-space (imaginary-frequency) Green's function `G(r,xi)` via 3-D
Brillouin-zone quadrature, instead of the plate case's transverse-averaged
`G_function`.
- `G(r, xi) = INT_BZ cos(Kx*r) / (omega(K)^2 + xi^2) d^3K / (2*pi)^3`, r
  along the x-axis (arbitrary reference direction -- lattice is anisotropic).
- Naive equal-weight cube quadrature doesn't work at large r (see
  doc/dst/failures/casimir-polder-naive-quadrature-convergence.txt) -- fixed
  by splitting the integral: outer Kx integral via scipy's Filon-type
  oscillatory quadrature (`weight='cos'`), inner (Ky,Kz) integral via an
  equal-weight grid (`Nyz`, checked by doubling via `converged_G_point`).
- Cross-check confirmed: the long-range limit of `G(r,0)` (and finite-xi
  `G(r,xi)`) recovers the continuum Yukawa propagator
  `exp(-xi*r/c)/(4*pi*c^2*r)` with the SAME `c=sqrt(10)*a` already used in
  `casimir.py`'s continuum reference (ratio within ~5-15% at r=5-8,
  `Nyz=250`) -- the plate-Casimir and point-scatterer effects come from one
  dispersion relation, not two unrelated formulas.
- Scope is the SCALAR/monopole analogue (coupling linearly to `u`):
  `U(r) = -(1/2*pi) INT_0^inf G(r,xi)^2 dxi`, closed-form continuum
  reference `U_continuum(r) = -1/(64*pi^3*c^3*r^3)` (derived and verified
  against direct numeric integration of the Yukawa form) -- confirmed
  negative (attractive) and ~1/r^3 scaling (r: 4->8 gives ratio 0.125±0.05).
  Lattice/continuum ratio sits in a 0.8-0.95-ish band at r=3-12, expected to
  approach 1 as r grows since the leading-order dispersion is isotropic
  (unlike the plate case's genuinely 2D BZ-boundary integral, which has a
  fixed ~0.72-0.76 constant). NOT a literal reproduction of the EM
  dipole-dipole `1/r^6`/`1/r^7` laws, which would need a vector (`phi`)
  coupling instead -- open extension.

**Resolution scaling confirmed 2026-08-09**: at r=30, once fully converged
in `Nyz` the lattice/continuum ratio approaches ~1 (not a fixed sub-1
BZ-shape constant) -- see doc/dst/resolution_requirements.txt, REQUIREMENT 3
for the convergence table and the under-resolution trap this rules out
(same lesson as `project_hop_telescope_resolved_spheromak`). No run/ demo
script written (matches `casimir.py`'s own convention -- module + tests
only, no lattice stepping needed).

**Next**: isolate whether n_xi or Nyz is the binding constraint on the U(r)
convergence at large r (the G(r,0) case converges in Nyz alone since it's a
single xi=0 point; U(r) additionally needs the xi-grid resolved, and that
requirement was not separately swept at r=30). A systematic "how does
required Nyz scale with r" table (like the spheromak's `cells_across`
table) has not been built yet -- only two spot-checks (r~25-30 range,
~13 minutes total runtime); queued as future work.
