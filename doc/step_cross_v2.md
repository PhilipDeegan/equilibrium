# step_cross v2 — specification draft

Status: **draft for decision** (2026-07-11). The gating criterion is the one design
decision everything downstream depends on (e/p translation-cost ratio, closed-knot
decay recalibration, AMR port). Nothing here is implemented yet.

## 1. Evidence driving the redesign

From `exp/2026/07/10/21_15_electron_translation_gate/` (pinned pressure-tilt kick on
the lam-free double-ring winding):

1. **cross = 0: pressure coupling is a machine-precision null** (mobility ~1e-13).
   With lam = 0 the operator set is linear; the tilt superposes and the winding
   cannot feel it. Omnibalance is untouchable by pressure in the linear sector.
2. **cross > 0: erosion without transport.** COM wander is sub-cell,
   sign-inconsistent, does not scale with drive. The winding erodes
   (pol_survival 0.17 → 0.03) and gains nothing.
3. Therefore v1 damages face-borne circulation it should ignore, and provides no
   channel by which pressure asymmetry translates a winding. The cost-of-translation
   half of inertia is **unimplemented physics**, not a miscalibration.

## 2. Requirements

- **R1 — free transit.** An unlinked winding at rest engages the interior crossing
  not at all (up to curvature order). The cross=0 run is the reference behaviour.
  Photons and electron internal circulation are face-borne and must pass free.
- **R2 — linkage.** Co-located transverse windings (the closed knot's Hopf link)
  engage the crossing at rest → unwinding proceeds. Closed-knot decay must survive
  the redesign (rates will need recalibration).
- **R3 — cost of translation.** Pressure asymmetry across a winding converts into
  coherent displacement with finite, sign-consistent mobility; erosion subdominant.
  This is the missing half of inertia (the other half, omnibalance, is the
  zero-net-force state and already exact).
- **R4 — conservation discipline.** Global Σ dphi = 0 exactly; face antisymmetry
  phi[i,k] = −phi[nb_k, mirror_k] preserved; the drain+mirror bookkeeping of v1 is
  kept.
- **R5 — DST discipline.** Strictly local (cell + 14 faces), per-tick discrete
  differences (no continuous derivatives), σ(λ)² throttle retained, no global
  solves at step time.

## 3. Why v1 fails

v1 fires on *every* inward face-phi (`delta = −dt·scale·σ²·phi[i,k]`), so a resting
coherent transit is treated the same as a genuine crossing — that is R1's violation.
Its redistribution to edge-adjacent faces is isotropic, so even when it fires it
diffuses the pattern instead of displacing it — R3's violation. And u appears
nowhere in it, so pressure cannot enter at all — R3's violation at the root.

## 4. Structural result: single-cell gates cannot see linkage

Define per cell i the axis transit components over the 7 face axes
(4 hex + 3 sq; mirror pairs (k, mk)):

    t_a = (phi[i,k_a] − phi[i,mk_a]) / 2        (through-flow on axis a)
    c_a = (phi[i,k_a] + phi[i,mk_a]) / 2        (head-on convergence on axis a)

A coherent winding transiting the cell along direction d with strength s produces
the pattern `t_a = s·(n̂_a·d̂)`, `c_a = 0` — this is the pattern the reversible step
propagates.

**Superposition identity.** Two windings crossing at the cell, s₁ along d̂₁ and s₂
along d̂₂, produce

    t_a = s₁(n̂_a·d̂₁) + s₂(n̂_a·d̂₂) = (n̂_a · (s₁d̂₁ + s₂d̂₂))

— *exactly* the single-cell pattern of one coherent winding along the vector sum.
The linked crossing and the plain oblique transit are indistinguishable from inside
one cell. Consequences:

- Any gate that is a function of one cell's 14 phi values — linear, quadratic
  ("transverse co-located flux"), or otherwise — either fires on both or on
  neither. The candidate gate named at the end of the translation-gate report is
  in this class and is therefore ruled out as stated.
- A least-squares "residual after projecting out the best coherent transit" is
  identically zero for *any* superposition of transits (the coherent patterns span
  a 3-dim linear subspace, and the weighted Kelvin frame is isotropic:
  Σ_a W_a n̂_a n̂_aᵀ = 5·I with W hex = 3, sq = 1). Ruled out.
- A collision-only gate (fire on c_a) detects anti-parallel head-on convergence
  but is silent on orthogonal linkage (two orthogonal transits have c_a = 0 on
  every axis). Insufficient alone.

Linkage is not a one-cell property. It shows up one ring out: where two windings
interpenetrate, *neighbouring cells carry differently-directed transit*. This is
the discrete current-sheet signature — consistent with the Sweet–Parker-style phi
diffusion result (doc/mems: phi-diffusion is the DST reconnection mechanism), of
which this gate is the topology-aware refinement. Still strictly local in the DST
sense: cell + its 14 neighbours.

## 5. Proposed gate — winding shear across a face

Per cell, the transit vector (computable in the existing single-threaded phase 1;
n̂_k are the 14 outward face normals, W hex = 3, sq = 1):

    v_i = Σ_k W_k · phi[i,k] · n̂_k

For a cell transited coherently, v_i points along the local winding direction.
Per face k, the **shear gate**:

    m_k = |v_i|·|v_nb_k| − v_i · v_nb_k        (≥ 0)

Properties (all follow directly from the definition):

| configuration | m_k | consequence |
|---|---|---|
| straight coherent transit (photon, tube interior) | 0 (directions parallel) | passes free |
| ring circulation, radius R cells | ∝ 1/R² (direction rotates ~1/R per cell) | quadratically suppressed — resting double-ring ~free |
| amplitude profile / tube flank (same direction, different magnitude) | 0 | no flank erosion |
| tube edge against vacuum (v_nb = 0) | 0 | no edge erosion |
| Hopf-linked windings (interpenetration region) | O(1) misalignment | fires — closed knot unwinds |
| head-on anti-parallel windings | maximal (= 2·v·v_nb) | fires — reconnection |
| vacuum noise | ∝ noise² (product of magnitudes) | negligible |

m_k is symmetric between the two sides of the face, so the inward-only
(receiving-cell) single-processing discipline of v1 carries over unchanged and
antisymmetry is preserved by the same drain+mirror bookkeeping.

Linkage delta on inward face k (phi[i,k] < 0):

    delta_link = dt · scale · σ(λ_i)² · g(m_k) · (−phi[i,k])

with g a normalisation mapping m_k (dimension phi²) to [0, 1] — e.g. saturating
g(m) = m / (m + m_ref). m_ref is a calibration knob (open item C1 below). The
kernel for the linkage channel is v1's kernel unchanged: drain face k, redistribute
isotropically to edge-adjacent faces, mirror everything — reconnection/unwinding.

## 6. Drive channel — pressure asymmetry → coherent displacement

u must enter the operator explicitly (Section 1 proves nothing else can move a
lam-free winding). Per-face pressure difference, exactly the quantity the
reversible step already uses:

    Δu_k = u[nb_k] − u[i]

and the per-cell pressure-asymmetry vector D_i = Σ_k W_k · Δu_k · n̂_k
(a per-tick discrete difference, not a derivative).

The drive fires only where a winding is present (delta ∝ |phi|) and only in
proportion to the asymmetry. Two coupling designs (decision D2):

- **D2a — biased kernel (recommended).** Drive delta on inward face k:
  `delta_drv = dt · scale · σ² · β · |Δu_k|/u_ref · (−phi[i,k])`, but redistributed
  *anisotropically*: adjacent-face weights w_j ∝ max(0, n̂_j·D̂_i) (normalised),
  instead of v1's uniform 1/n_adj. The circulation pattern is re-routed toward the
  drive side each tick — the winding walks. Displacement is coherent by
  construction; the phi² dissipated per cell of displacement *is* the cost of
  translation.
- **D2b — gate-only.** Drive opens the v1 isotropic kernel (G += β|Δu_k|/u_ref)
  without biasing redistribution. Simpler, but the translation-gate experiment
  already showed isotropic redistribution yields jitter + erosion, not transport —
  expected to fail R3. Listed for completeness.

Sign of the walk (toward higher or lower u) is not imposed here — the translation
probe measures it, and sign-consistency across configs is itself part of the R3
pass criterion.

## 7. Assembled operator

    for each cell i, each face k with phi[i,k] < 0:
        m_k   = |v_i||v_nb_k| − v_i·v_nb_k
        G_link = g(m_k)
        G_drv  = β · |Δu_k| / u_ref
        delta  = dt · scale · σ(λ_i)² · (G_link + G_drv) · (−phi[i,k])
        drain face k by n_adj·(...)+ mirrors, redistribute:
            linkage share  → isotropic over FACE_ADJ[k]   (v1 kernel)
            drive share    → weighted by max(0, n̂_j·D̂_i)  (D2a)

v1 is recovered by G ≡ 1 — keep v1 callable (separate entry point) until
recalibration lands; every published closed-knot decay number was measured
against v1.

## 8. Validation matrix

| # | probe | requirement | pass criterion |
|---|---|---|---|
| 1 | `probe_electron_translation.py`, kick off, cross v2 ON | R1 | resting double-ring frozen to ~cross=0 reference (pol_survival ≈ 1 over 5k steps) |
| 2 | photon / face-transit probe | R1 | free transit unaffected |
| 3 | closed-knot decay (`test_neutron_decay.py` path) | R2 | unwinding proceeds; τ recalibrated, monotone in scale |
| 4 | `probe_electron_translation.py`, kick on | R3 | mobility sign-consistent across {axis}×{cross}, |v| scales with g; pol_survival stays ~1 during coherent walk |
| 5 | conservation suite (`test_adaptive_vs_substrate.py` pattern) | R4 | Σdphi = 0 to machine; antisymmetry exact |
| 6 | lam-resistance sweep (`probe_step_cross_lam_resistance.py`) | R5 | σ² throttle behaviour preserved for lam-carrying knots |

Then the payoff measurement: **e/p translation-cost ratio** — pressure work per
unit displacement for the lam-free double-ring vs a lam knot. Note σ(λ)² already
makes the knot's crossing channel nearly frozen, so its translation cost is
enormously higher — the ratio may fall out of σ(λ)² directly.

## 9. Implementation plan

1. Python prototype — DONE 2026-07-11: `run/substrate/adaptive/cross_v2.py`,
   operating on the adaptive cluster's leaf arrays so all existing ICs and
   probes are reused unchanged. Selftest green: v1-mode reproduces the compiled
   `step_cross` to 8.9e-16 (after matching the adaptive boundary semantics —
   non-live faces skipped, n_adj counts live adjacent faces only); global
   Σdphi and antisymmetry machine-exact for all three gates; shear gate exactly
   zero on a coherent parallel transit (interior pairs).
2. Gate comparison probe: `run/substrate/adaptive/probe_cross_v2_gates.py`
   (electron double-ring vs closed-knot spheromak, gate × ref sweep, audit
   phase anchors the ref grid to the neutron's engaged-face raw-value tail).
3. Uniform C++ (`inc/eq/substrate.hpp`, new entry point alongside v1) after the
   winning gate is confirmed. Build/run stays with Philip.
4. AMR port last, per the established update-order policy.

## 10. Decisions (settled 2026-07-11)

- **D1 (Philip): prototype both, compare.** Python-prototype the shear gate AND
  the single-cell transverse-product gate; run both on the closed-knot decay and
  resting double-ring probes; decide on the numbers. §4 predicts the product gate
  fails the R1/R2 separation — the comparison is the empirical check of that proof.
  A third arm, **opposed** (`o_k = max(0, -v_i·v_nb_k)`, the >90°-only half of
  shear), was added to the comparison: shear fires progressively on any
  misalignment, including the electron's own ring curvature (§5's "failure mode
  to watch"); opposed is exactly zero below 90° per-hop rotation, so it is
  immune to that false-positive by construction. Same audit/ref-grid machinery
  as shear/product in `probe_cross_v2_gates.py`.
- **D2 (Philip): biased kernel (D2a).** Drive share redistributes anisotropically
  toward the pressure-asymmetry direction.

## 11. Remaining calibration knobs
- **C1:** m_ref in g(m) — set from closed-knot interpenetration amplitude; scan in
  row 3.
- **C2:** β, u_ref for the drive — set from the translation probe's kick window;
  scan in row 4.
- For the product gate prototype (D1 comparison arm), the per-face criterion is
  G_prod = g(p_k) with p_k = (−phi[i,k]) · Σ_{j∉{k,mk}} W_j |phi[i,j]| — the
  literal "transverse co-located flux" reading, kept single-cell by design so the
  comparison tests exactly the §4 claim.
