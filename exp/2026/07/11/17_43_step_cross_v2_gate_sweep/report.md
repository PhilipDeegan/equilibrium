# step_cross v2 — D1 gate comparison (v1 vs shear vs product vs opposed)

Decision probe for `doc/step_cross_v2.md` §10 D1: compare candidate interior-crossing
gates on the two configurations that must separate — electron double-ring dipole
(R1: must NOT erode at rest) vs neutron Taylor spheromak (R2: MUST keep unwinding).

## Parameters

`probe_cross_v2_gates.py` (all defaults): R=15, sph_frac=0.2 → scale=3.0,
cross_scale=0.01, lam_target=1.0 (neutron, frozen), n_audit=600, n_steps=4000,
dt=5.128e-3 (total t = 20.5 time units), n_threads=10. Auto ref grid = neutron
audit's {p90, p99, 3·p99} per gate, from `raw_shear`/`raw_product`/`raw_opposed`
sampled over engaged (inward-phi) faces.

Gates: v1 (reference, G≡1), shear (`m=|v||v_nb|-v·v_nb`), product (literal
single-cell "transverse co-located flux"), **opposed** (`o=max(0,-v·v_nb)`,
the >90°-only half of shear — added mid-session as the direct empirical test of
the shear gate's predicted failure mode: firing on the electron's own smooth
ring curvature).

## Bug found and fixed

`opposed`'s auto ref grid included `ref=0.0` (neutron audit p90 for opposed was
exactly 0.0). `CrossV2.gate_field` computed `raw/(raw+ref)` unconditionally →
`0/0 = NaN` at every non-engaged face (`raw=0`), and since delta is
`G_link * inward`, `NaN * 0 = NaN` corrupted the entire phi field even though no
face should have fired. Fixed in `cross_v2.py`:`gate_field` to return `G=0`
wherever `raw+ref == 0` (no signal → no gate, not undefined). Selftest re-verified
green after the fix (v1 equivalence 8.9e-16, conservation/antisymmetry exact,
shear null exact).

## Results

| gate | surv_e (R1: want ≈1) | surv_n (R2: want ≪1) | separation e/n |
|---|---|---|---|
| v1 | 0.274 | 0.639 | 0.43 |
| shear/ref=1.28e-2 | 0.364 | 0.646 | 0.56 |
| shear/ref=5.93e-1 | 0.697 | 0.715 | 0.98 |
| shear/ref=1.78e0 | 0.807 | 0.765 | 1.06 |
| product/ref=2.63e-2 | 0.395 | 0.644 | 0.61 |
| product/ref=1.68e-1 | 0.542 | 0.663 | 0.82 |
| product/ref=5.03e-1 | 0.652 | 0.696 | 0.94 |
| opposed/ref=2.14e-2 | 1.014 | 1.000 | 1.01 |
| opposed/ref=6.41e-2 | 1.013 | 1.000 | 1.01 |

("separation e/n" = surv_e/surv_n; the target is separation → 0, i.e. electron
frozen at 1 while neutron decays toward 0. All gates instead push separation
*toward* 1, i.e. toward no distinction.)

## Findings

1. **v1 baseline is inverted relative to what's required.** The unlinked
   electron (should be ~frozen, R1) decays *faster* (surv 0.274) than the
   linked neutron (should decay, R2, surv 0.639). v1 fires on any inward
   face-phi regardless of linkage, so it erodes the free-transiting electron
   dipole harder than the genuinely-linked knot — exactly the pathology
   `step_cross_v2.md` was written to fix. Electron decay rate is consistent
   with the independently-measured τ_φ≈16 t.u. from the R80 smoke run
   (`exp/2026/07/10/19_56_.../report.md`): exp(-20.5/16) = 0.278 ≈ 0.274
   measured here — good cross-check that v1's electron erosion is the same
   mechanism in both experiments.

2. **shear and product do NOT separate — raising ref moves both survivals
   together, not apart.** As ref increases (gate more permissive → less
   firing overall), surv_e and surv_n both rise roughly in lockstep toward
   1, and the separation ratio climbs monotonically 0.43→1.06 (shear) /
   0.43→0.94 (product). Neither gate ever produces "electron frozen, neutron
   still decaying" — they just uniformly throttle both configurations by
   similar amounts. This is the outcome predicted in `doc/step_cross_v2.md`
   §4 (single-cell/local-misalignment gates cannot see linkage — a coherent
   ring's own curvature and a linked crossing look alike from one cell) and
   matches the "failure mode to watch" logged after D1 was decided.

3. **opposed does separate on R1 — perfectly — but fails R2.** surv_e ≈
   1.014 (electron genuinely frozen, even slightly *above* 1, i.e. no
   erosion at all — matches the audit finding that the electron's engaged
   faces carry essentially zero `o_k` almost everywhere). But surv_n ≈
   1.000 too: the neutron also stops decaying. `<G>eng` for the neutron
   under opposed is only ~0.003–0.016 (vs v1's constant 1.0) — genuine
   >90°-opposed transit exists in the linked knot (nonzero, unlike the
   electron) but is too rare/weak at this scale, cross_scale, and step
   count to produce measurable unwinding in 4000 steps.

4. **Read together, this is informative, not a dead end.** `opposed`'s
   sparsity is itself evidence for the §4 mechanism: real linkage is
   confined to a small crossing/interpenetration region, not smeared across
   the whole knot the way v1's isotropic firing implies. The engagement
   fraction is nonzero specifically for the linked configuration and zero
   for the free one — the qualitative separation `step_cross_v2.md` asked
   for is present in the *raw* variable; it's just too weak in the *decay
   rate* at cross_scale=0.01 over this step count to see in survival curves.

## Next steps

- Sweep `cross_scale` upward for `opposed` only (e.g. 0.05, 0.1, 0.3) at
  fixed ref, holding n_steps fixed — if surv_n drops meaningfully while
  surv_e stays pinned near 1, `opposed` is the winning gate and the
  remaining work is calibrating cross_scale/m_ref, not redesigning the gate.
- If surv_n stays near 1 even at much larger cross_scale, genuine linkage
  engagement is too rare/local for a same-cell gate regardless of drive
  strength — confirms §4's "linkage is not a one-cell property, it shows up
  one ring out" and motivates extending the gate to the 14-neighbour ring
  (per the memory's failure-mode note), not just the cell's own transit
  vector.
- Longer neutron-only run (e.g. 20k steps) at the current opposed/ref to
  rule out "just needs more time" before concluding the effect is too weak.

## Artefacts

- `probe_cross_v2_gates.py`, `cross_v2.py` — script copies at run time
  (includes the `gate_field` NaN fix and the `opposed` wiring added this
  session)
- `run_output.txt` — full stdout (audit + all 9 arms + summary table)
- `cross_v2_gates.npz` — per-arm (t, surv, rms, eng) histories
- `cross_v2_gates.png` — survival-vs-time plot, electron/neutron side by side
- `run.sh` — exact invocation
