# step_cross v2 — opposed gate, cross_scale sweep (does drive strength fix R2?)

Follow-up to `exp/2026/07/11/17_43_step_cross_v2_gate_sweep/` (the D1 gate
comparison). There, the `opposed` gate (`o_k = max(0, -v_i·v_nb_k)`, the
>90°-only half of shear) nailed R1 — the electron double-ring stayed frozen
(surv_e≈1.014) — but also froze the neutron (surv_n≈1.000), which should
decay (R2). Open question from that report: is the neutron's engagement just
*weak* (fixable by turning up cross_scale), or genuinely *absent* (a gate
redesign is needed)?

## Parameters

Same R=15/scale=3.0/lam_target=1.0/n_steps=4000 configuration as the parent
sweep. `--gates opposed --refs 2.14e-2` fixed (the smaller of the two D1 refs,
already near-maximal permissiveness for this gate). `cross_scale` swept at
0.05, 0.1, 0.3 — 5×, 10×, 30× the baseline 0.01.

## Results

| cross_scale | surv_e | surv_n | neutron `<G>eng` (final) |
|---|---|---|---|
| 0.01 (baseline, prior report) | 1.014 | 1.000 | 0.001–0.016 |
| 0.05 | 1.034 | 1.000 | ~0.0018 |
| 0.10 | 1.037 | 1.000 | ~0.0015 |
| 0.30 | 1.036 | 1.001 | ~0.0017 |

## Findings

1. **Neutron survival does not trend downward across a 30× drive range** —
   it stays flat at 1.000 ± 0.001 (noise level) from cross_scale 0.01 through
   0.3. `<G>eng` (mean gate value over engaged faces) also does not scale up
   with cross_scale — it stays at the same ~0.0015-0.0018 floor throughout.
   This rules out "weak but present, needs more drive": if the underlying
   raw `o_k` signal were present but small, amplifying cross_scale 30× would
   show *some* movement in survival. It shows none.
2. **Electron survival creeps slightly above 1 as cross_scale rises**
   (1.014→1.034→1.037→1.036) — small, likely numerical drift in the winding
   diagnostic rather than genuine growth (the gate is designed to be exactly
   zero on the electron's own transit almost everywhere per the earlier
   audit), but noted for completeness; it plateaus rather than growing
   further from cs=0.1→0.3, consistent with drift saturating rather than a
   real effect scaling with drive.
3. **Conclusion: the `opposed` gate's near-total silence on the neutron is
   structural, not a calibration problem.** A per-cell gate built from one
   cell's aggregate transit vector `v_i` vs. its immediate neighbor's `v_nb`
   essentially never sees the Hopf-linked crossing as a >90° disagreement at
   this scale, no matter how hard it's driven. This matches
   `doc/step_cross_v2.md` §4's structural argument taken to its logical
   endpoint: genuine linkage is a **one-ring-out** phenomenon (neighbouring
   cells carrying differently-directed transit, i.e. a discrete
   current-sheet signature spread over several cells), not a same-cell/
   same-face property, however that same-cell property is thresholded.

## Next steps

- The gate needs to look one ring further: instead of (or in addition to)
  comparing `v_i` to `v_nb_k` across a single face, compare `v_i` to the
  transit vectors of `v_nb`'s *other* neighbours (the cells that actually sit
  in the interpenetration zone), or build a face-local current-sheet
  detector directly from face-adjacent phi patterns (ties back to
  `project_dst_phi_diffusion` — the Sweet-Parker phi-diffusion picture is the
  one already known to produce reconnection; this gate redesign should
  converge toward that, not away from it).
- Before that redesign: worth checking directly *where* in the neutron the
  nonzero `o_k` values (however rare) are spatially located — if they cluster
  at the actual Hopf crossing region, that's a positive sign the concept is
  right and only the one-cell radius is wrong; if they're scattered/random,
  `opposed` may not be measuring linkage at all and a different raw variable
  is needed.
- This does not block `test_electron_dipole.py`/R1 work — the electron side
  of the redesign (coherent transit passing free) already has a working,
  confirmed gate (`opposed` or even `shear` at high ref). R2 (linkage
  detection) is the piece still open.

## Artefacts

- `probe_cross_v2_gates.py`, `cross_v2.py` — script copies at run time
- `run_output.txt` — full stdout, all three cross_scale arms
- `opposed_cs_{0.05,0.1,0.3}.npz` — per-arm (t, surv, rms, eng) histories
- `run.sh` — exact invocation
