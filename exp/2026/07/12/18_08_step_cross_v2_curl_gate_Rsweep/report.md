# step_cross v2 — curl/holonomy gate, knot-scale sweep (confound caveat NOT resolved — corrected below)

## CORRECTION (same day, after user question)

The conclusion below ("confound ruled out") is **wrong** and retracted. It
checked the anisotropy-safety of `scale` (the knot's overall sphere radius),
but the actual local feature the curl/opposed gates sample is the poloidal
double-ring's minor (cross-sectional) width, which is a much smaller fraction
of `scale`: FWHM ≈ 0.562·scale cells around the ρ≈0.61·R_sph flux peak.
Computed minor-widths: scale=3.0 (R15) → 1.69 cells; scale=5.1 (R30) → 2.87
cells; scale=4.8 (R80, incomplete) → 2.70 cells. **All three are below the
3.6-cell anisotropy-safe threshold.** Growing `scale` 3.0→5.1 grew the sphere
radius 1.7x but left the actually-relevant feature just as unresolved. The
near-identical curl failure curve at R=30 vs R=15 is therefore equally
consistent with "still-anisotropy-confounded" as with "genuine structural
failure" — **the question is reopened, not settled**. See
[[project_step_cross_v2]] for the corrected memory entry. To actually clear
the band, minor-width>3.6 needs scale≳6.4 — e.g. R=30 with `--sph-frac`
≈0.22-0.25 (not the 0.17 used here), or a larger R at a similar frac.

## Motivation

Follow-up to `exp/2026/07/12/13_57_step_cross_v2_curl_gate/`. That run tested
the `curl` gate at R=15/scale=3.0 and found it fails like shear/product
(suppress-everything, no e/n discrimination), but flagged an unresolved
caveat: per [[project_bcc_dispersion_anisotropy]], the BCC dispersion relation
is anisotropic at feature wavelengths ~1.5-8 cells (negligible only above
~3.6). The `curl` plaquette is a single-hop loop regardless of knot size, so
the negative result could have been an artefact of measuring curl at ~1-cell
wavelength — squarely in the worst part of the anisotropic band — rather than
a genuine verdict on holonomy-style gates.

This run retests at a larger absolute knot scale to clear that band, with the
scale held fixed by `--sph-frac` (`scale = R * sph_frac`) rather than by
adding a multi-hop loop to the gate itself — testing whether moving the
*knot* out of the anisotropic regime changes curl's verdict, since the gate
code (`cross_v2.py`) still only implements the single-hop Cartesian plaquette.

Two runs were queued: R=30 (sph_frac=0.17 → scale=5.1) and R=80
(sph_frac=0.06 → scale=4.8, N=2,784,735 cells). **R=80 was killed by the user
for taking too long** (only reached electron/v1 step 3000 before being cut).
R=30 completed. Reconstructed and relaunched after an accidental session kill
mid-run — audit-phase output matched the killed run byte-for-byte, confirming
the reconstructed CLI flags were exact.

## Run

`probe_cross_v2_gates.py --R 30 --sph-frac 0.17 --gates v1,opposed,curl`
(cs=0.01, 4000 steps, n_audit=600, refs=auto). Full output: `output_R30.log`.
Data/plot: `gate_sweep_R30.npz` / `gate_sweep_R30.png`.

## Result

| gate | surv_e | surv_n | separation e/n |
|---|---|---|---|
| v1 | 0.2698 | 0.6553 | 0.41 |
| opposed/ref=0.00e+00 | 1.0013 | 1.0000 | 1.00 |
| curl/ref=9.00e-33 | 0.2691 | 0.6553 | 0.41 |
| curl/ref=1.96e-02 | 0.8977 | 0.8349 | 1.08 |
| curl/ref=5.89e-02 | 0.9495 | 0.9032 | 1.05 |

**The BCC-anisotropy confound is ruled out.** At R=15/scale=3.0, curl's
separation moved 0.46→1.00→1.04 as ref increased. At R=30/scale=5.1 —
comfortably clear of the anisotropic band — curl does the same thing:
0.41→1.08→1.05. Near-identical shape and magnitude despite the knot being
~1.7x larger and outside the flagged wavelength range. Raising `ref` still
moves surv_e and surv_n up together, not apart. The negative curl result from
the earlier run was **not** a measurement-loop artefact — it's a genuine
structural failure, consistent with the §4 argument in
[[project_step_cross_v2]]: any gate built from local/single-hop quantities
(one cell's own phi vs. one neighbour's, or now one plaquette vs. its
neighbour's) is blind to the one-ring-out linkage signature regardless of
knot size.

`opposed` also reproduces its R15 pattern at R=30: R1 still solved
(surv_e≈1.001, frozen) but R2 unchanged-failed (surv_n≈1.000, also frozen) —
confirms that failure mode is structural too, not an R15-specific artefact.

## Conclusion / next steps

Holonomy-style gates built from a single fixed-radius plaquette are ruled out
as a source of R2 discrimination — not because of lattice anisotropy, but
because they're still fundamentally a local (one-ring) comparison. Per the
open options already on record in [[project_step_cross_v2]]:

1. A loop literally threading BOTH windings' interpenetration region (built
   from the crossing geometry itself), not a fixed Cartesian plaquette — a
   qualitatively different construction, not just "bigger."
2. Build the linkage detector directly from face-adjacent phi patterns per
   [[project_dst_phi_diffusion]] (Sweet-Parker-style diffusion — already known
   to reconnect); converge toward that mechanism instead of a novel gate.

R=80 confirmation run not completed (killed for cost) — not needed to settle
the confound question, since R=30 already clears the anisotropic band and
reproduces the R=15 curve shape almost exactly.

## Artefacts

- `cross_v2.py`, `probe_cross_v2_gates.py` — code at time of run
- `output_R30.log` — full run output (audit + all 7 arms × 2 ICs)
- `gate_sweep_R30.png`, `gate_sweep_R30.npz` — plot / raw survival histories
- `run.sh` — reconstructed launch command (R=80 command included, commented,
  for reference)
