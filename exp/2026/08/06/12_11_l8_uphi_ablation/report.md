# u/phi ablation of the L8 base-update blowup — `u` alone reproduces (and exceeds) it, `phi` alone produces none

## Context

Part 27 (`exp/2026/08/06/11_26_l8_ancestor_debias_ruled_out`) ruled out the ancestor-substitution
Jensen-bias correction as the L8 blowup's driver and proposed, as the next lead, that `phi` (which
has real domain-wide structure in the real spheromak IC, unlike `u`'s near-zero background) might
be the actual driver via some analogous mechanism. This experiment tests that directly.

## Method

Built the real spheromak telescope/IC exactly as `refined_neutron_decay.py` does
(`init_spheromak_ic_refinable`, same params: `R_sph=1.0`, `B0=1.0`, `lam_target=1.0`, `u_max=1.0`,
`amplitude=1.0`, `sigma=None`), at `max_level` 4/6/8 with the same `--scale-override`-equivalent
calibration. Three conditions per level, one bare `evolver.step()` call each (no eta_h/cross_scale/
lam-step — pure base-update mechanics):
- **full**: `u` and `phi` both as constructed (the real IC)
- **u=0**: `phi`+`lam` as constructed, `u` force-zeroed everywhere right after IC
- **phi=0**: `u`+`lam` as constructed, `phi` force-zeroed everywhere right after IC

**One real bug caught along the way**: the first attempt at this ablation used
`init_spheromak_ic_refinable(..., amplitude=0.0, ...)` in the test harness — but `phi` is only
seeded `if amplitude > 0.0` (see the function itself, `refined_neutron_decay.py`); the real
script's own CLI default is `--amplitude 1.0`. With `amplitude=0.0`, `phi` was silently all-zero
in every condition regardless of the `zero_phi` flag, making that first attempt's "phi=0 identical
to full" result meaningless (both had phi=0). Caught by directly checking `phi0` rms before/after
the zero calls in isolation, not by trusting the aggregate output. Fixed and rerun (below). The
same `amplitude=0.0` bug was present in Part 27's supplementary raw-array diff-check script — but
Part 27's *primary* finding (bit-identical `H/H0=134.0626` via the actual CLI script) used the
script's real default and was unaffected; only the secondary magnitude characterization needed
re-verification (done here too, see Results — unchanged).

## Results

| level | full (u+phi) | u=0 (phi+lam only) | phi=0 (u+lam only) |
|---|---|---|---|
| 4 | H/H0=1.0347 | H/H0=0.9998 | H/H0=1.1189 |
| 6 | H/H0=3.6621 | H/H0=0.9978 | H/H0=10.0881 |
| 8 | H/H0=**134.0728** | H/H0=0.9833 | H/H0=**455.0812** |

(L8 `full` matches the independently-reproduced repro, `H/H0=134.0626`/`134.0728`, to 4 sig figs —
harness is consistent with the primary repro.)

Also re-verified the Part 27 debias-correction magnitude with the `amplitude` bug fixed: still
`H/H0(off)=457.455`, `H/H0(on)=457.455` (bit-identical to displayed precision), relative `u`
magnitude `2.45e-5` — unchanged from Part 27's number. The bug did not affect that conclusion.

## Conclusion

**`phi` alone (zero `u` background) produces essentially NO growth at any depth** (`H/H0` stays
within 2% of 1.0 at L4/L6/L8) — this **rules out `phi` as the driver**, reversing the hypothesis
Part 27 proposed.

**`u` alone (zero `phi`) not only reproduces the blowup, it EXCEEDS the full (u+phi) case** — at
L8, `u`-alone gives `455x` vs the full case's `134x`. **`phi`'s presence actually damps the
blowup**, it isn't neutral or contributing — the full run is consistently the smallest of the
three nonzero conditions at every depth (L4: 1.03 < 1.12; L6: 3.66 < 10.09; L8: 134 < 455).

**This decisively redirects the investigation back to pure-`u` dynamics at the coarse-fine
boundary**, not any phi-side mechanism. Combined with Part 27 (the `u`-curvature ancestor-
substitution Jensen bias is real but ~1e-5-relative, far too small to explain this), the actual
mechanism must be elsewhere in how `u` itself is updated across a boundary — most likely
`project_odd27_amr_junction_leak` Part 3's already-flagged, never-root-caused lead:
`disable_same_level_flux_at_boundary_only` gave a clean ~13x single-step-growth reduction at both
L4 and L6 when that investigation was active, localizing (not yet root-causing) the dominant
mechanism to same-level flux specifically at coarse-fine boundary cells. That lead was set aside
in favor of eta_h mitigation work and never revisited at the root-cause level — it should be the
next thing tried, now that phi and the ancestor-substitution bias are both cleanly ruled out.

**Also worth investigating, given phi actively damps rather than being inert**: since including
`phi` measurably reduces the blowup relative to `u`-alone at every depth tested, whatever
`phi`-related coupling exists in the base update (most likely via `_step_pass1`'s `u += sum
W_k*phi_k` term) is providing a real, if insufficient, negative-feedback/damping effect already —
worth keeping in mind if a fix is later found for the `u`-side mechanism, since removing/altering
`phi` coupling as part of a fix could inadvertently remove this partial mitigation too.
