# eta_h_per_level re-earn against the Part-12-fixed .so (L1-L6)

Goal: Part 12 (2026-07-26) fixed a real hex/sq face-weight-mixing bug in the
Odd27 AMR correction formulas (`_step_pass2_amr_correct` and friends mixed
hex-face weight 3 and sq-face weight 1 incorrectly in one correction group).
Every `--eta-h-per-level` hyper-resistivity value previously bisected (H/H0
boundedness sweeps) was calibrated against the buggy pre-fix code, so all of
them needed re-validating against the fixed `.so` — level by level, each
holding all shallower levels fixed at their own just-confirmed value, per the
methodology in `doc/dst_amr.md`.

Method for each level: 150-step candidate sweep first (cheap, picks a
sweet-spot-before-blowup value from H/H0@150), then a 500-step durability
check on the winner to confirm no delayed blowup. Full recipe (base command,
`--scale-override`/`--scale-ref-level` matched-footprint technique) in
`doc/dst_amr.md`; see `[[project_odd27_amr_junction_leak]]` Part 15 for the
running session log this experiment summarizes.

## Results

| level | pre-fix value | re-earned value | change | H/H0@500 (or @150 if durability pending) |
|---|---|---|---|---|
| L1 | `0` | `0` | none | — (no hyper-resistivity needed at this level) |
| L2 | `1e-5` | `1e-5` | **unchanged** | 1.4561 (≈identical to pre-fix 1.456) |
| L3 | `3e-7` | `1e-7` | shifted slightly | 1.5933 (`3e-7`=1.6028, statistically indistinguishable; cliff still 5e-6 stable/7e-6 blowup) |
| L4 | `3e-9` | `3e-9` | **unchanged** | 1.5967 (identical to pre-fix; cliff still 3e-8 stable/7e-8 blowup) |
| L5 | `3e-11` | `1e-10` | shifted ~3x | bounded oscillation 2.06-5.67 over 500 steps, no blowup (cliff 7e-10 stable/1e-9 blowup) |
| L6 | `1e-12` | `1e-12` | **unchanged** | 4.1742 @150 (500-step durability check not yet run) |

L6 150-step sweep detail (H/H0@150, `tmp/eta_sweep/l6_h_boundedness_reearn/eta_<c>.log`):
baseline `0`=13.0244, `1e-13`=8.2196, `3e-13`=5.2542, `1e-12`=4.1742 (best),
`3e-12`=4.7241 (still stable, worse than best), `1e-11`=37.7572 (blowup —
worse than even the uncorrected baseline). Cliff at the exact same location
as pre-fix (3e-12 stable / 1e-11 diverging).

## Finding

The Part 12 weight-mixing fix, despite being a real correctness bug, left
**most** levels' hyper-resistivity calibration essentially untouched — L2, L4,
and L6 came back numerically identical (or near enough) to their pre-fix
values, and L3 shifted only slightly. Only L5 moved meaningfully (~3x). This
suggests the weight-mixing bug's effect on the H-growth instability mechanism
was small relative to the dominant deep-telescope growth mechanism itself
(see `doc/dst_amr.md`) for most levels, but not negligible at L5 specifically
— worth keeping in mind if a similar future fix needs re-validating: don't
assume uniform shift across levels, but also don't expect every level to move.

## Remaining work (tracked in `now`, not repeated here)

- L6 500-step durability check on `1e-12` (queued next).
- L8 `--eta-h-ratio` re-earn — not started, lowest priority, isolated
  per-level array needed since uniform `s=2` scaling that works at L6
  diverges at L8 (pre-fix finding).

## Artefacts

- `run.sh` — representative commands for each level (L2-L4 were run as
  individual ad hoc commands, not saved scripts; L5/L6 used saved sweep
  scripts, copied here as `l5_run_sweep.sh` / `l6_run_sweep.sh`).
- `refined_neutron_decay.py` — snapshot of the driver script at time of this
  experiment.
- Full raw logs remain in `tmp/eta_sweep/l{2,3,4,5,6}_h_boundedness_reearn/`
  (not copied here — large and already durable in-place).
