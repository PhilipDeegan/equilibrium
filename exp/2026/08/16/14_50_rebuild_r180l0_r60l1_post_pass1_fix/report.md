# Rebuild L1R60 + uniform-R180 ICs post pass1/pass2 fix, 100-step first look

## Why

Follow-on from the 2026-08-13 rebuild
(`exp/2026/08/13/16_15_rebuild_ics_l1r60_uniform_r180/report.md`), which baked
in the stale-inter-parent-proxy fix but predates 2026-08-14/15's pass1/pass2
coefficient-parity fix (`now`, `doc/dst_amr.md`) -- pass1 was missing `1/L`,
using the wrong sigma average, and missing pass2's ancestor-substitution/
multiplicity logic entirely. That fix directly touches the machinery these
two checkpoints exercise (the L0/L1 junction in `l1_wider_margin_active`),
so a fresh rebuild+step bakes the new fix in at construction time and gives a
clean read on whether it changes real production behaviour, not just the
synthetic `TestFluxConservation`/`test_step_cross_per_level` tests.

Also switched the IC-seeding import from the per-experiment `dipole_mode_ic.py`
copy to the newly-promoted production module `equilibrium.substrate.
dipole_mode_ic` (see `project_dipole_mode_ic` memory) -- first real use of
that module outside its own test.

## Setup

Same parameters as 08-13/08-11: `cells_across=220`, `lam_scale=1.0`, L1R60 =
`r-l0=60.0`/`r-l1=46.667`, uniform R180 = `padding=70.0` (R_domain=180).
`cross_scale=1e-3`, `eta_h` off. Asymmetric thread split per user request:
L1R60 gets 6 threads (bigger mesh, 15.5M leaves), uniform R180 gets 3
(31.7M leaves but no AMR bookkeeping) -- sum=9 on the 8c/16t machine.

100 steps only (first look, not the full 300 the 08-13 run used), plotting
every 25. Compiled **release/NDEBUG** (`mkn build -Oqp mod_lattice -x mpi -g
0`) for the actual run -- per-step `check_total_dphi_balanced` is a full-
lattice checksum that runs on every single `step()` call under the
assertions-active build, so release is the right choice once the fix is
already validated by the C++/Python test suites (it is, per `now`).

New IC/output paths (kept `_pp1fix`-suffixed so the 08-13 checkpoints stay
available for comparison):
- `/bk/l1_wider_margin_active_ic220_pp1fix.h5`, stepped -> `/bk/l1_wider_margin_active_pp1fix_stepped/`
- `/bk/uniform_r180_active_ic220_pp1fix.h5`, stepped -> `/bk/uniform_r180_active_pp1fix_stepped/`
- Plots: `plots/L1_R60/`, `plots/uniform_r180/` (this dir)

Driver: `run.sh` (adapted from `exp/2026/08/13/.../run.sh`).

## Results

Both builds and both 100-step runs completed cleanly, rc=0, no NaN/blowup.
Wall time: builds 15:02:49-15:07:17 (~4.5 min), stepping 15:07:17-15:20:15
(~13 min) -- much faster than 08-13's ~3.5hr/300-step run (1/3 the steps,
plus release build skips the per-step debug checksum).

| | H0 | H/H0 @ step 100 |
|---|---|---|
| L1R60 (`l1_wider_margin_active`) | 4.462388e6 | 0.999415 |
| uniform R180 | 4.462518e6 | 0.998285 |

**Compare to 08-13 (pre-pass1-fix) @ step 130**: L1R60 was 0.912609 (matching
the junction-free `pure_l1size_control` reference of 0.912617 -- i.e. it was
decaying "correctly" relative to that control, but much faster than the
uniform-R180 case), uniform R180 was 0.999539 (essentially flat).

**Post-pass1-fix, the relationship has flipped**: L1R60 now decays *more
slowly* than uniform R180 (0.9994 vs 0.9983 at step 100), and both sit in the
same slow-decay regime -- L1R60's previously fast decay is gone. This is
consistent with `now`'s account of the fix: pass1 was missing `1/L` and using
the wrong sigma average, both of which would produce spurious extra leakage
specifically at the L0/L1 junction that `l1_wider_margin_active` exercises
and `uniform_r180_active` (no junction) does not. The fix changing L1R60's
*physical* decay rate this much (not just cleaning up a small residual) is a
stronger confirmation than the synthetic unit tests alone gave.

Plots (`plots/L1_R60/xz_step_100.png`, `plots/uniform_r180/xz_step_100.png`):
both show the expected dipole pattern (red/blue top/bottom lobes), a faint
reversed-sign ring at the construction edge (expected artefact, not new), and
clean circular log|u| falloff to floor with no fractal/quilt texture visible
at the L1R60 junction (r=46.67, inside the r<60 clip) -- consistent with
`project_dipole_mode_ic`'s earlier note that the new IC shows no fractal
texture at the L0/L1 junction.

## Continuation: 200 more steps (to step 300)

`continue_round.sh 200`, same 6/3 thread split, resumed from step 100.
Both rc=0, no blowup. H/H0 @ step 300: L1R60 0.998604, uniform R180
0.995181 — both still decaying slowly and smoothly, no acceleration.
Plots (`xz_step_300.png` in both plot dirs) show the same clean dipole
pattern, larger/fuzzier than step 100, reversed-sign ring more developed,
still no lobe-merging.

## Side investigation: why no PHARE-style flux-tube merging

User recalled PHARE's spheromak run shows the two lobes merge into a flux
tube quickly; DST's `u` plots above never show this (expansion+diffusion
only). Investigated using the checkpoints already on disk (no new sim
runs):

- Reconstructed an effective 3-vector from `leaf_phi`'s 14 face fluxes
  (weighted least-squares against `face_weight`, hex=3/sq=1 — closed form,
  `N^T W N = 10*I`) and streamplot'd it per-level
  (`plot_phi_vector_per_level.py`). Both t=0 and step 300 show independent
  radiating starbursts at each lobe, never a connecting loop.
- Read PHARE's actual reference data (`~/tmp/EM_B.vtkhdf`, a YZ slice of its
  spheromak B field) with a from-scratch VTK-Overlapping-AMR-HDF5 reader
  (`plot_phare_b_slice.py`) — found and worked around a real gotcha:
  `Steps/Values` is two IDENTICAL concatenated 35-step series, not one
  69-step run. PHARE's field shows a persistent closed loop directly linking
  the two Bx lobes at t=0, 1.4, AND 3.4 — real flux-tube topology.
- Root cause: `equilibrium.substrate.dipole_mode_ic.seed_dipole_ic` builds
  `phi = grad(U)` — a gradient field, curl-free by construction (Stokes),
  so it can never show closed-loop topology regardless of step count.
  PHARE's IC (`run/phare_plasmoses.py`, `get_taylor_spheromak`) uses the
  SAME K=4.493 eigenvalue but builds a genuinely rotational
  (Chandrasekhar-Kendall/Taylor-state) field via Cartesian cross products —
  the same construction our OLD (buggy, since-replaced) B-field IC
  attempted.
- Derived a coordinate-free discrete curl/circulation on the Kelvin lattice
  (12 elementary triangle loops per cell: 2 hex hops == 1 direct sq hop,
  verified 100% on real leaf data) and confirmed the crisp distinction:
  a discrete-derivative field (`phi[k,i]=u[nb[i,k]]-u[i]`) has EXACTLY zero
  circulation (float64 noise, pure telescoping) at any resolution, while
  `seed_dipole_ic`'s continuum-SAMPLED gradient has real nonzero circulation
  (~0.21 relative to phi's RMS) — a finite-hop curvature artifact, not
  evidence of rotational content. Full writeup, including a bug found+fixed
  in the diagnostic itself (sq-face index offset), in
  `project_dst_discrete_curl` memory.
- Promoted into permanent code: `equilibrium/substrate/discrete_curl.py` +
  `tst/substrate/adaptive/test_discrete_curl.py` (4 tests, all pass; full
  `tst/substrate/adaptive` suite re-run clean apart from one pre-existing,
  unrelated float-noise-level flake in `test_injecting_bc.py`).

## Follow-up: found the curl-full generator + confirmed the merging mechanism

`tst.substrate.adaptive.adaptive_flux_tube.init_toroidal_winding` +
`find_toroidal_ring` (predates this session, used in the older
`project_spheromak_decay`/`project_reconnection` work) already assigns phi
by ring-index only — coordinate-free. But its default cos-pattern turns out
to sum to exactly zero over one ring period (curl-free in disguise, same
"standing wave" pitfall flagged in `project_spheromak_decay`'s 2026-07-03
correction). The real counter-example: a CONSTANT phi value around the ring
(not cos) — genuinely nonzero net circulation (`phi_ring_circulation=42.15`
on a 24-tile test ring), no Cartesian formula.

Confirmed empirically: pure wave dynamics (`evolver.step()`, no
`step_cross`) conserves this circulation exactly (42.14859 unchanged over 10
steps, H also exact) — a Kelvin-circulation-conservation theorem, matching
the analytic prediction that `pass2`'s `dphi/dt=-σ·grad(u)` always has zero
circulation. `step_cross` instead erodes it monotonically (42.15→40.72 over
10 steps) with H dissipating alongside — confirming `step_cross` is the
DST-native resistive/reconnection mechanism. The dipole IC never shows
merging not because DST lacks the mechanism, but because it starts with
zero circulation — nothing to erode. Full detail + numbers in
`project_dst_discrete_curl` memory; saved as
`tst/substrate/adaptive/test_ring_circulation_conservation.py` (4 tests, all
pass).

**Not yet done**: combining two such genuinely curl-full rings into a
Hopf-linked pair (`find_toroidal_spheromak`) at CURRENT production scale
(dipole_mode_ic-style self-consistency, current pass1/pass2/junction fixes,
cells_across=220 telescope) to actually attempt PHARE-style merging.

## Next steps

- `now`'s flagged re-check of the L4/L5 fractal/quilt texture lead against a
  rebuilt checkpoint (separate from these two R180/R60 configs) is still
  open, not touched by this run.
- Blackbody/UV-catastrophe spectral test idea flagged by user, explicitly
  deferred (`project_dst_blackbody_uv_catastrophe` memory) — not started.
