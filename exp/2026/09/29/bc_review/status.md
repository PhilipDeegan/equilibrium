# Boundary-condition review — 2026-09-29 (inventory; no code changed)

Trigger: ambient-cooling probe found `DampedLattice.apply_sponge` breaks phi antisymmetry
(`exp/2026/09/29/ambient_cooling/status.md`). Philip: review BCs generally, discrete and continuous
possibly separately. Inventory by a read-only survey at HEAD 9f004ac; items marked (v) verified by
hand. Line numbers: lattice.py = equilibrium/substrate/lattice.py, AL = inc/eq/core/data/lattice/
adaptive_lattice.hpp, CAL = .../continuous/adaptive_lattice.hpp, bc.hpp, lattice.cpp = src/eq/lattice.cpp.

## Common facts
- Each face is stored on both cells; antisymmetry (phi_ki = -phi_ik) survives only because every
  stepper's dphi is symmetric. Nothing re-antisymmetrises. Any per-cell operation on phi breaks
  it for good, and the broken (symmetric) part is frozen by step().
- Boundary faces (nb < 0) are skipped in the u and phi updates of every stepper -> no BC = reflective
  zero-flux wall. Only `outflow` feeds a boundary face into u.
- AMR "dead" interior faces also have nb = -1, so every nb<0-based BC also catches them.

## Discrete model
| BC | where | acts on | antisym | dt | tests | notes |
|---|---|---|---|---|---|---|
| open wall | all steppers | - | ok | - | yes | reflective, conserves sum u |
| periodic (set_nb, `wire_periodic_bcc`) | adaptive.py:34-153 | interior faces | ok | - | yes (u_vol tol 1e-3) | BCC odd-period partners approximate; re-wire after adapt |
| numpy sponge `DampedLattice.apply_sponge` | lattice.py:947-970 (v) | u->u_bg, phi->phi_bg, per cell row | BROKEN | per call | many | load_snapshot never sets _u_bg; bc_flux ignores fixed_u |
| `RadiationBCLattice` | lattice.py:998-1109 | u->0 sponge + Dirichlet shell A/r | BROKEN | per call | none | not Sommerfeld despite name |
| C++ `quadratic_sponge` + step_sponge | AL:2721-2734 (v) | u, phi *= d -> 0 | BROKEN | per call | some | IGNORES target (numpy pulls to u_bg); adaptive test_saha assumes u0 |
| `quadratic_sponge_integrated` | AL:3246-3280, 3466-3495 | exp(-gamma dt) per row | BROKEN | rate | none | |
| `quadratic_sponge_directional` | AL:3502-3531 | phi per (leaf, face) | BROKEN | rate | none | measured worse than split and integrated |
| split-field PML | AL:3288-3334, 3540-3580 | aux fields | BROKEN | rate | none | beats split at R150 by 34%, loses at R65 |
| outflow (Mur 1st order) | bc.hpp:168-232 | boundary face, into u | n/a | ok | TestPulseOutflowBC | default c_lattice = sqrt10 (pre-09-28 speed!) |
| outflow_ledger / tapered / pml_outflow | bc.hpp, lattice.cpp | | | | none | ledger bit-identical to outflow; pml_outflow = outflow; tapered monopole only |
| fn: absorbing / dirichlet / fn-sponge; neumann; remote | bc.hpp:70-93 | boundary-face phi only | n/a | ok | characterisation | inert on u (known); neumann = no BC; dirichlet phi grows unbounded |
| face ghosts (wire_face) | bc.hpp:272-312 | one copy only | only if both wired | | none | not a periodic BC for u |
| injecting (test-local) | test_injecting_bc.py:80-84 | phi[13] one copy | deliberately broken | per call | qualitative | |
| ReciprocalBC (test-local, 2 copies) | test_saha.py:37-81 | u only | ok | per call | yes | alone ~35x worse than sponge (reflects) |

Discrete-path defects:
- `step_subcycle` never calls `_step_pass2_bc` (AL:1065 (v)): with TimeStepper.subcycle every
  fn / ghost / remote / outflow BC is silently off (vol_damp still works via step_sponge).
- BC arrays (vol_damp, gamma, proj_r, aux, outflow history) sized N at factory time, never remapped
  after split/coarsen; not persisted (cluster_io saves fn type/target/weight only).
- BCType / fn_type set but never read by any stepper.
- numpy step() uses cell sigma in the u update, C++ adaptive pass1 uses face sigma: sum u
  conservation differs when lam != 0.

## Continuous model
- C++: `_require_no_face_bc` throws for fn, ghosts, remote, outflow, taper, gamma, gamma_face
  (CAL:573-579); step_damped/dir/pml throw. Only `step_sponge` (CAL:512-528 (v)): same row-wise
  multiply toward 0 -> BROKEN antisymmetry, and it does NOT damp E (`an.e`) -> inside the sponge
  u and A decay while E does not.
- E boundary in step_ae: e = 0 on boundary faces; LS reconstruction pseudo-inverse on partial rows;
  DEC drops triangles with a missing edge -> truncated complex, not a designed BC.
- DEC `_dec_tri` reads ONE copy per edge: once a sponge has broken antisymmetry, the curl depends on
  which copy it reads.
- numpy continuous.py: no BC classes; AE e_mask holds E = 0 on boundary faces; edge_mask off
  (breaks energy); `advance()` picks up apply_sponge if lat is a DampedLattice.
- No tests of step_sponge + step_ae or of the throws.

## Cross-cutting
- dt dependence: split sponges damp a fixed fraction per call. The 09-28 weight change moved
  stable_dt 0.4/78 -> 0.4/30, so per unit time every split sponge is now ~2.6x weaker at the same
  damp_max. Candidate cause of the test_saha_eqiulibrium sponge failures and of the Coulomb
  fixed-window plateau shift (both use step-count / damp_max tuned before 09-28) -- unverified.
- outflow default c_lattice = sqrt10 is the pre-09-28 wave speed (now 2.309).
- No doc or memory records the antisymmetry breakage.
