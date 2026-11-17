# Entries for 2026-10-05 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-05 — v1/v2 counterflow has a finite threshold (~0.03); scale gap reframes all walk results as lattice-regime; copper toroid = electron picture (continuous helix vs tokamak separate loops = electron vs proton?); sponge was faking the halo; SCF averaging is a sink but one cleanup gives a near-equilibrium coil; N 16 on r0 120 clean to 8k; N 30 (below the spacing floor) collapses to a freely rotating 4-turn square; N 16 (above) recurs to 16k

1. **Counterflow** (`exp/2026/10/05/spin/status.md`; counterflow.py now --Pz): Philip "are results impacted by the
   size?" -- yes: cubic box ties v_rel and k_min (ratio 4 / Dn). Elongated / narrowed boxes: 0.057 rate 1.8e-3 is
   box-converged (Dn 4 / 8, transverse halved +5%); 0.0425 1.28e-3, 0.0354 9.3e-4, 0.028 flat at Dn 2 / 4 / 8 ->
   finite threshold ~0.03 (not zero); resolved rate ~0.031 v. Lattice regime.
2. **Scale gap** (Philip; mem project_scale_gap_forcing_coarse): cells ~1e-44 vs electron 1e-15..18; m > 0.71 = Compton
   ~1 cell -> coarse findings are lattice-regime unless they converge. 2D slab (Harris-style) proposed, then rejected
   (no dimensions; a slab removes the long modes that drive the counterflow instability).
3. **Ontology images** (repo root tokomak.jpeg, coil_top_down.jpeg; mem project_tokamak_toroid_ontology): toroid =
   field inside the core, windings wrap it, outside ~one-turn dipole; red tokamak rings secondary (Philip). Continuous
   helix (coil, electricity) vs separate loops (tokamak) ~ electron vs proton (Philip's lean). Ampere: N sets interior
   field ~ N and energy ~ N^2, exterior N-independent; ~22 turns a guesstimate; "electrons quantized regardless", then
   N may vary with extent; compressed lattice (shells under pressure) = where electrons are identical (mems
   project_target_standard_values, project_compressed_lattice_identical). Aim: standard theory values, one parameter set.
4. **Coil far field** (`exp/2026/10/05/coil/status.md`; spring_cpp.py C++ seed 3 s; far_field.py, interior_energy.py,
   radial_profile.py): seed exterior = one plain loop (residual falls with N); evolved R110 mismatch = sponge pinning
   (R160: loop dipole ~90%). Interior toroidal gradient x rho = N (Ampere flow confirmed). N^2 interior energy:
   2 valid points only, not established.
5. **Halo / equilibrium** (`exp/2026/10/05/halo/status.md`; Philip: background interaction may be the Coulomb mechanism,
   decay stays attached, uncompensated wavefunction -> infinity; mem project_attached_halo_wavefunction): sponge runs'
   l = 0 phase / density profiles stretch to the sponge = imposed. Periodic boxes (spring_box.py, C++ seed): structure
   boundary-insensitive; l = 0 phase = box clock drift (mean density), l = 0 deficit = flat floor ~1 / volume; E_far
   P 450 / P 300 = 1.15 for 3.4x volume -> mostly a one-off seed shedding + slow leak. No Coulomb-like profile.
   SCF by rotating-frame window averaging (scf_coil.py; k1 nearest-cell remap = noise, k8 IDW near structure only):
   shape converges but shedding rises, ring contracts -> averaging is a sink, dropped. ONE cleanup then free
   (free_after_clean.py): E_far 90-140 vs ~350 uncleaned, no regrowth -> near-equilibrium coil. Philip: the relaxed
   state may be more real than the IC (mem feedback_relaxed_state_more_real).
6. **Tight box N scan + spacing scan** (P 180, N 1-6; N 4 at P 180-300; summary/scan_N_and_spacing.png): N 1 = plain
   tilted ring, travels whole; N 2 / 3 keep open tubes; N 4 / 5 / 6 collapse to the same minor radius ~2.9 (tube
   collapse, NOT pinch-off -- Philip: N 24 never pinched). Tighter spacing collapses faster, smooth (no 'fit' window
   at ~11-cell steps). N / r0 limit = the 10-04 pitch floor restated; Bernoulli suction = candidate cause; walk coil
   collapses where copper bulges -> Philip: reciprocal, part of quantization; "only certain sizes fit together".
7. **Big loop r0 120, a 10, R 200** (43.5M cells, ~70 min / 8k): N 16 (11 xi spacing) holds 16 / 1 to 8k, round,
   rigid rotation ~12 deg / 1k (pattern period ~2k; 't 6k looks like IC' = whole periods), drift vs relaxed t 4000
   0.077 -> 0.166 by 8k; 8k mid-plane shows radial bars (copper-coil look; maybe plane offset). Axis anti-ring
   nucleates ~3.5k (winding +1 vs coil -1), shrinks by 5k (Philip: ignore the middle; fine if it cycles; idea parked:
   highest lam at the centre, mem project_high_lam_centre_idea). N 30 (6 xi): 30 to ~6k, collapse 7-10k, then a
   4-turn rounded square, steady 11-16k, rotating ~3 deg / 1k through the lattice axes (not locked) -> 4-turn
   attractor (also 10-03 N 8 -> 4; mem project_four_turn_attractor).
8. Side threads logged, not pursued: lam may be emergent from the walk deficit (mass ~ deficit, sign-blind attraction
   tests; mem project_lam_structure_unknown); deformation (gravity / time rate) may have to be imposed: per-cell m(x),
   unitarity of n-hop + m-local with varying m unchecked (mem project_walk_imposed_deformation).
9. /bk/dst pruned 149 -> 56 GB (rules: mem feedback_prune_bk_dst). Process: one bare '&' launch, killed after 4 s and
   relaunched tracked.


# Entries for 2026-10-05 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-05 (evening 10-04 update) — start here; read `exp/2026/10/04/pair/table.txt` (bottom: ITEM C + in-run zero count) and `exp/2026/10/04/spin/status.md`

10-04 evening done: A (separation scan: 8 / 20 apart pair survive, 40 ambiguous; E2 steady to 20k with in-run zero
count, no nucleation), C (cut_zeros.py: phase-winding zeros in the y = 0 cut, in-run in pair_box.py), B started
(spin branch = free walk, zero spin sound speed; relative v1/v2 flow unstable at rel. velocity >= 0.113, coflow stable).
Big correction: lone ring B does not just disperse, it sheds by an alternating shrinking cascade (new other-component
ring nucleates on axis, old shrinks out, 40 / 28 / 16 / 12 -> gone 12k); two like rings (D2) convert one ring to the
other component via ring/anti-ring nucleation -> churning v1 + v2 pair. Exclusion-like: like windings cannot share a component.
Next:
1. [DONE 10-05, exp/2026/10/05/spin/status.md: 0.057 was box-limited; with --Pz boxes a FINITE threshold ~0.03
   (0.028 flat at Dn 2 / 4 / 8, 0.035 grows); the 'N < ~3 per loop' estimate is withdrawn] counterflow.py.
2. Link to the ring: relative-flow profile around a lone ring vs the unstable range; growth time vs generation time.
3. Why the E2 pair is stable: relative flow field of the v1 + v2 pair vs separation (8 / 20 / 40) against the same range.
4. 40 apart to 20k with in-run zeros (ambiguous case). Mix 20 B run (does the cascade slow, as counterflow does).
5. Decisive intervention (only if 1-3 hold, ~half day): test-only small spin stiffness in step_walk (turn by
   g rho + gs (|a1|^2 - |a2|^2)-type split, i.e. g12 != g) -> lone ring should stop shedding / like rings stop converting
   once their relative flow is below the new threshold. If it still sheds, the mechanism is wrong. Not a rule proposal.
6. Electron poloidal windings IC = old item E, sharpened by 10-04: core = E2-type v1 + v2 pair (the only steady
   two-component ring), windings carried as phase turning N times along the loop in the core-filling component
   (Hopfion-type linking, not a density zero per turn -> no 8 xi pitch floor). Mechanism prediction: the twist IS a
   v1/v2 relative flow, q = N / (2 pi r0), rel. velocity (n / 3m) q; N 60 on r0 40 gives ~0.06 ~ the 0.057 edge
   -> the threshold from item 1 sets N_max per loop length (zero threshold => phase windings need compensating flow).
Parked idea (Philip 10-05, mem project_high_lam_centre_idea): highest lam at the coil centre to stop the axis
   bleed (N16 anti-ring) -- needs per-cell m(x) + norm check; control = radiation-cleaned seed.
Then the older items (D twist leak, E phase-carried windings = 6 above, F) below.

---

## QUEUED 2026-10-05 — start here; read `exp/2026/10/04/pair/table.txt`, `exp/2026/10/04/pbox/status.md`, `exp/2026/10/04/octagon/status.md`

Memory: ONE job at a time, `ps -C python3` before every launch, queue with `while kill -0 PID`; RAM quota ~40-50 GB
(P 450 ~21 GB, P 560 ~40 GB, P 300 ~5 GB). /bk 387 GB free. Seed is C++ now (P 450 ~1.5 min).
A. **Pairing separation scan**: E2 (v1 ring + v2 ring, same sign, coaxial) seeded 8 / 20 / 40 apart (P 300, 8k each,
   `cd exp/2026/10/04/pair && PYTHONPATH=/home/p/git/eq python3 -u pair_box.py --P 300 --ring 40,-10,1,1 --ring
   40,10,1,2 --T 8000`): if the pair dies beyond some separation while the 8-apart pair lives, the partner bounds
   the flooded core = real pairing. Also run E2 to 20k (does the pair stay compact).
B. **Why a lone component ring disperses**: core floods with the other component from t 0 (zero spin stiffness from
   the v1/v2-symmetric density turn?). Check the linear spin-wave spectrum of the mixed vacuum on the symbol (spin
   mode dispersion; any instability under relative flow); and B with mix != 45 (e.g. 20 deg).
C. **Piece counting**: switch pair diagnostics to local density minima in the y = 0 cut (close cores merge under the
   rho < 0.25 connectivity; C's split pair was counted as one).
D. **Twist leak vs lattice**: N 12 tight helix (P 300) with parity OFF vs ON -- faster L_z loss without parity =>
   pitch floor is the lattice leaking angular momentum (toroidal_tube.py). Then a third resolution step if needed.
E. **Phase-carried windings** (Philip 10-02 picture; pitch floor makes geometric N 60 unaffordable): seed one core
   with the partner component filling it and phase turning N times along the loop; first check the rule can hold v2
   in a core at all (B above says a lone component winding cannot).
F. Older: P 346 merge detail (states t12000 / t16000 kept), R125 parity 30k -> 40k (t30000 kept), neck approach law
   (R175 t5000 / R125 par t18000 kept), N 4 2-turn orbit past 40k (t40000 kept), R150 octagon resume (t15000 kept).

---
