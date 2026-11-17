# Entries for 2026-10-06 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-06 — N 16 recurs flat to 24k (rigid rotation, 'perfect' per Philip); structures don't lose energy (adiabatic) -> seed surplus is shed once; no quantization dip in r0 / a at N 2 (continuous family, ~18 % uniform seed-profile shed); ring rates follow Kelvin's law with the walk's own kappa; N 8 r0 120 holds 8 turns, its shed focuses into a counter-rotating axis ring that self-propels away; evening: the walk's own energy (quasi-energy + nonlinear term) is conserved, the gradient E is not -- in it the coil GAINS while the floor goes below background; a closed box suppresses the axis ring, r0 80 makes it transient; late: the '~18 % shed' = the seed's negative-branch admixture (sorted out) + its too-narrow core profile -- a + projected, relaxed-profile seed sheds ~nothing; the coil's energy within ~a + 7 xi is bound and spacing-independent, its density deficit sloshes

1. **N 16 r0 120 16k -> 24k** (`exp/2026/10/06/coil/status.md`): 16 / 1, recurrence vs relaxed t 4000 0.046-0.054 flat
   (within 0.04-0.06 since 9k), rigid rotation 11.75 deg / 1000 constant; Philip: "looks perfect" at 23k.
2. **In-run probes** in spring_cpp.py (10-06 copy, --probe_every 40): coil-only centre, 15 offset-circle windings, tube
   aspect, k-harmonics, axis ring + winding, off-structure census (rho < 0.6 pieces). probes_from<t>.npz.
3. **Deficit budget** (deficit_budget.py): total deficit = the seed's (norm conserved) and the floor inside the sponge
   swings +-150k -> only structure-local D is usable (N 16 +-2 % over 4-16k). D tracks the tube, E the turn count;
   E/D 0.03 (small N 1-3) / 0.045 (N 16) / 0.055 (N 30). 10-04 pairs: all same sense -> sign-blindness untestable.
4. **Philip: structures don't lose energy (vacuum exchange adiabatic); IC off quantized values -> unsupportable surplus
   shed; don't discount small quantized shares; but no particles -- don't reify** (mems project_structures_dont_lose_
   energy, feedback_reproduce_dont_reify). Sponge = one-way sink -> sponge-based loss readings conditional.
   Census: intact tight-box N 2 / 3 shed nothing countable; collapsed N 4-6 left 3-8 neutral rarefactions (E/D ~0.005).
5. **Quantization scan** (`exp/2026/10/06/quant`, N 2 P 220 8k): r0 20-40, a 6-14 -> no dip, tube radius kept, ~18 %
   one-off shed by 2k then flat (r0 20 keeps losing: squeezed hole) -> continuous family; offset = seed core profile.
6. **Rates vs geometry** (Philip: may calibrate a rate as 'the correct value'): N 2 helix rate vs R fits
   C (ln(8R / c) - 1/2) / R^2 to 2 %, C 0.260 vs the walk's kappa / 2 pi = 0.250 (unfitted) -> a rate picks R; the
   coefficient (circulation quantum h / M) is the classical-limit reproduction worth pegging (ties M to the tick).
7. **N 8 r0 120** (probes): 8 / 1 to 8k, no drop to 4 (above the floor). Shed focuses on the centre (nested shells,
   N8_nested_meridional.png) -> neutral rarefaction ring (2600) -> deepens -> winding +1 at 3680 (opposite the coil)
   -> ring r ~13 at 4000 (= N 16's axis anti-ring) -> self-propels along -z at 0.029 / tick (Kelvin with walk kappa
   0.025-0.034, unfitted) into the sponge. The anti-ring is how the surplus leaves.
8. Discussion: Dirac via Feynman's checkerboard (QED adds field quantization / renormalization, which we avoid);
   neutrinos = Weyl / PMNS oscillation, Dirac vs Majorana open (no confirmed Majorana particle).
9. **N 8 r0 80 periodic box** (`exp/2026/10/06/shed`, spring_box.py + the 10-06 probes, P 480 side 277, 16k ticks):
   8 / 1 throughout, minor 9.5-9.8, helix +0.90 rad / 500, rc breathes 80.0-81.2 (~5k). No axis core and no census
   piece at any probe -> no counter-ring, not even the precursor (Philip: "doesn't change much outside of rotating").
   (r0 120 does not fit a box with room: P 520 leaves 17 cells to the image; P ~640 ~75 GB.)
10. **The walk's energy** (shed/quasi_energy.py): two-tick parity symbol U(k) has doubly degenerate e^{-+i Om} ->
   i log U in closed form; E_q (exact linear invariant; --test g 0 exact) + E_nl = (g / 2) sum rho^2. Box run: sum
   -282.22..-282.41 over 16k (7e-4) while the gradient E of item 3 fell 8 % with no sink -> items 3-5's E (and the
   '~18 % shed') are gradient-E readings, not energy. Seed shed = 154 moved E_q -> E_nl by 1k, then flat.
   Region split: structure -4 -> +580 (1k) -> 650-730 breathing; floor -278 -> -862..-1012, i.e. BELOW the uniform
   background (bottom of the positive branch) -> the coil relaxes upward by emitting negative-quasi-energy content
   (negative branch? to check). No one-way trend after 1k.
11. **N 8 r0 80 sphere R 160** (coil/status.md, 5k): same coil; axis precursor at 2320, winding +1 2880-3360 (core r
   5-6 ~1.5 xi, -z at ~0.046 / tick), dissolves by 3960 inside the domain (not the sponge). -> r0 sets the counter-
   ring's size (r0 120: 8.5-9, persistent); the box (shed returns within ~500 ticks, images) suppresses its formation.
   Item 7's "anti-ring is how the surplus leaves" holds only for the sponge sphere at r0 120.
12. **Branch split** (`exp/2026/10/06/shed/status.md`, quasi_energy.py --branch): branch- norm 743 of the N 8 r0 80
   P 480 seed is a near-invariant (4e-4 over 16k at g != 0); the coil expels ~97 % of it by 1-2k (strands 365 -> 5)
   into the floor; E_- / N_- = -0.93 ~ -om0 (near-rest negative branch). -> the floor going 'below background' was this.
13. **+ projected seed** (`exp/2026/10/06/branch/status.md`, spring_box.py --project_plus; N 2 r0 30 a 10 P 220):
   branch- stays 1e-3 (g mixing only); coil diagnostics identical to the unprojected run (passive admixture). The coil
   still loses ~24 % of its structure E (313 -> ~240) as + branch content by 1k.
14. **Relaxed-profile reseed** (core_profile.py, --profile): relaxed core is fatter (rho 0.5 at d 5.3 vs 3.3; xi_eff ~5.2);
   seeded with it + projected: 258 -> 257 at 1k, then breathes +-15 around ~245 (net -6 %). Both seeds end at ~240-250.
   -> the '~18 % shed' = (i) branch- admixture + (ii) the narrow seed core; neither is the structure losing its own energy.
15. **Spacing scan** (`exp/2026/10/06/spacing/status.md`, N 2 r0 30 a 10, P 180-340): core deficit within ~a + 3.5 xi
   ~55k +- 6 %, no trend = bound; the rest of the seed deficit (grows with the box) spreads as a ~0.003-0.005 floor to the
   midplane (forced by norm). Relative to each box's own far level (midplane mean): halo closes at dt ~38 (~a + 7 xi),
   cum D 56-70k, cum e 210-320, no trend over P 220-340 (gap 63-133) -> bound extent, spacing-independent (an apparent
   +40 % energy zone at P 340 was the rho = 1 reference). Compensated seeds (D_tot 0): same profile within 1-3 % relative
   to the far level -> the seed's uncompensated D_tot did not set the halo; only the floor level moves.
   Clean seed (projected, relaxed core, no tail) P 180 / 200 / 220 / 340, 8 snapshots: ENERGY within dt 38 (~a + 7 xi)
   steady (sd 4-7 %) and spacing-independent (278-303, gap 41-133) = the bound quantity; DEFICIT there sloshes (sd
   10-21k of ~50k, energy-free density wave), swing larger / mean lower in tighter boxes = where neighbours act.
   (Single-snapshot and --ref_mid D comparisons above are phase / reference artefacts; branch/status.md CORRECTION.)
16. Library: equilibrium/substrate/walk_energy.py (symbols, quasi_energy, minus_branch, deg_norm) + tst/substrate/
   test_walk_energy.py (rest state E_q = om0 N and no branch-; P_- projector; g 0 E_q and branch- exact). Found: the
   50 / 50 split at Om ~ pi (U = -I) is not a projector -> those k now left out of P_- (exp copy unchanged; ~1e-6 there).


# Entries for 2026-10-06 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-06 — start here; read `now`'s 2026-10-05 entry, `exp/2026/10/05/coil/status.md`, `exp/2026/10/05/halo/status.md`

Memory: ONE job at a time (`ps -C python3`), r0 120 / R 200 runs ~43.5M cells ~25 GB, ~70 min / 8k ticks, states
2.8 GB each (prune per mem feedback_prune_bk_dst). Probes: centre-circle winding fails once the tube collapses or an
axis ring biases the centre -> recount with the coil-only centre over offset circles (winding_N30_check pattern).
1. **N 16 r0 120 long-term** (10-05: resumed to 16k, 16 / 1 kept, recurs to within 0.04-0.06 of relaxed t 4000 at
   9-16k after an 8k excursion -> near-periodic): resume 16k -> 24k (state_t16000) -- does the recurrence hold? Recount turns (coil-only probe), recurrence vs relaxed t 4000 / 8000, k 16
   and k 4 harmonics (square_orient pattern). If it holds, resume again to 24k (2 h job limit: 8k per job).
2. **In-run probes for spring_cpp.py**: every 20-50 ticks: coil-only centre (exclude r < 0.5 r0), winding over offset
   circles, tube aspect ratio, axis-ring cells + winding -> periodicity of the axis anti-ring (Philip: fine if it
   cycles) and of the breathing; no more 1000-tick aliasing.
3. **Validate the 4-turn attractor**: (a) N 8 / 12 on r0 120 (above the floor): do they also end at 4? (b) xi change
   (xi 5.2, scaled loop) -- does 4 survive? (c) radiation-cleaned seed of N 30 (one averaging pass) -- same end state?
4. **Floor boundary**: N 22 on r0 120 (spacing ~8 xi): sharp boundary or a delay curve (N 16 none / N 30 ~6k)?
5. **Deficit as lam** (no new runs): deficit budget (strands / tube interior / near halo / floor / anti-ring) for
   relaxed N 4 (halo/scf_free t25000) and N 16 t8000; deficit vs gradient energy across N (mass ~ deficit?);
   reread 10-04 pair runs for a sign-blind attraction.
6. **Halo test with a clean coil**: radial_profile.py on the free-after-clean N 4 states at P 300 and a P 450 twin
   (clean the P 450 t8000 the same way) -> an attached l = 0 part that is box-size independent = the Coulomb candidate.
Validations owed (tests, mem feedback_save_validations_as_tests): C++ solid-angle seed == numpy seed in spring_cpp /
spring_box (core cells, windings identical at t 0, checked by hand 10-05); scf_coil alignment sign check (rms 0.016
aligned vs 0.22 unaligned); free_after_clean E_far stays below the uncleaned level.
Ideas box (10-05, not queued): (i) sponge-distance check for isolation: N 8 r0 60 a 10 at R 120 (sponge 90) vs
R 160 (sponge 120), ~15 / 35 min per 8k -- if equal, the N 16 sphere result stands as 'isolated'; (ii) N 16 in a
periodic box P 520 (side 300, ~35M cells, tube gap ~40) = the COMPRESSED limit only (spacing / size ~1.15 vs ~1e5 in
metals, ~1e3 white dwarfs; Philip: electrons dislike being near each other) -- not a stand-in for isolation; depends
on whether the coil's tube or its halo is the electron's 'extent'.
Parked: high-lam centre (mem project_high_lam_centre_idea); imposed deformation m(x) (norm check first, mem
project_walk_imposed_deformation); fit / commensurability scan with a self-stable coil (N 2 / 3) at finer spacing.

---
