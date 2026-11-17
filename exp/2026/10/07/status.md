# Entries for 2026-10-07 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-07 evening (/bk machine) — N 4 at the spacing floor does not drop a turn: it reels its helix in to a tight 4-turn twisted ring (settled ~40k, Et ~539, winding -4 kept), while N 4 at r0 40 holds wide; the relaxed coil core is ~3 % THINNER than GP in the same lattice metric (the "5.11" was a continuum shortcut); a relaxed seed rings ~1/3-1/2 as much (ringing ~ seed mismatch), breathing ~5.3k unchanged (coil-local); pair speed-up re-measured with fitted drifts: opposite sense = none net (symmetric +4.5 % only within ~25 in z of the pass), same sense = real and short-range, +8.9 / +3.5 / +1.3 % at d 120 / 150 / 180, exponential with length ~31 (not 1 / r); /bk/dst pruned 172 -> 96 GB

1. **Seed**: regenerated gp_profile_xi3.9.npz here (5.11 / 3.34, matches fedora). Fedora states not transferred ->
   N 4 runs restarted from t 0. All analysis is in tracked logs; scripts below are in exp/2026/10/07/{slosh,core}.
2. **N 4 to 40k** (slosh/status.md, scan_N4_long.sh, scan_N4_resume40k.sh, n4_long.py -> n4_long_10-07.log):
   r0 30 (inner spacing 8 xi): Et 632 -> 547 at 20k (leak -6.7 -> -1.4 / 1k), then flat 536-541 to 40k (slope -0.7,
   +1.0 / 1k); core-set rad 10.4 -> ~4.0-4.5 (~ one core width) and stops; centre-circle winding -4 throughout; k4
   modulation 9.8 -> 2.2-3.2 = still a 4-turn helix, wound tight. Settled Et below N 3 at 10k (557-566), near N 2 (529):
   at the floor extra turns cost little once tight. r0 40 (11 xi): Et 816 -> ~797-801, -0.8 / 1k, rad ~8.5, holds wide.
   -> the leak is set by turn spacing, not N; the floor does not quantize N down. Et / L along the contraction is not
   constant (2.0 -> 2.5 with L(a = rad)): "Et = strand length x const" holds across seeds only.
3. **Core, same metric** (core/status.md, core_compare.py -> core_compare_10-07.log): GP rho on the BCC lattice, d to
   the nearest rho < 0.02 cell: rho 0.5 at 5.43 (GP) vs 5.25 (relaxed); 0.9 at 13.49 vs 12.44. Relaxed is ~3 % thinner,
   inner d < 3 equal to < 0.01, a +0.02 excess over d 5-16. Not one xi rescale. GP stays the right seed to ~3 %.
4. **Relaxed vs GP seed** (slosh/status.md, scan_relaxed_seed.sh, seed_compare.py -> seed_compare_10-07.log; N 2 r0 30
   P 220 8k): relaxed seed ~0 excess deficit, 0-2k swing 7.3k vs 21.2k, later sd 2.7k vs 4.9k -> ringing ~ mismatch;
   first minimum both ~1.1k (box transit); breathing ~5.3-5.5k vs 5.1-5.3k -> coil-local, confirmed. ~2k residual not
   resolvable in 8k windows. End states differ (D_in 61.9k vs 69.4k, Et +1 %). Caveat: the relaxed npz's d is to the
   core SET, read as d to the strand centre (seeded ~0.6-0.95 thin).
5. **Pair, fitted drifts** (slosh/status.md, drift_fit.py -> drift_fit_10-07.log, pass_q-1.py -> pass_q-1_10-07.log):
   lone coil 0.0131 / tick independent of P 180-340 and seed (lone P 300 redundant); 2k windows swing +-2.5 % with the
   breathing. q -1 over 1k-8k = lone (0.01315); vs the lone at the same t: +0.00059 (+4.5 %) within |z-sep| < 25 of the
   partner, symmetric on approach and recession (not attraction-like), ~0 / slightly below far. q +1 rerun to 8k
   (identical to fedora's through 5.5k): +3.6 %, the same breathing curve shifted up. CORRECTS 10-07 "sense-blind".
6. **q +1 spacing scan** (scan_pair_q1_P.sh): d = P / 2 = 120 / 150 / 180 -> excess +0.00117 / +0.00046 / +0.00017
   (+8.9 / +3.5 / +1.3 %). Exponential, length 31 (~8 xi), fits to 1-2 % in log; power law needs d^-4.7 with
   inconsistent local exponents. Short-range (screened / evanescent?) same-sense coupling; 31 is not xi / sqrt2. Not a
   distance-only effect: lone P 180 has coaxial / lateral images at 104 and runs at lone speed.
7. **/bk/dst pruned** 172 -> 96 GB (intermediates of 10-06 branch / spacing / shed / coil, all finished 10-07 slosh runs;
   kept finals, N 4 r0 30 t20000, P 480 t8000 + t16000, the clean-seed state_t08000s). Not touched (needs Philip's OK):
   10-03 coil (named in next), 10-04 pair / pbox (mix45, 16 GB), 10-05 coil R200 intermediates (22 GB), halo / scf
   (5.6 GB), 10-06 coil N 8 R200 t4000.

## SESSION 2026-10-07 (fedora box) — the fat core is the GP vortex of the walk's own small-k equation (analytic clean seed, no saved state needed); the bound quantity is Et = E + (g/2) D (grand-potential form), steady to 0.2-0.5 %, not E (E sloshed through the (g/2) floor bookkeeping; corrects 10-06); the deficit slosh (+-4-9 %) = the seed's mismatch ringing between copies (first swing ~ spacing) + a coil-local ~2k oscillation and ~5k breathing; clean-seed r0 / a scan: no dip, Et = strand length x const (log in r0); in N at r0 30: N 1-3 settle, N 4 (8 xi spacing) leaks keeping its turns; spheromak / neutron decay under the walk parked as an idea

1. **Machine**: fedora box has no /bk, no 10-06 states, no 3D V-cache (mem reference_machine_fedora); snapshots ->
   ~/tmp/dst/2026/10/07 (26 GB, reproducible; Philip's 10-01 convention for this box). P 220 8k ~12 min, P 340 ~50 min
   with the in-run probes. npz / png are gitignored -> in-run series also written as tracked text logs
   (slosh/slosh_io.py -> slosh_<run>.log; slosh_plot.py / quant_table.py / pair_compare.py read them).
2. **(B) fat core** (`exp/2026/10/07/core/status.md`, gp_vortex.py): exact GP q = 1 vortex with the --walk_xi healing
   length: rho 0.5 at 5.11 from the core set vs relaxed coil 5.25 (3 %); seed d / sqrt(d^2 + xi^2) 3.34 (logged 3.3).
   -> the --walk_xi label is right, the seed envelope is ~sqrt2 too narrow; gp_profile_xi3.9.npz = analytic clean seed.
3. **Energy bookkeeping** (`exp/2026/10/07/slosh/status.md`, refmid_to_raw.py on the committed 10-06 logs): a uniform
   floor of deficit delta carries -(g/2) delta per cell, so deficit moving structure <-> floor moves E_in by -(g/2) dD.
   Et = E + (g/2) D is 2nd order in a uniform floor: within dt 38 Et 545 / 541 / 540 / 529 (gap 41 / 53 / 63 / 133),
   sd 0.2-0.5 % in time. Raw D sloshes +-4-9 % (not +-20-40 %: the 10-06 CORRECTION table mixed references);
   corr(D, E) = -1 is the bookkeeping, not a --ref_mid artefact. The slight Et rise in tight boxes is NOT yet a
   neighbour energy (item 4). Caveat: the (g/2) term assumes + branch content near rest.
4. **Slosh period** (spring_box.py 10-07 copy: in-run deficit shells every 20 ticks, energy every 100, H2 cached;
   GP seed P 180 / 220 / 340, slosh/status.md): the big swing is the seed's deficit mismatch ringing between the
   copies -- first D_in minimum at ~8.6 x side at all three (half the (1,1,0) sound period = 7.8 x side); the residual
   ~1800-2100 oscillation and the rc breathing (~5.1-5.3k) do NOT follow the box -> coil-local (the P 220 1971 vs 1981
   (1,1,0) match was a coincidence of that box). Et_in 534 / 529 / 519: at P 340 it drops 530 -> 518 while the
   emitted content is outside dt 38 -> the Et spacing trend is at least partly where the radiated seed mismatch sits.
   Separating needs 16k+ runs or a mismatch-free (relaxed) seed.
5. **Clean-seed r0 / a scan** (slosh/status.md, quant_table_10-07.log; GP seed N 2 P 220 4k): Et within a + 28 smooth
   in both -- 330 / 431 / 529 / 627 / 721 for r0 20-40 (r0 20 still falling: squeezed), Et / L(helix) 2.320-2.329 for
   a 6-14 at r0 30 (energy = strand length x a constant), growing ~log in r0 (~0.7-1.1 per ln; GP benchmark pi n / (3 m)
   0.785). No dip: N 2 is a continuous family in r0 / a, now without the seed artefact.
6. **Quant in N** at fixed geometry (r0 30 a 10 P 220, GP seed; N 3 / 4 resumed to 10k): Et within 38 ~489 / 529 / 566 /
   591 for N 1-4 (~+40 per turn); N 1-2 steady, N 3 nearly settled after ~3k, N 4 (inner turn spacing 8 xi = the 10-04
   pitch floor) leaks ~-8 / 1k to 10k (638 -> 560) keeping 4 turns -> a gradual leak at the spacing floor, not (yet) a
   step in N. Open: 20k+ for N 4 (drops a turn?), N 4 at r0 40 (does the floor move with r0?).
7. **CsCl-type pair** (slosh/status.md, pair_box.py; GP seed N 2 r0 30 a 10, P 300, partner at the body centre, lateral
   axis distance 122): q -1 counter-propagates at -0.0269 / tick (2 x lone), passes with no measurable deflection;
   q +1 co-moves exactly. Both pairs' coils run ~3-4 % faster than a lone coil (0.0133-0.0136 vs 0.0130-0.0131),
   sense-blind, not ordered by coil density -> candidate neighbour effect, preliminary (two-point drifts). Et_in q -1
   1.2 % lower (confounded: partner tubes sweep its own region). q +1 stopped at ~5.5k (task time limit), 1k-5k compared.
8. **Idea, parked (Philip 10-07: not the current priority)** -- spheromak / neutron decay under the walk. ONE model
   (Philip 10-07: two models for two things defeats a unified geometric model): step() / step_cross is a superseded
   rule (frozen divergence-free phi, 10-01), not a model kept for nuclear structures. Its results (Hopf face loops,
   barrier gate m0 2.49 / 0.16, tau_A / tau_B ~4.5, H / phi_rms decay; non-conservative step) are claims to reproduce
   in the walk or drop: to reproduce if real = a metastable linked state opening with a fixed budget + some radiation;
   artefacts = the gate, the rates. Same status for everything still on step() (proton knot, charge = u-flux, Bohr /
   hydrogen). In the walk, decay can only be conversion (Et conserved; structures + waves, fixed budget), topology
   changes only where cores meet (reconnection at density zeros, no gate). Gap that blocks posing it: the walk has no
   long-range field -> charge / Coulomb must come out of the walk itself (cf. item 3's bound extent), not a second sector.
   Expected break: lifetime -- GP benchmark linked loops untie in a few crossing times (Kleckner, Kauffman & Irvine
   2016; Proment et al. 2012; helicity partly kept, Scheeler et al. 2014) vs ~1e26 internal periods; a barrier would
   have to come from a relaxed linked state. Cheap test when wanted (~10 min P 220): Hopf-linked walk ring pair (phase
   Om1 / 2 + Om2 / 2) vs the same rings unlinked at the same separation: reconnection time, products (rings, windings,
   senses), Et split products / floor. Linked outliving unlinked by a large factor = the linking is a barrier.


# Entries for 2026-10-07 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-07 evening — back on the /bk machine; read `now`'s 2026-10-07 entry, `exp/2026/10/07/slosh/status.md`, `exp/2026/10/07/core/status.md`

Status of the 10-07 morning queue below: (B) DONE (fat core = GP vortex of the walk's small-k equation, 5.11 vs 5.25);
(A) DONE as far as 8k windows go (big swing = seed mismatch ringing between copies, first minimum ~8.6 x side; residual
~2k oscillation + ~5k breathing coil-local); 3 DONE (clean seed: no dip in r0 / a; in N at r0 30 a 10: N 1-3 settle,
N 4 leaks); 0c(ii) pair done, preliminary. The bound quantity is Et = E + (g/2) D, not E (10-06 corrected).
Run on the fedora box (no /bk, no V-cache); its states are in ~/tmp/dst/2026/10/07 there (26 GB, reproducible, not
transferred). Transfer = git: scripts + logs. npz / png are gitignored, so the in-run series are ALSO written as text
logs (slosh/slosh_io.py: slosh_<run>.log, read by slosh_plot.py / quant_table.py / pair_compare.py; spring_box.py /
pair_box.py dump at the end of a run, resumed segments merged). Before running anything there:
  - regenerate the seed profile (gitignored): cd exp/2026/10/07/core && python3 gp_vortex.py --save gp_profile_xi3.9.npz
    && cp gp_profile_xi3.9.npz ../slosh/
  - spring_box.py / pair_box.py --snap_root defaults to /bk/dst/2026/10/07/slosh when /bk/dst exists.
  - scan_*.sh are portable (cd to their own dir, PYTHONPATH = repo root); they run one job each, sequentially.

1. **N 4 at the spacing floor** (now 10-07 item 6): rerun N 4 r0 30 a 10 P 220 (GP seed, scan_N.sh form) to 20k+:
   does the ~-8 / 1k Et leak end by dropping a turn (to 3? 2?) or continue? Then N 4 at r0 40 (inner spacing 11 xi):
   does the leak go away when the turns are further apart (floor = spacing, not N)? Quantization lives here, if anywhere.
2. **Et on the saved 10-06 states (/bk/dst/2026/10/06)**: add the (g/2)(1 - rho) term as an option to
   spacing/shell_profile.py and shed/quasi_energy.py --regions (raw rho 1 reference, no --ref_mid) and redo the raw
   shell profiles of the clean-seed runs (branch/, P 180-340) directly instead of the log reconstruction
   (slosh/refmid_to_raw.py). Caveat: Et's (g/2) assumes the moving content is + branch near rest; the P 480 N 8 r0 80
   budget (shed/) moved branch- content (E_- / N_- ~ -om0) -> there split E_+ / E_- first (D_str 148k -> 177k: Et on
   the unprojected seed is not the right bookkeeping).
3. **Does Et's spacing trend survive?** Resume the 10-06 clean P 340 run (state_t08000 of
   prof/pp/springbox/.../P340/.../tail1) with the 10-07 spring_box.py (--resume, slosh probe) to 16k-24k: does Et_in
   come back from ~519-529 toward the P 220 value once the radiated seed mismatch returns? If the trend survives at
   16k+ = a neighbour energy (~+3 % at gap 41); if not = where the seed's radiation sits.
4. **Relaxed vs GP core directly**: core_profile.py's profile_pp_N2_t8000.npz (/bk machine, branch/) against
   core/gp_vortex.py in the same metric (distance to the rho < 0.02 core set) at all d, not only rho 0.5 -> where the
   3 % sits (curvature a / xi ~2.6? lattice dispersion?). Plus ringing amplitude vs seed mismatch: the relaxed-profile
   seed (profile_pp_N2_t8000_tail1.npz) at P 220 with the slosh probe vs the GP seed (~21k excess deficit) -> ringing
   ~ mismatch; the ~2k oscillation and ~5k breathing should not change (coil-local check).
5. **Pair, confirm** (now 10-07 item 7): fitted drifts (not two-point) + a lone coil in the same P 300 box -> is the
   ~3-4 % sense-blind speed-up real? Rerun the q +1 control to 8k (stopped at ~5.5k).
6. Older still open: 10-07 morning items 2 / 4 (counter-ring return, circulation quantum); parked: walk Hopf-pair
   decay test (now 10-07 item 8). Prune /bk/dst/2026/10/06 (keep the clean-seed state_t08000s and the P 480 16k final
   for item 2).

---

## QUEUED 2026-10-07 — start here; read `now`'s 2026-10-06 entry, `exp/2026/10/06/coil/status.md`, `exp/2026/10/06/shed/status.md`

Late 10-06 (now items 12-16): shed = branch- admixture + narrow seed core; clean seed = spring_box.py (exp/2026/10/06/branch)
--project_plus --profile profile_pp_N2_t8000.npz (sheds ~0). Use equilibrium/substrate/walk_energy.py for E_q / branch split.
Status of the items below: 0b DONE (energy within ~a + 7 xi spacing-free; deficit sloshes -- see (A));
0c DONE (branch/status.md: compensation leaves the halo unchanged vs the far level; shell_profile.py --ref_mid);
1 DONE (yes, negative-branch); 3 DONE for N 2 r0 30 a 10 (redo the r0 / a / N quant scan with the clean seed: does a dip
appear once the seed shed is gone? the profile is r0 30 a 10's -- check it transfers or relax per geometry).
New: (A) DONE (branch/status.md CORRECTION: energy within a + 7 xi steady + spacing-free over gap 41-133; the deficit
    sloshes, more in tighter boxes); next: the slosh -- period (state every 100-200 ticks, raw D(dt) time series),
    origin (seed density mismatch vs a coil breathing mode: does it decay at P 340 over 16k?), and whether it is what
    neighbours couple to (confinement-by-neighbours). Note 0b / 0c 'cum D' readings used --ref_mid snapshots -> redo raw.
(B) why the relaxed core is fatter (xi_eff 5.2 vs 3.9 -- the walk_xi label vs the walk's own healing length?).

Memory: ONE job at a time; r0 120 R 200 ~25 GB, ~70-75 min / 8k; P 480 box 55M cells ~32 GB, 4800 s / 16k; r0 80 R 160
~25 min / 5k; P 220 box ~5 min / 8k. Run length case by case (Philip 10-06).
Language: no particles -- results as structures and relations (Philip 10-06).
Energy: use shed/quasi_energy.py (E_q + E_nl, the walk's own energy; conserved to 7e-4 over 16k in the P 480 box), NOT
deficit_budget.py's gradient E (-8 % in a closed box = not conserved). Box states only (needs periodic FFT).
0. **Bound extent vs context** (Philip 10-06: a single isolated electron is spurious; QM's infinite tail is gibberish;
   the Coulomb extent is bound to the structure; something else generally compensates it). Bound vs shed, not
   compact vs spread: (a) shell profile (deficit + quasi-energy density vs distance from the tube, co-moving) over the
   saved P 480 states: stationary shells = bound; does the bound profile end where it meets the images' (~half the
   spacing, ~138)? (b) spacing scan (P 220 N 2 is cheap): does the extent follow the spacing? (c) compensation: the
   boxes are uncompensated (same-sense copies; D_tot = +266k, the seed removes norm) -> seed with D_tot = 0 (background
   raised to absorb the deficit) and / or an opposite-sense (q -1) partner in the cell; does the halo close locally?
   Which walk quantity is 'charge' (winding sense vs deficit) is open -- state results in both.
1. **Is the shed negative-branch?** In the walk's energy the N 8 r0 80 box coil GAINS ~650 and the floor goes below
   the uniform background (which is the bottom of the positive branch) -> split the floor's norm / energy into
   +- branches (P+- = (1 +- H2 / Om) / 2 per k) on the saved P 480 states (/bk/dst/2026/10/06/shed). Then redo the
   quant N 2 scan budgets (P 220 states) with quasi_energy.py: does the '~18 % shed' read the same way?
2. **Does the counter-ring come back?** (Philip: on its return, absorbed / merged / passes and survives?) Status:
   r0 80 box (P 480): no ring at all, not even the neutral precursor -> the closed box suppresses the axis focusing.
   r0 80 sphere: ring forms (precursor 2320, +1 2880-3360, core r 5-6 ~1.5 xi) and dissolves by 3960 in the interior
   -> r0 sets its size; too small to persist. Persistent case = r0 120 (r 8.5-9, steady 5k+), which needs P ~640
   (~130M cells, ~75 GB): does not fit. Options: elongated box (cpp_box with Px Py Pz; long along z, ring travels
   +-z), or scale r0 / a / xi down together (same r0 / xi as 120 / 3.9); or a sphere with the ring caught and
   re-injected (no -- one-way sponge, conditional).
3. **Reseed from a relaxed core profile** (scan offset ~18 % = seed profile): take a relaxed N 2 state's core
   radial profile (density + phase around the strand) and seed r0 / a variants with it -> shed ~0 everywhere?
   Then shed vs N at fixed geometry with the clean profile (quantization in N, not geometry). Measure with quasi_energy.py.
4. **Peg the circulation quantum**: verify the helix-phase = N x rotation bookkeeping; N 8 axis ring speed vs R
   against Kelvin with kappa = 2 pi n / (3m) and core xi (r0 120: 0.029 at R 8.5; r0 80: ~0.046 at R 5-6, +1 phase
   only ~500 ticks); same for N 2 rings in boxes. Then N 16's rotation (5x a thin-ring estimate). N 8 r0 80 box
   rotation: helix phase +0.90 rad / 500 ticks, rc breathes 80.0-81.2 (~5k cycle).
5. **Counter-ring formation, rest**: r0 dependence started (r0 80: smaller, transient); xi variation still open.
6. Older 10-06 queue items still open: 3 (N 12 / xi change / cleaned N 30 -> 4?), 4 (N 22 floor boundary),
   6 (halo test with a clean coil); validations owed (tests: quasi_energy.py --test g 0 exactness -> tst/);
   sign-blind attraction needs a q +1 / q -1 pair.

---
