# 10-06 branch: does a + branch projected seed shed? (follows shed/status.md 'Branch split')

spring_box.py = spacing/spring_box.py + --project_plus (seed psi -> psi - P_- psi via shed/quasi_energy.py branch_split).
N 2 r0 30 a 10 P 220 m 0.8 xi 3.9 wxi parity, 8k; control = spacing/ P 220 (same seed, unprojected).
Projection removed branch- norm 192.19 of 2.59M (control: 121 in strands, 38 near, 10 interior at t 0).
compare_pp_P220.log (quasi_energy.py --regions / --branch, g 0.00822):
                     t 0     1k      2k      4k      8k
  control structure  +3.1    +168.1  +161.7  +152.7  +161.5   rest -67.1 -> -225.5   total -64.00 (conserved)
  control branch-    192.19  192.10  192.12  192.06  192.12   strands 121 -> 13, floor+outer 23 -> 155
  pp      structure  +313.1  +251.6  +229.5  +253.3  +237.8   rest -20.1 -> +55.3    total +292.99 (conserved)
  pp      branch-    1.6e-6  1.2e-3  1.2e-3  1.2e-3  1.3e-3   (g mixing only, flat after 1k)
Coil diagnostics at 8k identical to control (D_near 60276 vs 60236, helix -0.100 vs -0.101, core cells 11678 vs 11676).
Reading:
- The nonlinear turn barely mixes branches (1e-3 created, then flat) -> branch- norm is a near-invariant; the seed's
  branch- content was dynamically passive: removing it changes nothing in the coil, it only stops the wrong-branch
  content being sorted out to the floor (control's structure 'gain' of +160 = that sorting).
- The projected coil still relaxes: structure E 313 -> ~240 (-24 %, by 1k, then flat +-12), the rest +75, all + branch.
  So the seed surplus is real + branch content in the walk's own energy, ~1/4 of the structure's, one-off -- not an
  artefact of the branch admixture. Same size as the gradient-E '~18 %' of the quant scan.
-> Next: is the + branch shed the seed's core profile (item 3: relaxed-profile reseed, now with --project_plus)?

## Relaxed-profile reseed (next 10-07 item 3): core_profile.py + spring_box.py --profile --project_plus
core_profile.py: <rho> vs distance to the core-cell set (rho < 0.02, periodic tiling); check at t 0 reproduces the seed
form d^2 / (d^2 + xi^2) to <0.01 for d >= 2. Relaxed pp t 8000 core is fatter: rho 0.5 at d 5.25 (seed 3.3, i.e. effective
xi ~5.2 vs 3.9), far floor 0.993-0.996 -> profile_pp_N2_t8000.npz. Seeded with a0 sqrt(rho(d)) + projected (branch- 176):
qe_prof_pp_P220.log:  structure t 0 / 1k / 2k / 4k / 8k = 258.3 / 257.2 / 222.0 / 252.3 / 243.2, rest 16 -> 17-53 (total 274.49 conserved)
vs pp (seed form): 313 -> 252 / 229 / 253 / 238.
Reading: the + branch surplus was the seed's core profile. With the relaxed profile nothing leaves by 1k; after that the
structure breathes +-15 around ~245 (net -15, 6 %, inside the breathing). The coil ends at the same structure E (~240-250)
from both seeds -> a fixed relaxed state for N 2 r0 30 a 10 (r0 / a / N scan with the clean profile still owed: does the
quantization question look different with ~0 shed?).
Summary of the '~18 % shed': (i) branch- admixture of the seed (passive, sorted out to the floor; a gain for the coil in
E_q + E_nl), (ii) + branch excess of the seed's narrow core (~24 % of structure E), both one-off; neither is a loss of the
structure's own energy.

## Compensated seeds (next 10-07 item 0c; scan_comp.sh, --compensate --project_plus, P 180 / 220 / 260)
--compensate = uniform rescale of the projected seed to norm = cells (x 1.0446 at P 180) -> the floor sits ABOVE 1
(rho ~1.02-1.08) and the core is seeded at the same factor. shell_comp_P180.log (spacing/shell_profile.py, refs rho 1):
  t     D(dt<22)  midplane <1-rho>  far shell (dt 30-45) <1-rho>
  0     29.2k     -3.3e-2           -2.2e-2 .. -3.4e-2
  4k    45.1k     -4.7e-2           -3.4e-2 .. -5.0e-2
  8k    35.6k     -1.9e-2           -4.0e-2 .. -4.3e-2
Reading (P 180 only, so far): the core deficit (vs 1) is smaller than uncompensated (55k) and swings 45k / 36k; the excess
does not sit uniformly, it moves between a shell at dt 30-45 and the midplane (sloshing, 2 times only). The rho = 1 reference is
wrong for this case (e > 0 everywhere far out = the raised floor), so 'does the halo close locally' needs the profile
relative to the box's own far level -- shell_profile.py has no such reference yet. Not a result yet.

### Relative to the box's own far level (spacing/shell_profile.py --ref_mid: rho, e minus their midplane means)
shell_refmid_P180.log / _P220.log, comp (c) vs uncompensated (u), same P, t 4k / 6k / 8k:
  P 180 cum D(dt 30): c 71.7k / 56.4k / 39.8k   u 71.8k / 52.6k / 39.5k    midplane rho c 1.047 / 1.031 / 1.019, u 1.004 / 0.984 / 0.977
  P 220 cum D(dt 30): c 48.0k / 55.6k / 58.8k   u 49.4k / 53.9k / 59.0k    midplane rho c 1.012 / 1.016 / 1.018, u 0.987 / 0.987 / 0.992
  cum e(dt 30) P 220: c 323 / 282 / 268         u 282 / 315 / 274
Reading: relative to the surrounding level the structure is the same with and without compensation (D within 1-3 %);
compensation only offsets the global floor. The midplane level itself moves in time by the same amount in both
(P 180: -0.028 over 4k) -> a global slosh between the structure's halo and the inter-structure region, shared by the
pair of runs. P 220: cum D flat from dt ~30-38 -> the halo closes locally at ~a + 5-7 xi. P 180 (gap 41): halos overlap,
the region between sits at an excess relative to the midplane (cum D falls after dt 22). D_tot = 0 does not change the
local closure -> the seed's uncompensated D_tot was not what set the halo.

## Clean seed across the spacing (next 10-07 (A); scan_clean.sh; profile_pp_N2_t8000_tail1.npz = relaxed core, tail
## blended to 1 between d 20-40; --project_plus; shell_clean_P220_P340.log with --ref_mid)
  P    gap   cum D(dt 38) t 0 / 4k / 8k     cum e(dt 38) t 0 / 4k / 8k
  220  63    62.2k / 44.1k / 58.3k          275 / 334 / 281
  340  133   62.7k / 59.2k / 57.0k          283 / 278 / 292
Beyond dt 38: P 340 a 0.0015-0.004 residual (cum D 57k -> 64k by dt 70 at 8k), P 220 4k an excess shelf (-0.009, the
midplane low) -> sloshing waves, not stationary shells; little energy in them (cum e flat or falling past dt 38).
Reading: without any seed tail, the structure's deficit and energy within ~a + 7 xi agree between gap 63 and 133 to
2-4 % at 8k -> the bound extent is set by the structure, not by the spacing (for gaps >= ~60; P 180 overlaps). ~7-9 % of
the seed's local deficit moves outward as low-energy rarefaction waves while energy stays inside (a deficit-only
readjustment; the seed is still not exactly the state). Philip's 'Coulomb extent is bound to the structure' holds
for the walk coil at this geometry; where the long-range (1/r) part of a Coulomb field would live is not answered by
this -- the walk has no gauge sector (mem project_structures_dont_lose_energy).

### CORRECTION (time series, 8 snapshots 1k-8k; shell_clean_alltimes.log + the two above; P 180 / 200 clean added)
The 8k-only agreement above was a phase coincidence. Within dt 38, raw (rho 1 reference):
  P (gap)    E mean +- sd    D mean +- sd       midplane rho_ref sd
  180 (41)   287 +- 16       37.5k +- 21.5k     0.0092
  200 (53)   278 +- 12       44.7k +- 16.0k     0.0065
  220 (63)   281 +- 11       47.9k +- 11.8k     0.0040
  340 (133)  303 +- 20       50.2k +- 10.6k     0.0025
(--ref_mid makes it worse: rho_ref moves too, and x ~1M cells inside dt 38 it adds ~+-10k; corr(D_rel, e_rel) = -1.00 = the
reference, not physics. Do not use --ref_mid for time series; use raw or a time-averaged reference.)
Reading: the ENERGY within ~a + 7 xi is the bound quantity -- steady in time (4-7 %) and the same across gap 41-133
(+-8 %). The DEFICIT is not steady: it sloshes through dt 38 as a density wave carrying almost no energy, with a larger
swing and lower mean in tighter boxes (P 180 sd 21k vs P 340 10.6k) -> neighbours act on the density slosh, not on the
bound energy. Open: the slosh's period / origin (seed's remaining density mismatch? a breathing mode of the coil?).

### 10-07 note: the CORRECTION table above mixed references (its D values are --ref_mid-relative); raw D sloshes only
+-4-9 % and corr(D, e) = -1 is the (g/2) floor bookkeeping -- the bound quantity is Et = E + (g/2) D. See
exp/2026/10/07/slosh/status.md (refmid_to_raw.py, refmid_to_raw_10-06.log).
