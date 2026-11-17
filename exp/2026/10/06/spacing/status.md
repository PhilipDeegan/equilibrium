# 10-06 spacing: does the bound extent follow the box spacing? (next 10-07 item 0b)

spring_box.py = shed/spring_box.py (snap_root /bk/dst/2026/10/06/spacing). scan_P.sh: N 2 r0 30 a 10 m 0.8 xi 3.9 wxi parity,
P 180 / 220 / 260 / 300 / 340, 8k, states every 1000. shell_profile.py: per dt (distance to the tube centre circle)
<1 - rho> and the walk's energy density e = e_q + e_nl - background (shed/quasi_energy.py local split), own-Voronoi cells
(nearer the own tube than any of the 26 images'), plus the tube/image midplane. Logs shell_t8000_P180-260.log, _P300-340.log.

t 8000:
  P    gap x  D(dt<24)  cum e peak (dt)  e = 0 at dt  D total (= seed)  midplane <1-rho>
  180  41     54.9k     158 (22)         ~23          62.1k             2.3e-2 (overlap)
  220  63     58.5k     163 (26)         ~28          71.4k             7.9e-3
  260  86     53.9k     213 (22)         ~25          78.9k             6.9e-3
  300  109    59.3k     179 (26)         ~30          85.5k             2.8e-3
  340  133    52.5k     245 (30)         ~46          91.5k             3.7e-3
Reading: the core (D within ~a + 3.5 xi ~ 55k +- 6 %, no trend in P) is spacing-independent = bound. The rest of the
seed's deficit (7k -> 39k, growing with the box because the seed's 1 / d^2 tail does) spreads as a ~0.002-0.005 floor
out to the midplane; with norm fixed in a closed box that floor is forced by the seed, so 'extent follows spacing' for
the tail is not a measurement. Energy: positive-e zone ends at dt 23-30 for P 180-300; P 340 is wider (46) with a higher
peak (245). Repeat at t 4000 / 6000 (shell_t4-6k_P340_P220.log): P 340 cum e peak 231 / 222 (e > 0 to dt ~34-40),
P 220 156 / 169 (to ~26) -> not a breathing phase: the structure's positive-energy zone is wider and ~40 % larger at P 340.
So the energy extent does grow with the box while the core deficit does not -- but the seed tail (total deficit) grows too;
box vs seed not separated.
Discriminator = compensated seeds (branch/scan_comp.sh, D_tot 0).

## Correction: relative to the box's own far level (--ref_mid; shell_refmid_P300_P340.log, branch/shell_refmid_P*.log)
  P    cum D(dt 38) 4k / 8k   cum e(dt 38) 4k / 8k
  220  46.5k / 59.4k          319 / 275
  260  61.6k / 55.9k          310 / 242
  300  62.2k / 69.8k          260 / 210
  340  61.3k / 56.0k          271 / 301
No trend in P; spread = breathing. The '+40 % energy zone at P 340' above was the rho = 1 reference (each box's floor sits
at a different level, set by the seed's total deficit), not the structure. Beyond dt ~38 (~a + 7 xi) a 0.002-0.009
residual that changes between snapshots (sloshing, not stationary shells). -> Bound extent ~a + 7 xi, independent of the
spacing from gap 63 to 133 (P 180, gap 41: overlapping halos). Clean-seed pair (branch/scan_clean.sh) is the check
without a seed tail.
CAVEAT (later, branch/status.md CORRECTION): --ref_mid snapshot D values carry +-10k from the moving midplane level and
the deficit itself sloshes (sd 10-21k); only the ENERGY within dt 38 is a steady, spacing-free quantity (clean seeds:
278-303 over gap 41-133). The D columns above are not comparable snapshot to snapshot.
