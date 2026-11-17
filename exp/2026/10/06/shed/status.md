# 10-06 shed: does the shed come back? (next 1 / 4)

spring_box.py = quant/spring_box.py + the coil 10-06 in-run probes (coil-only centre, 15 offset-circle windings, axis
ring with z relative to the coil (zrel), census with periodic adjacency), z distances periodic.
r0 120 does not fit a box with room (P 520: 17 cells to the lateral image, 70M cells ~40 GB) -> N 8 r0 80 a 10 P 480
(side 277, lateral gap 73, 55M cells), m 0.8 xi 3.9 wxi parity, recentre 500, states every 1000 in /bk/dst/2026/10/06/shed.
Also answers next 4 (counter-ring at r0 80: same r ~13 / +1, or geometry-set?).
Log: box_N8_r80_P480.log

## Result N 8 r0 80 P 480, 16000 ticks (4793 s)
8 / 1 windings on all 15 circles throughout; minor 9.5-9.8, aspect 1.02-1.14, min rho on circles 0.29-0.40; helix phase
+0.90 rad / 500 ticks (pattern period ~3.5k), rc breathes 80.0-81.2 (~5k cycle); D_near 148k -> 181k at 500, then 166-176k.
Axis core 0 at every probe; census: no off-structure piece (rho < 0.6, >= 10 cells) at any probe.
-> no counter-rotating ring: the shed stays diffuse in the box, so 'does the ring return' is not posed here.
Not separated: r0 80 geometry vs box (shed returns within ~500 ticks at sound speed; images). Discriminator: N 8 r0 80
in the sphere (R 160, sponge at 120, ~22M cells).

## Budget (budget_N8_r80_P480.log; quant/deficit_budget.py --box 277.1)
  t      0     1k    2k    4k    8k    12k   16k
  E_str  7443  6237  6272  6273  6275  6212  6192   (structure = interior + strands + near)
  E_flr  1281  1510  1531  1545  1536  1540  1558
  E_tot  10802 9943  10006 9970  9987  9948  9947
  D_str  148k  177k  170k  171k  166k  174k  176k   (D_tot 266063 exact: norm conserved)
Reading: one-off shed 16 % of structure E by 1k (N 2 quant scan: ~18 %), then flat (-1.3 % over 8k-16k, the quant
scan's +-1.3 % band). No return into the coil over 15k ticks in a closed box; the shed sits as a floor of
~2.5e-3 deficit per cell. Caveat: the gradient E is NOT conserved in the closed box (total -8 % in the first 1k, no sink)
-> it is not the walk's conserved energy; the ~860 lost from E_tot went into a term this budget does not count. Any
'energy returns / doesn't return' reading needs the walk's actual invariant first.

## The walk's invariant (quasi_energy.py): E_q (exact linear invariant, closed form for the 2-tick parity symbol:
## i log U = Om / sin Om * i (U - U^dag) / 2) + E_nl = (g / 2) sum rho^2, both relative to the background
--test (P 48, rough random-phase state): g 0 -> E_q exact to 1e-6; g 0.0082 / 0.05 -> E_q + E_nl within ~10 % of the exchange.
N 8 r0 80 P 480 (qe_N8_r80_P480.log), g 0.00822:
  t      0     1k     2k     4k     8k     12k    16k
  E_q    631.3  477.2  488.8  481.9  485.1  477.8  477.6
  E_nl  -913.5 -759.5 -771.2 -764.2 -767.5 -760.1 -759.8
  sum   -282.22 -282.28 -282.41 -282.29 -282.37 -282.37 -282.26   -> conserved to 7e-4 of the sum, 1e-3 of the exchange
Reading: E_q + E_nl is the walk's energy here (the gradient E's -8 % was not a loss). The seed shed = a one-off transfer
of 154 from E_q to E_nl by 1k (structure's density deficit relaxes), then both flat to 16k: nothing comes back, nothing
leaves (closed box, no sink). Region split (which part is in the coil vs the floor): qe_regions_N8_r80_P480.log.
Region split (qe_regions_N8_r80_P480.log; per-cell e(x) = Re psi^dag (H2 psi) / 2 - om0 rho; H2 nonlocal -> split is
a convention near region edges), structure = interior + strands + near, sum = E_q + E_nl relative to the background:
  t          0      1k     2k     8k     16k
  structure  -3.9   +579.5 +648.2 +729.3 +667.2
  rest       -278.3 -861.8 -930.6 -1011.7 -949.5    (floor E_q -349 / -670 / -915 / -818 from +74)
Reading: in the walk's own energy the coil does NOT lose energy -- it GAINS ~650-730 and the floor goes NEGATIVE relative
to the uniform background (the background is the bottom of the positive branch, om0 = arcsin m; going below it
needs negative-branch (-om) content). So the 'shed' = the coil emitting negative-quasi-energy waves while it relaxes
upward; total fixed. Then structure 580 -> 730 (8k) -> 667 (16k): breathing with the floor, no one-way trend.
To check: positive / negative branch norm split (P+- = (1 +- H2/Om) / 2 per k) of the floor -> is the shed
negative-branch?

## Box vs sphere at r0 80 (coil/status.md): the sphere forms the precursor ring at 2320, cored +1 2880-3360, dissolves
by 3960 (r ~6 ~ 1.5 xi); the box never forms the precursor -> the closed box suppresses the axis focusing. So the
'does the counter-ring return' test needs a geometry where it persists (r0 120: r 8.5-9, +1, steady 5k+) in a box
with room: P ~ 640+ (side 370, lateral gap ~85), ~130M cells ~75 GB -- does not fit (60 GB). Alternatives: elongated
box (needs cpp_box Px Py Pz), or larger xi / smaller everything at fixed r0 / xi.

## Branch split (next 10-07 item 1; quasi_energy.py --branch, branch_N8_r80_P480.log)
psi_- = P_- psi per k, P_- = (1 - H2 / Om) / 2; --test --branch: uniform rest state branch- norm 1e-28 (sign right), g 0 conserved exactly.
  t        0     1k    2k    4k    8k    16k
  total    743.4 743.1 743.1 743.1 743.2 743.1   (E_- -692.5 -> -694.6; conserved to 4e-4 although g != 0)
  strands  364.8 13.9  5.3   5.2   6.7   10.9
  interior 32.1  0.5   0.2   0.3   0.4   0.7
  near     111.2 53.4  17.0  11.3  14.3  22.1
  outer    126.5 359.8 129.4 66.3  69.2  130.7
  axis     18.6  26.5  38.6  26.6  18.5  25.2
  floor    90.1  289.0 552.7 633.5 634.1 553.4
Reading: YES, the shed is negative-branch. The seed carries 743 of branch- norm, ~2/3 inside the structure; by 1-2k the coil
expels ~97 % of it (strands 365 -> 5) and it sits in the floor; E_- / N_- = -0.93 ~ -om0 (arcsin 0.8 = 0.927) -> near-rest
negative-branch content. N_- is (nearly) a separate invariant, so the 'shed' is the seed's wrong-branch admixture being
sorted out of the coil, not loss: the relaxed coil is (nearly) pure + branch. The floor's E drop (~-670) is mostly this
(-0.93 x +540 moved ~ -500). Prediction: a + projected seed sheds ~nothing -> branch/ (spring_box.py --project_plus).
