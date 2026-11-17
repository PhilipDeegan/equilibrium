# 2026-10-18 glider -- search, don't impose (Wolfram / CA gliders): fixed rule, many source-free seeds, keep what persists

gl.py: hop / triangle DEC, H = sum_cells f(uE) + f(uB), f(u) = beta^2 (sqrt(1 + 2u / beta^2) - 1) (saturating: throughput
f'(u) falls with the cell's own energy; fields capped). uE / uB = cell shares (hop 1/2 per end, triangle 1/3 per
vertex). Separable -> Stormer-Verlet (symplectic, dH/H <= 7e-3 incl. wrap transients). Seeds div-free exactly:
b = curl alpha, d = dual curl C. Families: pack (linear-polarized travelling packet, k = 2 / sig), circ (circular),
mag (pure B lump, starts magnetic -> switching), ring (poloidal B ring), hopf (ring + axial, A.B != 0), rand (random).
Scores: Ewin/H = energy within 2 sig of the hottest cell; |v| = late window-centroid speed; reuse = energy overlap
with the same cells 2 sig / c earlier (c-mover ~0.2-0.3, stationary ~1; Wolfram: reuse = rest mass).
Linear check: pack travels +z at 1.00, diffracts (Ewin 0.90 -> 0.18 by t 30).

## Batch 1 (P 100, sig 3, T 100, beta 1; b1/)
| fam | linear | a 0.3 | a 3 | a 30 | a 300 | (Ewin/H end) |
| pack | 0.039 | 0.036 | 0.010 | 0.445 | 0.795 |
| circ | 0.037 | 0.034 | 0.017 | 0.527 | 0.829 |
| mag | 0.029 | 0.038 | 0.368 | 0.698 | 0.863 |
| ring | 0.024 | 0.025 | 0.031 | 0.376 | 0.733 |
| hopf | 0.024 | 0.025 | 0.030 | 0.374 | 0.706 |
| rand | 0.031 | 0.031 | 0.072 | 0.565 | 0.813 |
Every persistent case is stationary (|v| <= 0.014, reuse 0.86-0.95) with umax growing 3-35x: collapse to a hot spot.
No travelling persistent structure in batch 1.

Hot-spot anatomy (P 100 saved finals, /bk/dst/2026/10/18/glider/): half the energy within r ~2-4 cells; fpE, fpB
0.006-0.04 (local throughput ~2-3 % of c: trapped by its own slowdown, a dark spot); linear Rayleigh omega^2 8-16
(zone-edge photon branch, not the gapped 27 modes); E / B split 0.14-0.68 (oscillating, switching).
Expectation: at large u, f ~ beta sqrt(2u) is degree-1 homogeneous in the field -> no length scale -> collapse runs
to the grid. Refinement test (P 140 sig 4.2, P 200 sig 6, refine/) decides: grid-scale spot = lattice artefact.

## Refinement (refine/, P 140 sig 4.2 T 140, P 200 sig 6 T 200 = same physical run; rule is invariant under x, t -> lam x, lam t)
| case | Ewin/H end P 100 / 140 / 200 | half-energy radius / sig (P 200) |
| mag a30 | 0.698 / 0.684 / 0.686 | ~0.9 |
| pack a300 | 0.795 / 0.736 / 0.716 | > 1 |
The bulk converges (same curve in sig / c units): NOT a grid artefact. Only the central peak keeps sharpening toward the
grid (umax 7 -> 200, r < 0.5 holds 0.3-1.5 %). Bulk leaks slowly (mag 0.79 at 20 sig / c -> 0.69 at 33). Held or slow?
Throughput inside is 1-8 % of c, so the escape time ~ sig / (c sqrt(fpE fpB)) ~ 30-100 sig / c: long run decides (long/).

## Long runs (long/, P 100 sig 3, T 1000 = 330 sig / c)
| case | Ewin/H at t 100 / 500 / 1000 | leak per 100 t (late) | umax end / 0 |
| mag a300 | 0.863 / 0.811 / 0.762 | ~0.010 | 28 (plateau ~650-800 from t 300) |
| mag a30 | 0.698 / 0.590 / 0.524 | ~0.015 | 33 (plateau ~250-300) |
| pack a300 | 0.795 / 0.581 / 0.495 | ~0.015-0.03 | 9 |
| rand a30 | 0.565 / 0.405 / 0.300 | ~0.02-0.03 | 11 |
Leaks steadily, no plateau in 330 sig / c: held by its own slowdown, not bound. Leak slower at higher amplitude (lower
throughput inside). Central peak saturates (no runaway to the grid). Stationary (reuse 0.9): "massive" by Wolfram's
reuse reading; none travels. Known analogue: Wheeler's geon (1955) -- field held by its own gravity-like slowdown,
long-lived but leaking; here the slowdown is the cell's own throughput (Philip's throughput-from-cell-condition).
Next: leak rate vs amplitude (amp/, a 100-10000, T 600): does the leak vanish above some amplitude (valve) or only slow?

## Amplitude scan (amp/, mag, T 600): leak per 100 t (t 100 -> 600)
a 30 ~0.015 | a 100 0.020 | a 300 0.012 | a 1000 0.008 | a 3000 ~0.003 | a 10000 ~0.004 (noisy, umax still rising).
Leak falls smoothly with amplitude; no sharp threshold (no valve) seen. Rule has an energy scale (beta^2) but no
length scale: held lumps exist at every size (x, t scaling), nothing selects a size.

## Momentum (mom/; D x B from cell-reconstructed E, B -- diagnostic only)
Linear pack: Pz total 268.1 -> 267.7 over T 200 (conserved, diagnostic OK).
Saturating pack, P 100: a 300 Pz 80430 -> 14840 (-82 %) while Ewin/H stays 0.71; a 30 8040 -> ~900.
The packet stops because its momentum goes INTO THE LATTICE (grid-scale central peak: umklapp / Peierls-Nabarro
pinning), not because the lump has inertia. The "stationary = massive (reuse)" reading of the lumps is void until this
is resolved. Refinement (P 140 / 200, sig scaled) running.

## Integer mode (--q; int/): d, b integer counts x q, kicks rounded BEFORE the curl:
b -= q Cm rint(dt/2 gE / q); d += q CT rint(dt gB / q). Cm is +-1 -> integers stay integer, div d = div b = 0 exactly
(charge exact). Each half-kick depends only on the other field -> exactly reversible: forward t 20 + back = bit-identical.
Smooth limit (linear pack, amp 1, P 100, T 60):
| q | H/H0 - 1 at t 10 / 30 / 60 | Ewin/H t 60 | nonzero elements outside window t 60 |
| float | 3e-4 / 3e-4 / 3e-4 | 0.049 | -- |
| 2^-12 | 0.004 / 0.048 / 0.17 | 0.042 | 4.61M of 4.75M |
| 2^-8 | 0.53 / 8.2 / 37.5 | 0.007 | 4.56M |
| 2^-6 | 5.4 / 82 / 476 | 0.007 | |
| 2^-4 | 39 / 562 / 4010 | 0.009 | |
| 2^-2 | 106 / 1580 / 14200 | 0.018 | |
Momentum Pz stays ~268-275 (conserved by the translation structure), but energy grows ~t^2 at every q, ~q^2:
rounding pumps grid-scale content over the whole box (97 % of elements nonzero). Same failure as 10-16 zclock, worse.
Verdict: rounding the amplitude kicks is a noise source, not a valve. Reversible + integer is not enough: an integer
rule needs an exactly conserved quantity, i.e. counts that are MOVED (what leaves one element enters another), not
amplitudes that are rounded. For a linear wave the conserved form is quadratic; integer maps that conserve a quadratic
form exactly are permutations / sign flips (lattice gas / walk lineage). Open: what the integer should count.

## Momentum under refinement (mom/, pack a300, sig scaled with P): Pz / Pz(0) at matched t / sig
| t / sig | P 100 | P 140 | P 200 |
| 10 | 0.52 | 0.55 | 0.59 |
| 20 | 0.43 | 0.39 | 0.36 |
| 33 | 0.29 | 0.28 | 0.17 |
Momentum loss does NOT go away with refinement. Either the continuum version of this rule doesn't conserve D x B (it
should for a local translation-invariant H(D, B) -- suspect), or the central peak collapses to the grid at every
resolution and keeps exchanging momentum with the lattice (umklapp) whatever P is. Either way the lump does not keep
its momentum: no inertia, it stops relative to the lattice. Fails Newton's first law as a held structure.

## Verdict (Philip 10-18): a lump isn't a structure -- nothing gives it a reason to remain bounded.
The saturating lumps were held only by their own slowdown (dark spots): scale-free, leaking, no momentum. Not a
candidate. Structures need an organising reason to stay bounded (circulation / winding / topology = blocked
dissipation). Glider search over unstructured seeds closed.
