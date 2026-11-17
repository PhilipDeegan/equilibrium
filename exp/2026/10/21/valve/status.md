# 2026-10-21 valve: can held bodies exchange circulation by reconnection? R 20 timing run

Prior (structural): step() + Cartan (+ store) is Lie-Poisson: circulation is advected as a 2-form by b = dH/da, so flux
through every material surface is frozen (Kelvin) for any H -> no reconnection in the continuum limit; anything seen is
the lattice. R 9 too coarse (10-18 zoo: resolution-dominated; contact would be grid-scale from the start).
Cost: speed ~ 1 / R in lattice units, dt fixed by the lattice -> an event costs ~ R^5 (R 40 ~ 32x R 20).

## Run (r20.sh; r20_tilt.txt; /bk/dst/2026/10/21/valve/{seed,end}_tilt_R20.npz; slice.py -> slice_*.png)
zoo.py --struct side_tilt (new: two rings, normals tilted 45 deg toward each other, near rims 0.25 R apart; --ang, --gap;
diagnostics: |G| share near the midplane, per-half x centroid, connected components of |G| > 0.3 / 0.5 max).
R 20, P 192, linear store, 8 threads, T 120: 0.70 s / tick, 6600 ticks, 61 min wall. dE/E -2.6e-8.

| t | midplane share |x|<R/4 | half x centroids | sum|G| | dz | comps > 0.5 max |
| 0   | 12.6 % | -17.78 / +17.67 | 1633 | 0     | 2 rings |
| 30  | 16.8 % | -16.52 / +16.41 | 1682 | +1.10 | 2 (halos touch at 0.3) |
| 60  | 21.8 % | -15.23 / +15.11 | 1804 | +1.68 | 2, contact strips flatten |
| 76  | --     | --              | --   | --    | contact = 1-2 cell hot spot, peak 2.7x |
| 90  | 27.4 % | -13.89 / +13.75 | 2034 | +1.61 | hot spot dominates max |
| 120 | 36.2 % | -12.19 / +11.79 | 2528 | -0.15 | peak 16x initial, grid tangle |

Reading:
1. Approach is steady (~0.043 / t per ring) and the near rims flatten into an anti-parallel sheet by t ~ 55 (classic
   pre-reconnection), cores still distinct.
2. By t ~ 75 the contact collapses to a 1-2 cell spot; from then sum|G| grows and accelerates (+55 % by T 120, still
   rising), peak |G| x16, forward motion stalls and reverses. That is inviscid-Euler-type core thinning reaching the cell:
   a lattice-set tangle, not a clean 2 -> 1 -> 2 reconnection. No whole-unit transfer is readable.
3. Expected under refinement: the smooth phase (approach, flattening) converges; the onset of the grid-scale phase moves
   later (thinner sheet resolvable) at ~R^5 cost (R 28 ~ 5x, R 40 ~ 32x). No refinement-fixed threshold by construction.
-> the valve question needs a mechanism in the rule (what breaks frozen-in smoothly), not more resolution. Philip's call.
