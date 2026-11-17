# 2026-10-19 glider momentum diagnostic check (next 10-18 item 4)

mom() = D x B (d / S1 = D via linear hodge, b = B): the conserved (Minkowski) momentum of any local H(D, B) in the
continuum -- formula right. Not the energy flux (S = E x H = f'E f'B D x B): a dark spot (f' ~ 0.01) can hold D x B and
sit still. Open: is the loss lattice (umklapp from grid-scale content) or reconstruction error?

Smooth-ish regime, pack beta 1 P 100 sig 3 (gl.py 10-18, T 100):
| amp | Pz tot t 0 / 10 / 20 / 30 / 100 | Ewin/H end | note |
| 3 | 804 / 596 / 417 / 379 / 379 | 0.010 | packet disperses; Pz -53 % during the nonlinear phase (t < 30), then exact |
| 10 | 2681 / 1742 / 985 / 923 / 93-625 (swinging) | 0.33 | lump; late Pz oscillates +-300 = reconstruction noise |
Loss happens only while nonlinear (constant once the field is linear again). sig 3 = wavelength 9.4 cells: the
nonlinear harmonics (3k, 5k) are at / beyond the zone edge immediately -> umklapp is expected at this resolution.
Decisive (queued, 4-core cap): a 3 under refinement (P 140 sig 4.2 T 140, P 200 sig 6 T 200). Loss fraction shrinks ->
lattice harmonics (discretisation); constant -> diagnostic or scheme issue.

## Refinement result (a 3, sig scaled with P): Pz / Pz(0)
| t / sig | P 100 | P 140 | P 200 |
| 10 | 0.47 | 0.50 | 0.53 |
| end (33) | 0.47 | 0.44 | 0.36 |
Loss does NOT shrink with refinement, and at P 140 / 200 the a 3 packet no longer disperses (Ewin/H end 0.01 -> 0.09 ->
0.16): the scale-free rule collapses toward the grid at every resolution, so a grid-scale core always forms and takes
momentum. Closed as moot (Philip 10-19, no isolated cell: anything that needs grid-scale contrast is an artefact).
