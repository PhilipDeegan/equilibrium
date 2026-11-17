# Hex:square Hodge star ratio -- is 9:1 / 3:1 arbitrary? (2026-09-28)

`star_ratio.py` (log `star_ratio.log`), symbols in `equilibrium/substrate/bcc_symbol.py`, test
`tst/substrate/test_hodge_star.py` (4/4). r = *1_hex / *1_sq, DEC on the BCC Delaunay / Kelvin complex,
u sector (0-forms, step()) and T sector (1-forms, step_ae curl curl), spatial symbol only.

| r | meaning | u dc/c spread | T dc/c spread (= birefringence) | c_T^2/c_u^2 with *2 = 1 |
|---|---|---|---|---|
| 2 | doc/dst.md w = 2/3 : 1/3, "maximal isotropy" | 0 (1e-7) | 0 (1e-7) | 16/9 |
| 3 | Kelvin geometry: |dual face| / |edge| exactly | 0.0046 | 0.0035 | 1 |
| 9 | step() now (W = 3:1 on both legs = W^2) | 0.0130 | 0.0097 | 0.16 |

(dc/c in units of (kh)^2, h = hex distance, 404 directions; means -0.026..-0.029 u, -0.009..-0.015 T.)

1. Not arbitrary in the continuum: every r is consistent (c direction independent at k -> 0, both
   sectors): c_u^2 = (r+1)/4, c_T^2 = 4/(r+1). The ratio only sets the grid-scale error.
2. r = 2 makes the 2nd-order speed error exactly isotropic in BOTH sectors, no birefringence; the four
   gapped 1-form modes become degenerate (omega^2 = 48 at *2 = 1). The u result is analytic: the 4th
   moment of the 8 hex links plus the 6 square links is rotation invariant iff c_hex = 2 c_sq.
3. r = 3 is the only ratio where the geometric *2 = 1 gives c_T = c_u with no extra factor (the Kelvin
   complex is self-consistent DEC). Any other r needs *2 = (r+1)^2/16 (r = 9: 6.25; the C++ 'dec' 5/2 is
   in its own (W/h)^2 normalisation).
4. r = 9 is justified by neither and is the most anisotropic of those tested. It matches no documented
   derivation: doc/dst.md says 2:1 ("uniquely determined by maximal isotropy"), the code has W = 3:1 on
   both legs of step() (du = -sigma sum W_k phi_k, dphi_k = -sigma W_k (u_j - u_i)).

In step()'s symmetric form the per-leg W is sqrt(r): r = 2 -> W_HEX = sqrt2, r = 3 -> W_HEX = sqrt3
(equivalently W on one leg only). PROPOSAL (revised after Philip: the continuous model is only an approximation of the discrete
fundamental lattice, and its coarse-cell geometry is also an approximation): step() (fundamental) ->
3:1, the lattice's own geometry; continuous model -> 2:1, since its (kh)^2 error is discretisation error
and its target is the lattice's isotropic long-wave limit (the lattice's own anisotropy ~ (k h_0)^2 is
negligible there). Philip: coarse Kelvin cells are used mostly to avoid a preferred axis -> isotropy
is the coarse criterion -> 2:1 for the continuous model. step()'s rule still open.
