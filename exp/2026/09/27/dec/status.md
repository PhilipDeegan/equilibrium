# DEC curl curl on the real lattice (numpy prototype; C++ untouched)

`dec.py`: DecCurl (sparse d1 over the 12 Delaunay triangles/cell, Kelvin stars) and DecAE (continuous.AE
with curl_curl_sym -> DecCurl). `checks.py` (log checks.log), `packet.py` (packet.log),
`dipole_dec.py` (copy of 09-26 dipole_radiation.py, --op dec|ls, numpy path).
Memory note: SubstrateLattice(radius_cells=R) takes the PHYSICAL radius (a first run passed R/h and
hit 52 GB). DecCurl ~2 KB/cell.

## Operator checks (interior)
- curl curl of a discrete gradient: DEC 2-6e-15 (exact) vs LS 5e-2 smooth / 0.3-1.2 random.
- step()'s divergence of the output, sum_k W_k cc_k: DEC 1-6e-15 (exact: W^2/*1 x (W/h) makes it
  d0^T d1^T = 0) vs LS 0.11 / 0.017 / 0.0022 (h = 0.5 / 0.25 / 0.125) -- LS is NOT discretely div-free,
  contrary to its docstring (it is Euclidean-symmetric, but its output has O(h^2) step()-divergence).
- symmetric in the *1/W^2 face metric (4.8e-15).
- pointwise truncation on a smooth sampled A: DEC 0.30 at every h (does NOT converge) vs LS 0.29 /
  0.086 / 0.022. Cause (symbol): a smooth field has O((kh)^2) amplitude in the 4 gapped modes, and
  their eigenvalue is O(1/(kh)^2), so the output error is O(1). Exact edge-integral sampling doesn't
  fix it (0.25). It does not show up in the evolved fields (below): that component just oscillates
  at omega h/c ~ 5 with small amplitude.

## Transverse packet (lam=4, travel 3; A from phi, E; rel. residual in the measuring cylinder)

| h | n | LS A res | DEC A res | LS E res | DEC E res | LS dc/c | DEC dc/c |
|---|---|---|---|---|---|---|---|
| 0.5 | [100] | 0.92 | 0.051 | 1.00 | 0.079 | -0.093 | -0.0054 |
| 0.5 | [111] | 0.55 | 0.069 | 0.79 | 0.103 | -0.068 | -0.0075 |
| 0.25 | [100] | 0.25 | 0.013 | 0.36 | 0.020 | -0.027 | -0.0013 |
| 0.25 | [111] | 0.17 | 0.017 | 0.25 | 0.037 | -0.018 | -0.0016 |

DEC ~20x smaller residual/phase error, converging ~4x per halving (E along [111] 2.8x). numpy DEC is
7-10x faster than numpy LS.

## Dipole benchmark (R=20, lam=4, numpy DEC), reference F^2 = 0.4114

| h | source work | shell flux r=5/6.5/8/9.5 | slope | pattern resid | azimuth spread |
|---|---|---|---|---|---|
| 0.5 | 0.422 (err +0.011) | 0.386 / 0.385 / 0.388 / 0.384 | -1.981 | 0.016-0.020 | 0.1-0.8% |

LS (C++) for comparison: source work err 0.187 / 0.033 / 0.0076 at h = 0.5 / 0.25 / 0.125, so DEC at
h=0.5 is roughly LS at h~0.15. Flux is 6% LOW, but the flux diagnostic (reconstruct_A_at +
curl_A_at, i.e. the LS gradient with its -0.14 (kh)^2 amplitude error) is itself under test here;
source work (J.E, no derivative) is the cleaner number.
h=0.25 (numpy, geometric star): source work 0.414 (err +0.0026; ratio 4.2 from h=0.5 -> 2nd order,
already below LS at h=0.125, 0.0076); flux 0.406 / 0.408 / 0.407 / 0.406 (err -0.005, ratio ~5);
slope -1.995; pattern resid 0.015-0.018 (= the 20-degree bin average of sin^2, 0.017).

## Correction: which divergence is exact

The geometric star (3:1) makes the UNWEIGHTED sum_k out_k exactly zero, not step()'s sum_k W_k out_k
(an earlier check used an axisymmetric field, where symmetry hid it; random field: 4.2). step()'s u
sector is itself DEC with *1 ~ W^2 = 9:1 (its u dispersion equals the 9:1 DEC 0-form Laplacian
exactly). With *1 = 9:1 the transverse operator is still isotropic and consistent (c^2 0.4 in DEC units),
and a single *2 factor 5/2 makes transverse c = step()'s c. That "dec" star: sum_k W_k out_k = 0 exactly
(no transverse -> u coupling), curl(grad) = 0, one Hodge star for both sectors (the "one
reconstruction" option, by construction). Transverse dispersion -0.011 [100] / -0.017 [111] /
-0.020, -0.011 [110] (kh)^2 (geometric: -0.009 / -0.011 / -0.012, -0.009); gapped modes at
omega h/c 4.5-8.2.

## Ported to C++ level 0 (continuous::AdaptiveLattice)

set_curl('ls' | 'dec' | 'dec_geo') for step_ae; curl_curl_dec(i, phi, star). numpy reference
continuous.curl_curl_dec(lat, phi, star), AE(curl=...). Tests TestDecCurl (C++ == numpy to 1e-13;
step_ae == AE to 1e-11; exact identities for both stars; packet 2nd order, >10x better than LS): 19/19
pass. Cost: 167 ms/step vs LS 126 at N=2.8M, 8 threads (DEC at h=0.25 beats LS at h=0.125: ~12x
cheaper per accuracy). C++ 'dec' dipole sweep h = 0.5 / 0.25 / 0.125 running (run_dec_cpp.sh).


## C++ 'dec' (step()'s 9:1 star) dipole, R=20, lam=4

| h | source work (err vs 0.4114) | shell flux r=5/6.5/8/9.5 | slope | pattern resid |
|---|---|---|---|---|
| 0.5 (R40) | 0.425 (+0.0136) | 0.388 / 0.387 / 0.389 / 0.386 | -1.976 | 0.020-0.023 |
| 0.25 (R80) | 0.415 (+0.0036) | 0.406 / 0.409 / 0.407 / 0.406 | -1.993 | 0.015-0.018 |
| 0.125 (R160) | running (started 18:59, ~5 h) | | | |

Source-work ratio 3.8 (2nd order); ~25% larger error than the geometric star, as the symbol predicts,
and ~9x below LS at every h. Default switched to 'dec' (Philip): C++ Curl::dec, AE(curl="dec"); LS
tests explicit; rebuild pending until R160 finishes.
