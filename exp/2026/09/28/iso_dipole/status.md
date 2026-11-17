# Dipole benchmark under the continuous model's iso weights (2026-09-28)

`dipole.py` (copy of 09-27 dec/dipole_dec.py + `--weights`), `run.sh`; C++ step_ae, curl 'dec', R = 20,
lam = 4, reference F^2 = 0.4114. iso: W = sqrt2:1 (star 2:1, s = 3/4), c = 2.

| h | weights | source work (err) | shell flux r=5/6.5/8/9.5 | slope |
|---|---|---|---|---|
| 0.5 | iso | 0.4216 (+0.0102) | 0.387 / 0.386 / 0.388 / 0.384 | -1.984 |
| 0.25 | iso | 0.4145 (+0.0031) | 0.406 / 0.408 / 0.407 / 0.406 | -1.994 |
| 0.5 | discrete (09-27) | 0.425 (+0.0136) | 0.388 / 0.387 / 0.389 / 0.386 | -1.976 |
| 0.25 | discrete (09-27) | 0.415 (+0.0036) | 0.406 / 0.409 / 0.407 / 0.406 | -1.993 |

Source-work error 25% / 14% lower than the discrete-weight 'dec'; ratio 3.3 (2nd order). Shell flux
unchanged (its diagnostic, the LS reconstruct_A_at / curl_A_at, dominates that error). Along the z
(cubic) axis the anisotropy gain is not expected to show much; an off-axis ([111]) run would.
Cost ~ the same (1.5 / 23 min at 8 threads): c = 2 lengthens the window, dt is 3.5x larger.

Bug found on the way: the source work used a hardcoded 3/80 = 1/(2 C0SQ) at C0SQ = 40/3; fixed to
1/(2 C0SQ) (first run's 0.126 / 0.124 in bad_norm/). The same constant is in the 09-25..27 dipole
scripts, which are now pinned to set_weights("legacy") (as are all 20 pre-09-28 scripts using
continuous), so they reproduce as recorded.
