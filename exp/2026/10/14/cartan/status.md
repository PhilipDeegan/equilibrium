# 2026-10-14 cartan

vill.py = 10-13 copy + --seed (cache post-smoothing phi), --ckpt (resume; verified bit-identical to an uninterrupted
run at P 72), --rcut (energy fraction beyond r from the core line), Eu (u-energy fraction = wave part; seed is pure
div-free phi).

## 00 R 40 late leak
DONE 17:28 (14:30 start). Log vill_P278_R40_smooth19.75_off_cartan_k1_T145_cpp_10-14.log. Travel at t 140 = 0.135 R
(T 145 reaches ~0.14 R, not 0.15). E drift -1.1e-9 (R 20 -1.6e-8, R 9 -4e-7).
Growth vs t 0 at matched travel (R 9 / 20 / 40), interpolated:
  0.06 R: rms +1.76 / +0.29 / +0.21 %   sum|G| +0.08 / -0.08 / +0.01 %
  0.10 R: rms +4.87 / +0.99 / +0.54 %   sum|G| +1.97 / +0.16 / +0.06 %
  0.12 R: rms +7.63 / +1.72 / +0.82 %   sum|G| +3.78 / +0.44 / +0.11 %
R 20 -> 40 at 0.12 R: rms x 1/2.1 (first order), sum|G| x 1/4.0 (second order). The late leak is a discretisation
residual and converges away; the onset isn't delayed, it's weaker. Slow pre-onset rms growth (~+0.2 % per 0.04 R) is
about the same at R 20 and R 40: possibly a resolution-independent slow core widening, small; watch, not acted on.
Seed /bk/dst/2026/10/14/cartan/seed_P278_R40_smooth19.753.npz, ckpt ckpt_P278_R40_T145.npz (same dir).
Resume: rerun the same command (seed + ckpt are picked up).

## 1 valve test, R 9 (P 72, smooth 1, T 20), off-quantum circulation --circ 1 + eps, eps = +-0.003 .. 0.1 (valve/)
- linked circulation stays at 1 + eps to 1e-5 for every eps: no shedding toward the quantum (Kelvin exact).
- response is linear in eps from 0.003 to 0.1, no threshold: d rms / eps = 1.33-1.40, d Eu / eps = 0.0139-0.0146
  (end-time difference vs eps 0). E drift -4e-7 for all.
- Eout(4/6) differences are jitter (sign flips, no eps scaling): the ring's far flow holds ~55 % / ~30 % of E
  beyond 4 / 6 cells at eps 0, so Eout is not a shedding metric here.
- Verdict: no valve in step() + Cartan, as expected. Any eps is held; there is no preferred circulation.
R 20 not run (same exact conservation; would only confirm linearity).
