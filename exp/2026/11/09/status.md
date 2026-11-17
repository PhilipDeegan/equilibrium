# 2026-11-09

## Resume 1b: where does the coarse helicity go? (no new runs; helicity/fine_hel.py on the 11-08 dumps)
fine_hel.py: LS helicity sum A.curl A (A = 3-DOF least-squares vector per cell) of raw phi, of the curl-curl smoothed
phi_s (tau 1 at a 9, 4.938 at a 20, as coarse.py), of phi_f = phi - phi_s, cross term, + / - parts of the fine
density; signed FFT helicity spectrum of A in bands of n = |k| side / 2 pi (alias-safe up to 0.35 P); non-vector share.
Logs: helicity/a9.txt (P 96, all dumps), helicity/a20.txt (P 192, every 2nd). All H / raw H at t 0.

  a 9:   t      Hraw   Hs     Hf    Hf+    Hf-    nonvec  E share n 16-33
         0      1.000  0.608  0.061 0.061  0.000  0.000   0.000
         150    0.620  0.366  0.050 0.090 -0.040  0.129   0.089
         200    0.085  0.062 -0.002 0.084 -0.086  0.282   0.476
         240    0.014  0.009  0.002 0.092 -0.090  0.294   0.729
  a 20:  0      1.000  0.605  0.062 0.062  0.000  0.000   0.000
         310    0.487  0.298  0.056 0.160 -0.105  0.166   0.151
         450    0.024  0.006  0.009 0.202 -0.193  0.285   0.733

Reading:
- The vector-part helicity is NOT conserved in total: raw H falls with the coarse H (0.014 / 0.024 at the end).
  No band of the spectrum gains net helicity at any time; the energy goes to the finest band (73 % at the end).
- The fine vector content is helicity-NEUTRAL: + and - parts grow equal (0.09 / -0.09 at a 9; 0.20 / -0.19 at a 20,
  more mixed sign at better resolution). Content moves down without its handedness = not a helicity cascade.
- Caveat: the LS measure sees only 3 of 7 face DOF per cell; the other 4 (non-vector, 28 % of content at the end)
  carry no helicity definition here. The missing helicity could sit there.
Next (not started, needs Philip's go): a helicity on the complex itself, a cup da summed over Delaunay tets
(sees all edge DOF): (1) is it conserved by Cartan; (2) does the non-vector part hold the lost helicity.

Box sizes used (11-08 runs): a 9 in P 96 (side 55.4, N 221k), a 20 in P 192 (side 110.9, N 1.77M) -> box ~3 body
diameters. Philip: too small may be invalid, go larger if needed -> a larger-box check belongs with any conclusion here.

## Box-size check: a 9 in P 192 (box ~6 diameters vs ~3 at P 96), same seed recipe, step() + Cartan, T 250
cmd: PYTHONPATH=. taskset -c 0-3 python3 -u exp/2026/11/08/cascade/cascade.py --P 192 --a 9 --vmax 0.3 --T 250
  --every 2.5 --smooth 1 --threads 4 --seed /bk/dst/2026/11/09/bigbox/seed_a9_P192.npz --dump 10
  --dumpdir /bk/dst/2026/11/09/bigbox/a9_P192 --save /bk/dst/2026/11/09/bigbox/end_a9_P192.npz  (log bigbox/a9_P192.txt)
Started ~12:55, 0.47 s/tick, ETA ~107 min. Seed matches P 96 (cos 0.9651, lam a 3.943, Cartan |T|/|phi| 8.749e-3;
nonvec at t 5 1.258e-4 = P 96).
Result (bigbox/a9_P192.txt, coarse_a9_P192.txt, hel_a9_P192.txt): the box does NOT matter here.
- Scorecard identical to P 96 to 3-4 digits through t 160 (H/H0, nonvec, shape, centroid, in/shell/out); later
  turbulent divergence only (H/H0 t 200: 0.098 vs 0.085); end state the same (dE/E -6.5e-6, in 0.746 vs 0.744).
  Only systematic difference: Eu/E (wave share) ~20-40 % higher at P 192 mid-run (waves not re-entering), still ~1e-3.
- Coarse body (tau 1): E_s/E_s0 0.944 / 0.624 / 0.139 / 0.043 at t 100 / 150 / 200 / 240 vs P 96 0.945 / 0.624 /
  0.137 / 0.038; overlap 0.983 at t 150 both; H_s faster than E_s as before.
- Fine helicity: same picture (Hraw 0.009 at t 240; fine + / - 0.089 / -0.093; nonvec 0.296).
-> the 11-08 cascade / ideal decay / helicity-loss results are not periodic-image artefacts at box ~3 diameters.

## Whitney (all-DOF) helicity: the hidden DOF do NOT hold the lost helicity; Cartan destroys it
helicity/whitney.py: H_W = sum over BCC Delaunay tets of int W(a) . W(da), a = phi / W (edge integrals; all 7 DOF
per cell), 6 congruent tets per cell (vol 0.1283 each). validate.py: tets fill the box exactly; uniform curl
reproduced exactly; gauge-exact (random f: 1e-15); symmetric H(a, b) = H(b, a); seed a 9 H_W = 1.03 x LS helicity
(LS sum x cell volume 0.77).
whitney_series.py on a 9 P 96 dumps (helicity/whitney_a9.txt), H / H_W(0):
  t      H_W    Hvec   Hrest  Hcross  Hs     Hf     Hf+    Hf-    rate_Cartan  rate_fd
  0      1.000  0.970  0.000  0.030   0.606  0.062  0.062  0.000  0
  100    0.956  0.928  0.003  0.024   0.574  0.058  0.079 -0.020  -0.0025     -0.0021
  150    0.641  0.602  0.008  0.031   0.365  0.066  0.274 -0.208  -0.014      -0.015
  200    0.112  0.081  0.030  0.001   0.062  0.019  0.575 -0.556  -0.14       -0.039 (late: H small, noisy)
  240    0.014  0.012 -0.002  0.004   0.009 -0.001  0.648 -0.649
- Full-DOF helicity falls like the LS one (0.112 vs 0.085 at t 200). The hidden part (rest + cross) never holds more
  than ~0.04: it is NOT where the helicity went.
- Instantaneous rate from cartan_transport matches the dump-to-dump decline (t <= 180): Cartan itself destroys H_W,
  from t 0 (-6e-5 per time unit at t 10, when the field is still smooth), accelerating as fine content grows.
- Fine content: large + and - helicity densities (+0.65 / -0.65 of H_0 at the end) cancelling: cell-scale content is
  locally strongly handed both ways, net zero.
Reading: continuum Euler conserves helicity because da ^ i_v da = 0 (no 4-forms in 3D); the per-triangle Cartan
discretisation has no such identity -> helicity is not an invariant of the rule. Energy is (by construction), helicity
is not. Caveat: at cell scale helicity is definition-dependent; Whitney is the canonical one and agrees with LS on
smooth content; an exotic definition the rule conserves is not excluded, but it would have to disagree with every
smooth-limit measure about where the coarse helicity went.

## Is Cartan legal under DST? No (Philip agrees)
Passes only the vector-free test. Conflicts: it IS incompressible Euler (kappa 1 picked as Euler's coefficient; substrate
is not a fluid; reproduce, don't reify); phi used as its own velocity (step() has phi ~ A, linear); separately motivated
second relation (added 10-13 because step() freezes div-free phi); quadratic -> rates from amplitude, not cell size;
runs to cell scale (UV); not even a faithful Euler (helicity, above); BCC-Delaunay specific.
Philip: "there's no such thing as an incompressible fluid, it is itself an approximation; really it's about asymptotes,
how much you can compress something before it pushes back more than you're pushing in."
-> step() + Cartan results (10-13 .. 11-08: ring travel, 11-03 thermalisation / pulse anisotropy, spheromak cascade,
ideal decay, today's helicity loss) are about Cartan, not DST. Open again: what moves circulation in the rule. Lead:
a compressible substrate response whose push-back stiffens asymptotically as compression nears a limit.

## Pinch test: step() + compression-dependent face rate s(u) + one-sided asymptote e(u), NO Cartan (pinch/pinch.py)
H = sum e(u) + 1/4 sum s_face phi^2, e = um^2 (-x - ln(1-x)) (x = u/um; infinite push-back at u -> um, soft for u < 0),
s = exp(-beta x); u_t = -sum W s phi, phi_t = -W grad(e'(u) + 1/4 s'(u) sum phi^2). Energy-exact (RK4 drift 1e-6..4e-5 at
T 60). phi changes by gradients only -> loop sums frozen: circulation cannot move here, only compression responds.
a 6 spheromak, P 96 (box 4.6 diameters), u 0, T 60. Plain step() (beta 0): u stays exactly 0.
  beta +0.5 um 0.3: core COMPRESSES, u 0.136 (x 0.45), steady t 30-60; profile smooth, 90 -> 10 % over r 1.75 -> 5.75
                    (the body's own circulation profile), u = 0 beyond a; Eu/E 0.07; ~3 % of E radiated as waves.
  beta +1.0 um 0.3: same shape, u 0.172 (x 0.57), Eu/E 0.17.
  beta -0.5 um 0.3: core EXPANDS, u -0.62 (x -2.1) and noisy: the soft (expansion) side has bounded push-back
                    (e' -> -um) -> no limit; um 0.15 runs away (nan by t 14).
  um 0.15 / 0.08 with beta > 0: blow up by t 2: the asymptote is stiff (local wave speed^2 ~ e'' = 1/(1-x)^2 -> inf
                    near the limit) -> explicit dt too large. Numerical, not physics: needs dt ~ (1 - x_max) or implicit.
Reading: with beta > 0 circulation pinches compression into its own volume (static pinch equilibrium, smooth, no edge of
its own); the asymptote was not reached (x <= 0.57), so the 'surface' question is untested. Sign matters: compression
side bounded (holds), expansion side unbounded here (runs away) -- that asymmetry is from the chosen e(u), not derived.
Next: adaptive dt to drive the core onto the asymptote (does a saturated plateau + sharper edge form?); then an ambient.
Adaptive dt (dt0 x min(1, 4 (1 - x_max))), beta +0.5:
  um 0.15: core held at x 0.77 (x_max 0.80), steady t 20-60, dE/E -2.3e-3 (RK4 + adaptive dt, grows slowly);
           profile u 0.115 / 0.111 / 0.096 / 0.077 / 0.055 / 0.032 / 0.009 at r 0.2..6.2: FLATTER top and a slightly
           WIDER edge (90 -> 10 % over 4.5 vs 4.0 at um 0.3). The asymptote caps the core and spreads it, no sharp surface.
  um 0.08: nan before t 2 (stage overshoot past um; loop then hung on nan t -> killed, guard added). Needs implicit.
Reading: as expected for a convex response, a stiffening limit alone flattens and widens, it does not make an edge.
Surface tension needs a non-convex part (a range where more compression is favoured) or an outer push (ambient).
