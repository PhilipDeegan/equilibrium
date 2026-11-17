# 10-08 vs: the walk against step() on matched ICs, no structure (Philip: quantify the difference)

pulse.py -> pulse_10-08.log (P 300 periodic box, both rules; step() iso weights c = 2, dt 0.4 / sum W^2; walk m 0.8 xi 3.9,
c_s = 0.0453 / tick); dispersion.py -> dispersion_10-08.log (linear modes, analytic).

## Irrotational (Gaussian density / u bump of width w, zero flow; compared at the same front radius r_f = c t)
L2 difference of r x profile, normalised (0 = identical; walk density / eps vs step u / eps):
  w / xi   eps 0.01           eps 0.1            eps 0.3            eps 0.6         (r_f 20 / 35 / 50)
  4.1      0.08 0.12 0.16     0.11 0.15 0.19     0.17 0.22 0.26     0.26 0.32 0.37
  2.05     0.44 0.67 0.82     0.46 0.69 0.83     0.50 0.72 0.84     0.56 0.76 0.86
  1.03     1.2-1.3 (unrelated profiles; walk front 2x ahead, reaches the box half-width by r_f 35)
- Same law at long wavelength: both carry the bump off as an outgoing sound pulse (walk density obeys u_t = -c^2 div A,
  A_t = -grad u with c -> c_s); fronts coincide for w >= 4 xi.
- Dispersion: the walk is faster at short scales (Bogoliubov group velocity 1.03 c_s at k xi 0.2, 1.18 at 0.5, 1.63 at
  1) -> the difference grows with distance and with xi / w; step() (leapfrog) is non-dispersive to O(h^2).
- Nonlinearity: walk-only; at w 4 xi the L2 difference grows 0.08 -> 0.26 (r_f 20) from eps 0.01 -> 0.6. step() is
  exactly linear.
## Rotational
- step(): a smooth divergence-free A (poloidal ring flow, R 20, tube 4; discrete-div-free projection, change 9e-7) is
  frozen: after 200 ticks max |d phi| / max |phi| 1.3e-15, u stays 2.7e-16. Any transverse A is exactly static
  (u_t = -div A = 0 -> A_t = -grad u = 0): circulation can be stored, never moved.
- walk: no smooth vorticity at all (velocity = grad of a phase); circulation exists only on phase-singular lines, in
  quanta, and those lines move (self-induction + advection by the flow, 10-08 couple/).
## Reading
The two agree exactly where both are linear sound (long wave, small amplitude). Everything the walk does beyond that --
dispersion below ~4 xi, amplitude-dependent transport, quantized and mobile circulation -- step() has no term for: step()'s
only coupling is grad / div between u and phi, so its rotational sector is inert. The walk's extra content is the
nonlinear transport (advection of the flow by itself, via the density turn + phase), which is what carries circulation.

## Can step() carry circulation with one more relation? (carry.py -> carry_10-08.log, P 120 box)
Density-weighted flux (u's equation) cannot: step()'s phi update is a discrete gradient, curl of it is exactly 0, so
circulation is frozen whatever u does. Added the minimal non-gradient, state-dependent relation to phi's update:
phi_k += dt kappa W_k (A x B)_face . d_k (flow crossed with its own circulation; gradient part left out).
Smooth ring flow (R0 12, tube 3), u = 0 at start:
  kappa 0: frozen (centroid z 0, max u 4e-16).
  kappa +1 / -1: ring moves along its axis at +0.031 / -0.031 per time unit, steady over t 20 (z 0.61), shape kept
    (centroid rho 13.03 -> 13.06), sum|B| constant to 0.1 %. Direction set by the circulation sense x kappa.
  a0 2: speed 0.061 at start (linear in the flow amplitude), slows to 0.050 as the ring widens (rho 13.03 -> 13.13):
    larger rings slower.
  R0 18: 0.021 vs 0.031 at R0 12, ratio 0.68 (thin-ring estimate (R1/R2) ln(8R2/a)/ln(8R1/a) = 0.74).
  Not conservative as built: E (u^2 + sum phi^2 / 2) grows 0.9 % (a0 1) / 3.3 % (a0 2) over t 20 -- the omitted gradient
  part and/or the face averaging of A x B; to fix before reading anything energetic.
Reading: one relation between existing quantities (phi's change depends on phi and its circulation, locally) gives
step() self-transported circulation with ring-like scaling, with no new quantity and no phase. Whole-number circulation
is NOT produced (phi stays real-valued): that would need a periodic (holonomy) quantity.

## Energy-conserving form (carry.py --flow wsum --midpoint 3 --centred; carry_centred_10-08.log)
Cell flow in A x B = sum_k W_k phi_k d_k / (2 C0SQ) (the energy's own pairing -> first-order change exactly 0, checked
1e-16), implicit midpoint, time-centred on step()'s staggered phi. step()'s leapfrog invariant 5241.665-5241.668 over
t 20 at a0 2 (2e-6; was +3.3 %); ring motion unchanged (z 1.128 vs 1.133 at t 20). Explicit / plain-midpoint / Strang
versions drift O(dt) (pairing mismatch with the staggering, halves with dt).
