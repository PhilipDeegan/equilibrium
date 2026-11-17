# 10-08 clock: cell.L / L0 as a local clock in the walk (Philip: L is how the continuum model gets discretized)

Implemented (walk.py Walk / CppWalk beta=, C++ AdaptiveLattice::set_walk_clock, step_walk): sigma = 1 + beta (rho0 -
rho1) / rho0 per cell after the hop; on-site rates scaled by sigma: extra mass-coin turn (sigma - 1) om0 (checked: shifts
the rest frequency exactly, om0 -> om0 + phi) and density turn g sigma rho1. Per-cell rotations -> norm exact,
reversible (sigma read from rho1, which they keep). Hop (speed) NOT scaled. Tests: tst/substrate/test_walk.py
TestWalkClock (C++ == numpy at beta +-0.5, norm to 1e-12 over 200 ticks, uniform state turns at sigma (om0 + g rho)).

Finding (before any coil run): to first order the on-site clock is a change of g:
  omega(rho) = (1 + beta (1 - rho)) (om0 + g rho)  ->  d omega / d rho at rho0 = g - beta (om0 + g).
om0 = 0.927 (m 0.8) >> g (0.0082 at xi 3.9, 0.0041 at xi 5.515), so
  - beta > g / (om0 + g) = 0.0088 (xi 3.9): background focusing -> modulational instability (C++/numpy round-off
    grows ~0.3 / tick at beta 0.5 in the sphere test);
  - beta < 0: g_eff = g + |beta| (om0 + g); beta -0.5 -> x58, xi_eff ~0.5 cells at ANY xi:
    xi_eff^2 = n / (6 m |beta| om0) -> the clock ties the core to the walk's Compton scale (om0 per tick), a length in
    cells that does not scale with xi. That is a second length, but on-site it only resets xi; the beyond-linear part
    (-beta g rho^2) is O(g beta), tiny.
So depletion-dependent on-site clock = choosing xi in Compton units. Physically meaningful |beta| ~ g / om0 ~ 1 %.
The c ~ L part (speed) needs a position-dependent hop: not unitary as n(x) W; a split-step walk C(theta(x)) diag(W, W^dag)
(local coin carries both c ~ cos theta and m ~ sin theta, exactly unitary, Arrighi / Di Molfetta curved-space walks)
would carry it, but is a different walk from the current one (mass acts on the shifted state) -> coils to re-validate.
