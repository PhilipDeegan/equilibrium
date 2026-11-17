# Charge in the continuous E/A model — 2026-09-29

Question (Philip): the vacuum needs capacitance for light to travel -> test the capacitive and
inductive halves. Inductive half already done 09-25 (exp/2026/09/25/induction: mutual induction
with mu0 = 1/c^2 to 0.03% at h = 0.125, 2nd order; TestStaticCurrentBalance). In AE, eps and mu
are in the equations (u_t = -c^2 div A, A_t = -grad u - E, E_t = c^2 curl curl A - J), so what
can be tested is whether the lattice operators deliver them: charge conservation and statics.

1. Charge = div E (sum_k W_k e_k per cell). With the DEC curl it is exactly invariant: random
   state, 200 steps, change 1.5e-13 (|div E| 17.6). With a current: change = -dt div J to 1e-14
   (continuity). LS curl: O(1) change on the same random state (3e-3 on a smooth field) -- LS leaks
   charge.
2. `relax.py`: seed only a charge (E with div E = +1 at the core, -1 on the outer shell of a closed
   lattice), u = A = 0, nothing held, no sponge, R = 8, h = 1. Time-averaged u over the second
   half of 40 crossings = psi (the discrete Poisson solution) with scale 1.0006, rel rms residual
   3.8e-5 (12 crossings: 0.994, 2.7e-4). div E change 2e-15.

Correction (Philip, same day): the seeded div E has no carrier. AE defines rho = div E with no
Gauss constraint tying it to matter, so the model admits "charge without particles" only because
it was put in by hand. J is a prescribed stand-in for moving matter, not a vacuum DOF; with J = 0,
div E is conserved at 0 -- the vacuum cannot make charge. What these runs establish: exact charge
conservation (DEC) and correct statics around a given div E. Missing: the link from matter to
div E -- the lam/sigma coupling in step_ae (div(sigma A) at the face), then test (a) a moving lam
structure produces J that changes div E, (b) div E never appears or vanishes without one (Gauss).
In step() (E = 0) charge = div phi = -u_t is not conserved at all.
Tests: tst/substrate/test_ae_charge.py.
