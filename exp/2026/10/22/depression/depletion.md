# 2026-10-22 depletion relation (paper). Philip: structure = substrate depression; relations before geometry;
# spheromak is the trusted geometry (acceptance target).

## Reading of the current rule
step() = linear acoustics: u_t = -c^2 div A, A_t = -grad u (c = 2 in Weights::iso). Cartan adds the vortical
self-transport -i_v da. Together = compressible Euler with two terms missing:
  (a) mass weighting: the flux that moves u is rho v, not v;
  (b) Bernoulli: A_t also gets -grad(|v|^2 / 2) (kinetic content pushes on the scalar).
Without them a fast core does not lower the scalar around itself -> no depression, nothing depletes.

## The relation: per-cell density rho_j carried by the existing scalar slot, H(a, rho)
  H = sum_j rho_j K_j + e(rho_j),   K_j = 1/4 sum_k phi_jk^2 (= the store's S_j),   e = c^2 (rho ln rho - rho + 1)
  rho_t = -div(dH/da)                       step() pass 1 with face flux phi_f (rho_j + rho_n) / 2   (continuity)
  a_t   = -i_v da - grad(dH/drho)           Cartan with slot b = dH/da (as the store already does) + step() pass 2
                                            on B_j = K_j + c^2 ln rho_j   (Bernoulli)
Energy: dH/dt = <dH/da, -i_v da> + <dH/da, -grad B> + <B, -div dH/da> = 0 + (by parts) 0. The discrete form keeps it
exactly as the 10-20 store did: Cartan antisymmetric per triangle in b = dH/da; step() pass 1 / 2 share one face
coefficient (adjoint pair). Code pattern = set_cell_capacity with G(S) -> rho K + e(rho), plus pass 2 reading B.
Implicit midpoint (as step_cartan) since H is non-quadratic (09-30: Euler step O(dt) energy error otherwise).

Limits:
- no structure (K -> 0, rho = 1 + u / c^2 small): e ~ c^2 (rho - 1)^2 / 2 -> H = c^2 u'^2 / 2 + sum K: today's rule
  (u rescaled), exactly in the linear limit.
- isothermal e: h = c^2 ln rho. Steady line vortex: v^2 / 2 + c^2 ln rho = 0 -> rho = exp(-v^2 / (2 c^2)).
  With v = n / r (n turns, h = 1): rho -> 0 smoothly AT the line only, kinetic content rho v^2 finite (no infinity,
  no clamp), half-depth radius r_1/2 = n / (c sqrt(2 ln 2)) ~ 0.42 n cells.

## What it gives / does not give
Gives (on paper):
1. Self-made, co-moving depression (cyclostrophic: fast core -> low scalar), not a picked external well.
2. Default core width ~ n / c set by the content and the substrate's own c -> deforms, returns (spongey + default).
   Full depletion needs v ~ c at the core: structures with near-c circulation (cf. walk "flux at c", zitterbewegung
   c-transit) deplete, slow ones barely do (today's R 20 seeds |a| <= 0.6 -> rho >= 0.96).
3. Spheromak (Beltrami, omega x v = 0) stays steady at rest: Bernoulli B = const -> rho lower where |v| high = its own
   depression. Compact steady Euler flows exist in the continuum (Gavrilov 2019, Constantin-La-Vicol 2019 -- to check),
   so an isolated held spheromak is not ruled out by a virial argument as in MHD.
Does not give:
4. Quantization: a stays a real label; still Kelvin (barotropic) -> frozen-in. What changes: where rho = 0 the edge
   content a costs nothing (rho K = 0) -> a is unconstrained there. Whether that is enough for a smooth slip /
   reconnection through a depleted region is the first thing the antiparallel test must show; if not, a dispersive
   (cell-scale) term is the next relation to look at, and periodicity is still open (Philip 10-12: 2 pi = pressure
   equalisation, emerge not force).
5. Resolution cost: one whole turn has a sub-cell core (0.42 cells); a resolved core needs many turns (n ~ 20 -> 8
   cells), cf. the tight-winding picture (~60 windings).

Reading of rho in substrate terms (not reified): cell-count density / degree of compression (L = h / <ell>); a depleted
core = locally larger cells. Kept as a reading only.

## Tests (geometry-free first; spheromak last)
T0 unit (CI-safe): rho = 1 + small -> matches step_cartan to O(u^2); energy orthogonality per triangle / pass 1-2
   adjoint to 1e-13; H drift O(dt^2).
T1 straight line, periodic box, n = 10, 20: relaxes to rho = exp(-v^2 / 2c^2)? core width vs analytic (finite-source
   reference), h vs h / 2 converged.
T2 antiparallel straight pair at a few separations: approach, depleted overlap, does the linked content change, is it
   in whole units, where does the energy go (u waves out), refinement.
T3 spheromak (relaxed seed) at rest: stays, own depression forms, perturb -> returns.

## Philip 10-22: decay is fundamental -> assert the relations with spheromak decay (electron pinch = long term)
Phenomena (lab spheromaks, not a mechanism story) a rule must reproduce, all measurable in sph.py:
D1 it decays: vortical energy falls, goes out (here: into u, outward shells) -- not imposed by a picked diffusion.
D2 selective decay: helicity falls slower than energy -> lam_eff = sum A.curlA / sum A^2 drops toward the lowest
   state (Taylor); a free (unconfined) spheromak also tilts / shifts.
D3 the rate is structure-scale: converges under refinement (a 9 -> 20 -> 40 at fixed shape), not a lattice residual.
Expected for step() + Cartan (running, sph.py run.sh): D1 fails or only as a lattice residual (frozen-in, conservative,
steady Beltrami) -> asserts a missing relation. Depletion relation: a deforming core radiates into u through the
Bernoulli / mass coupling (Lighthill-type acoustic offload, real), so D1 can happen without a diffusion term; D2 needs
frozen-in breaking at rho -> 0 (open).

## CORRECTION (Philip 10-22): this design is backwards
The substrate is what fluids sit on; compressible Euler / MHD are fluid DESCRIPTIONS, limits to reproduce, not a
source for a substrate relation. And the sign is wrong: bound structures hide flux from the background = compression
-> discrete cells CONTRACT toward the centre (L = h / <ell> up), where this design empties the core. Superseded by a
hidden-flux -> cell-contraction -> junction-medium relation (to design).
