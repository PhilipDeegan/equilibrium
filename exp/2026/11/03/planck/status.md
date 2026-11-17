# 2026-11-03 A3 step 0 + A1 step 1 precondition (doc/dst_versus.md)

box0.py: periodic BCC box n 32 (exp/2026/10/25/defect/bcc.py, discrete step()), dt 0.05, T 4000, seed = random-phase
u at |k| <= 3, phi 0. Energy per |k| shell + temporal power of u at 64 probe cells, windows [0, 400] and [3600, 4000].
  uniform: shells 0-3 share unchanged (0.02 0.12 0.53 0.33), content above k 4: 1e-28 (round-off) at t 4000.
           omega quantiles 10/50/90/99 % early = late (0.69 6.91 6.94 6.96).
  rough (fixed cell sizes 9 / 10 / 11, S 10): content scatters out in k by t ~800 (above k 4: 0.34), then plateau;
           omega quantiles early ~ late (0.69 1.18 7.07 7.13 vs 0.68 1.18 7.08 7.12) = elastic scattering, same
           frequencies. Equipartition (share = mode count) would put most content at high k: not approached.
  E/E0 - 1 oscillates within ~3 % (symplectic-Euler shadow energy at dt 0.05), no drift.
=> as expected: a linear fixed rule never thermalises (eigenmode energies are invariant); a fixed rough landscape
   only redistributes in k at fixed frequency. Equilibration needs exchanging bodies (step 1), not the field alone.

branches.py: full one-step map of the box (n 4, 6), eigenvalues: of 16 dof per unit cell, 12 static, 4 oscillating
(= 2 branches, the u sublattices, acoustic + optical). step()'s propagating content is scalar: no polarisation.
=> A1 step 1 (Malus through a polariser, firing frequency vs passed share) cannot be posed in step(): there is no
   polarised carrier to pass or block. Stage 1 (07-16) read a STATIC transverse-phi imprint, consistent with this.
   Needs a carrier with transverse propagating content (Cartan transport / a step_ae-like pair, but stated as
   loop / face relations, not E / B) and a threshold body. Neither the detector nor A3 step 1 is runnable yet.

   Correction (Philip): 'no polarised content' was an E-field reading. Polarisation as observed = cos² transmission
   through oriented filters; a scalar packet with a two-lobe cross-section at angle a gives exactly that by overlap.
   A1 step 1 is posable in step(): two-lobe packet through a slot at b; failure modes = lattice birefringence
   (lobe orientations travel at different speeds), loss of coherence over distance.
   Philip: wrong question anyway: polarisation implies symmetry, symmetry is a product of competing pressures ->
   don't seed the symmetric packet; ask what balance in emitter / carrier / filter forms the orientation.

cartan_box.py (A3 with exchange): periodic continuous BCC box P 48 (N 27.6k, side 27.7), seed = smooth random u and
div-free circulation, |k| <= 2 only; step_cartan (kappa 1, 3 iterations), T 300. Spectrum per sublattice FFT (one grid
for both sublattices aliased the seed into high k: fixed). Reference = the same pipeline on white noise in all dof.
  amp 0.3: content above the seed band 0 -> 0.85 (t 60) -> 0.992 (t 120, equipartition reference 0.994), then flat;
           shell shares match the reference shell by shell (0.001 0.002 0.005 0.008 0.017 0.029 ... vs 0.000 0.001
           0.004 0.007 0.015 0.025 ...). E/E0 - 1 <= 6e-5.
  amp 0.1: same, ~3x slower (0.15 / 0.42 / 0.79 / 0.93 at t 60 / 120 / 180 / 300): rate ~ amplitude (nonlinear).
  kappa 0 (step() only): spectrum unchanged to 3 decimals (control).
=> with Cartan transport the closed box thermalises, all the way to equal energy per mode in the rule's own dof, at
   the grid scale = Rayleigh-Jeans / the UV catastrophe in the current rule. Same failure as the 10-24 spheromak
   cascade (content runs to the smallest loops): one requirement, a stop at structure scale; the offload quantum is
   the candidate (doc/dst_versus.md A3). Side fix: continuous.spacing_of read a wrapped face in periodic boxes (h ~ box
   size, stable_dt 47x too big); now the minimum over cells (spheres unchanged; test_continuous 17 OK).
