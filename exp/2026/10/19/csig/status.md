# 2026-10-19 csig: throughput inside the transport (queue item 1). grad/ showed sigma enters step() only, so a
# receiver's gradient never acts on what carries the structure.

Rule: alat.set_cartan_sigma(True) (C++ cartan_transport): each Delaunay triangle's three contributions scale by the
mean face sigma_f of its edges (sigma_f stored by set_face_medium). Continuum reading: -sigma(x) i_v da, i.e. the
carrier is slowed where the tick rate is low and the carried form is untouched. Pointwise v . (sigma omega x v) = 0, and
the discrete form stays antisymmetric per triangle -> sum phi out = 0 exactly, so energy is as exact as without csig.
(10-12 naive sigma x transport weighted the edges of one triangle differently -> broke the antisymmetry.)
Test: tst/substrate/test_continuous_cpp.py TestCartanTransport.test_sigma_weighted (uniform sigma scales the output,
varying sigma keeps antisymmetry + energy orthogonality to 1e-13, off ignores the medium).
zoo.py --csig.

## R 9 smoke (r9/, tilt 0.1, T 6)
| | dE/E | disp x t 6 | rot | rms2r |
| off | -3.0e-7 | -0.016 | 0.3 | +6.99 % |
| csig | -3.1e-7 | +0.017 | 0.8 | +6.98 % |
Energy unchanged (the residual is step()'s). Drift sign flips but both are tiny.

## R 20 (r20/, r20.sh; P 140, T 30, seed /bk/dst/2026/10/18/zoo/seed_ring_R20.npz; ends + ckpts /bk/dst/2026/10/19/csig)
tilt 0.03 / 0.1 with csig (compare grad/r20 without), dip 0.1 25 6 (receiver stand-in 25 cells along +x from a R 20
ring, just past its rim) off / csig.

## R 20 result (T 30, dE/E -1.7e-8 in all, same as without csig)
| run | disp x / z t 30 | sum|G| t 30 | Eu t 30 | outer shell t 30 | ring normal tilt (x-z, end state) |
| tilt 0 (grad) | +0.012 / 2.290 | 940.73 | 10.22 | 178.30 | +0.02 deg |
| tilt 0.03 | +0.015 / 2.291 | 943.38 | 10.26 | 178.32 | |
| tilt 0.03 csig | +0.017 / 2.291 | 943.40 | 10.25 | 178.32 | |
| tilt 0.1 (grad) | +0.018 / 2.292 | 950.42 | 10.61 | 178.24 | -0.49 deg |
| tilt 0.1 csig | +0.029 / 2.294 | 950.51 | 10.49 | 178.17 | -1.40 deg |
| dip 0.1 25 6 | +0.014 / 2.290 | 940.78 | 10.32 | 178.07 | +0.10 deg |
| dip csig | +0.016 / 2.280 | 940.85 | 10.22 | 178.09 | +0.15 deg |
(normal = smallest-variance axis of the top 2 % |phi|^2 cells; sigma is 0.9 at x -20, 1.1 at x +20 for tilt 0.1.)
1. csig makes the gradient steer the ring: the normal turns toward the slow side (-x for the tilt, toward the dip
   for the dip), 1.40 deg vs 0.49 without csig. Kinematic estimate (two halves moving at sigma v for t 30): 0.66 deg;
   measured extra from csig 0.91 deg -> same order, a bit more. Sign = refraction toward slower cells (lensing sign).
2. Nothing else changes: Eu, sum|G|, shells, speed equal to 0.1-1 %; no offload, no threshold, no energy exchange
   with the medium (by construction: the medium enters as a rate, not as an energy store).
3. Dip (receiver stand-in at the rim, sigma >= 0.93 on the near side): z speed -0.4 %, normal +0.15 deg toward it.
Reading: sigma in the transport is pure kinematics: the structure is steered (refracted) toward the slower region,
energy-exact, with no exchange. A receiver acting only through a throughput gradient can turn / attract a held
structure but cannot make it offload. Offload needs the receiver to touch what holds the structure (an energy-carrying
coupling), not only its rate.
