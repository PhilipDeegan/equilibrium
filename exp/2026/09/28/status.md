# Entries for 2026-09-28 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-28 — Hodge star ratio settled: discrete step() 3:1 (Kelvin geometry, was 9:1 by accident), continuous model 2:1 (isotropic); both implemented, built, tests triaged; AVX-512 alignment bug fixed

1. **Star ratio** (`exp/2026/09/28/hodge_star/status.md`, `equilibrium/substrate/bcc_symbol.py`,
   `tst/substrate/test_hodge_star.py`): r = *1_hex / *1_sq on the BCC Delaunay / Kelvin complex. Any r
   is consistent (c_u^2 = (r+1)/4, c_T^2 = 4/(r+1)); r = 2 is the only ratio whose 2nd-order error is
   direction independent in BOTH sectors (no birefringence; analytic for u: 4th moments of the two
   shells iso iff c_hex = 2 c_sq) = doc/dst.md's w 2/3:1/3; r = 3 is the Kelvin geometry (|dual face| /
   |link|) and the only r with c_T = c_u at the geometric *2; step() was r = 9 (W = 3 on both legs;
   the C++ comment says "Kelvin geometry area ratios" -> the geometric 3 applied per leg).
2. **Philip's principles** (mems project_lattice_fundamental, project_absolute_zero_floor): the discrete
   lattice is fundamental; the continuous model (and its coarse Kelvin geometry) only approximates it
   and is continuous at every h (no level-0 identity requirement); coarse Kelvin cells are there to
   avoid a preferred axis -> isotropy is the coarse criterion. -> discrete 3:1, continuous 2:1.
   Discrete model only for pure field / vacuum tests from now on.
3. **Continuous model** (C++ `anchor::Weights {iso, discrete, legacy}`, `AdaptiveLattice.set_weights`,
   per-instance C0SQ; numpy `continuous.set_weights`, AnchorCluster checks they match): default iso
   W = sqrt2:1, C0SQ = 4, c = 2. DEC star derived: r = (W_hex/W_sq)^2, s = (r+1)/4 (c_T = c for any r;
   reproduces 09-27's (9, 5/2), (3, 1)). legacy = old 3:1; the 20 pre-09-28 exp scripts pinned to it.
   Iso dipole benchmark (`exp/2026/09/28/iso_dipole/status.md`): source work 0.4216 / 0.4145 at
   h = 0.5 / 0.25 (err +0.0102 / +0.0031 vs 0.4114; 3:1-weights dec +0.0136 / +0.0036), flux same.
   Found: the dipole scripts' source work hardcoded 3/80 = 1/(2 C0SQ) at 40/3 -> now 1/(2 C0SQ).
4. **Discrete step()**: W_HEX = sqrt3 (bcc::face_weight, substrate.hpp FACE_W, lattice._W); new
   face_star / lattice.W_STAR = 3:1 for complete operators (leaf Laplacian, curvature estimate, wsum,
   step_lam_force, mhd phi-diffusion / vector Laplacian: unchanged). c 3.651 -> 2.309
   (lattice.wave_speed()), stable_dt 0.4/78 -> 0.4/30. doc/dst.md weights section rewritten.
5. **Tests** (phlop, 429 ran): fixed phi_diffuse, maxwell (identity value (W_HEX - W_SQ) a),
   casimir_polder (C_COEFF was hardcoded sqrt10 -> sqrt(W_HEX^2 + W_SQ^2); Yukawa xi by decay length),
   multiclock (re-pin 0.723 = c ratio), continuous tests (legacy/discrete/iso references), double_slit
   run() now in wavelength / travel units with a period-averaged reflection metric. Open: see `next`
   item 2 (double-slit reflection marginal in both versions; casimir ratio3 normalisation; Coulomb
   plateau needs ~45k steps, -1.927 at R=20; saha / sponge / mhd step-tuned tests).
   Thread counts in tests -> `EQ_TEST_N_THREADS` (mem feedback_threads); runner command in
   mem feedback_test_runner.
6. **test_lazy segfault**: `inc/eq/nanobind.hpp` allocated with a hardcoded 32-byte alignment; on
   AVX-512 (this Zen 4, -march=native) mkn::avx needs 64 -> GP fault on aligned __m512d access. Now
   `mkn::avx::Options::ALIGN()`; LazyNumpyArray visibility hidden (warning gone). Philip updated
   mkn.avx (was stale); not the cause.

---
