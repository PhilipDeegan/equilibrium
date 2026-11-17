# 2026-11-03 llvm-mca pass over the hot C++ kernels

Setup
- Markers: inc/eq/core/mca.hpp, EQ_MCA_BEGIN/END, active only with -DEQ_LLVM_MCA. Placed as the first / last
  statement of the per-cell loop body (discrete _step_pass1 / _step_pass2_uniform / _step_pass3, continuous _pass1 /
  _pass2, cartan_transport tri / face loops) and around the cartan_mid loop (the only one that vectorizes; a marker
  inside its body would block that).
- Text-order regions fail on these loops: outside-the-loop markers get tail-duplicated (overlapping BEGINs), inside-the-
  body markers land with END in the rotated latch above the header. mca_regions.py extracts each marked loop (and its
  child loops) from clang's "in Loop: Header=" block annotations into regions/<name>[.<child>].s.
- asm: clang++-19 -O3 -funroll-loops -march=native -DNDEBUG -DEQ_LLVM_MCA -S src/eq/lattice.cpp (include flags in
  inc.txt); mca: llvm-mca-19 -mcpu=znver5 -bottleneck-analysis.
- Caveats: mca assumes L1 hits (all neighbour gathers free) and treats every block of a region as executed each
  iteration (cold hold / sig / cap / vcap branches, guard slow paths included) -> upper bound on core cost, no
  memory cost. Whole-loop regions with nested loops count the inner loop once -> use the inner-loop regions.

Inner per-face loops (2 faces / iteration except cartan_tri = 1 triangle):
  discrete_pass2_uniform  12.5 cyc / 2 faces   1 div / face (nb.sigma() = 1/(1+lam))
  continuous_pass1        14.5 cyc / 2 faces   1 div / face (_coef -> nb sigma)
  continuous_pass2        22.5 cyc / 2 faces   2 div / face (no-medium path); 8.4 with face medium (coef array, 0 div)
  discrete_pass1          49.4 cyc / iter      5 div
  cartan_tri             146   cyc / triangle  7 div + sqrt (x12 per cell); phi / w[k] divides by a constant table
  cartan_mid (vector)      8.2 cyc / 32 doubles (zmm, interleave 4) -- fine, stream-bound in practice
Backend pressure 65-92 % on the scalar loops, mostly data dependencies (serial acc += over 14 faces, divide latency).

Candidates (not done)
1. phi / w[k] -> phi * inv_w[k] in cartan_transport (3 divides / triangle, 36 / cell). Changes rounding (last bit).
2. sigma() = 1 / (1 + lam) recomputed per neighbour per face in every pass -> per-cell sigma array once per step.
3. cl[g] decode (g >> 12 chunk, g & 4095, 96-byte chunk descriptor, per-field base pointer loads) on every neighbour
   access; continuous does cl[cl.leaf_indices[n]] (double indirection) although an.nbj is already leaf-indexed ->
   flat leaf-indexed u / sigma arrays.
4. anchor::tris() is a function-local static: guard check inside the triangle loop -> hoist the reference.
5. Alias reloads: fixed_phi mask per face (pass2_uniform), captured `a` in cartan_mid's scalar fallback.
6. Split accumulators for the 14-face sums: reassociation, not bit-identical; only if the round-off contract allows.
Real cost needs perf (cache misses / IPC) on a representative run before choosing.

Compiler check (Philip switched the build to clang 22 today; mkn -x mpi -> ccache mpicxx, OMPI_CXX=clang++ -> llvm-22)
- Redone with /usr/lib/llvm-22/bin/clang++ + llvm-mca-22 (lattice22.s, regions22/): per-face inner loops identical to
  clang 19 within ~1 % (pass2_uniform 12.5, continuous pass1 14.5, pass2 22.5 / 8.4, cartan_tri 148, cartan_face 43.6
  cyc / iter); cartan_mid vector body 8.19 cyc / 32 doubles (zmm), same as 19. Clang 22 lays the vector blocks out
  after the END marker -> extractor now runs unmarked-loop regions to .Lfunc_end.
- Debug flags (-gW 9 = -g3 -pg -fno-omit-frame-pointer, no NDEBUG) cost 0-4 % in the inner loops (cartan_face 21 %);
  the larger cost is outside mca: EQUILIBRIUM_DEBUG_DO check_total_dphi_balanced every discrete step(), asserts, mcount.
- Release build: mkn clean build -KOqdp mod,mod_lattice -x mpi -g 0 -W 9 (dry-run -R: -DNDEBUG -O3 -funroll-loops
  -march=native -mtune=native, no -g / -pg).

Before/after snapshot (snap.py; baseline snap_baseline.npz, release build -g 0, clang 22, 12:16)
- 8 fixed-seed cases x 20 steps: discrete step (lam), continuous step (lam / face medium), step_cartan plain / sigma /
  hold / vcap / cap. Each option changes the output 4-45 % vs off; all finite.
- Deterministic: 4 vs 1 threads bit-identical. Release (-g 0) vs debug (-gW 9) build bit-identical.
- Usage: PYTHONPATH=$PWD python3 -u exp/2026/11/03/mca/snap.py compare  (exit 1 if any case differs).
- Gap: discrete case is a uniform sphere (pass2_uniform); AMR path (_step_pass2 / _face_flux_coefficient) not covered.

Fixes applied (inc/eq/core/data/lattice/continuous/adaptive_lattice.hpp only; discrete untouched)
- per-leaf sigma (an.sig) filled once at the top of step() (no medium); _coef reads an.sig[j], an.sig[n]
- 1/h hoisted out of _pass1 / _pass2 (the phi store blocked the compiler from hoisting it)
- anchor::tris() reference hoisted out of cartan_transport's loops
- cartan_face coefficient 0.75 / (h w[k]) as a per-k table
- cartan_transport: phi / w[k] -> phi * (1 / w[k])  (the only non-bit-identical change)
snap.py after each: all but the last bit-identical on 18 / 18 outputs; after 1/w the 10 cartan outputs differ by
<= 2.6e-16 of max (20 steps), all step() / discrete outputs identical. unittest test_continuous_cpp (29),
test_continuous (17), test_continuous_spheromak (2): OK.
mca (clang 22, znver5) inner loops before -> after: continuous pass1 14.5 -> 8.4, pass2 22.5 -> 14.2, cartan_tri
148 -> 129, cartan_face 43.6 -> 39.1 cyc / iter.
bench.py (R 16, h 0.5, N 178119; pre-fix .so built from HEAD headers via mkn -f, min of 5):
  1 thread : step 10.70 -> 9.01 ms (-16 %), step_cartan 95.6 -> 86.3 ms (-10 %), with lam 96.5 -> 84.9 (-12 %)
  8 threads: step_cartan 24.7 -> 22.1 ms (-10 %), with lam 22.4 -> 20.5 (-8 %); 8-thread step() too noisy (1.5-2.0 ms)

mkn clang.llvm-mca mod (12:51, todos done): mkn build -KOqp mod_lattice -x mpi -g 0 -W 9 -a -DEQ_LLVM_MCA
  -m 'clang.llvm-mca{link{regions: only}}'  -> per-region results in bin/mod_lattice/res/<hash>-lattice.cpp/,
  picks /usr/lib/llvm-22/bin/llvm-mca through ccache mpicxx, -mcpu=native (= znver5). Rebuild without -a after.
CORRECTION: mca_regions.py misses the header block of nested loops (clang puts "This Loop Header" on the line
  after a "Parent Loop" label), so its inner-loop numbers above undercount (index load + n < 0 test dropped).
  Same extractor before / after, so the direction holds; bench.py timings are unaffected. Use the mod.
  Mod numbers, post-fix code (cyc / iter): continuous pass1 inner 8.37, pass2 inner 8.99 (medium) / 12.51,
  cartan_tri inner 136.4, cartan_face inner 49.0, discrete pass2_uniform inner 12.4.

perf (13:00, perf record on bench.py --only step_cartan, 1 thread): cartan_transport face loop (_cart_sum) 52 %,
triangle loop 15 %, _pass2 6 %, step_cartan load / store copies ~8 %, _pass1 3 % -- the reverse of mca's ranking
(mca counts cold option branches; the face loop was a heap-vector table walk with a stack-spilled accumulator).
- Face-table walk: _cart_table() (std::vector per face) -> fixed-capacity CartTab (entries per face 0..6: faces
  0, 8, 10, 12 none, face 7 six), hoisted out of the loop. Bit-identical (snap_current.npz, 18 / 18).
- Two-pass variant (each face sum once into an NF*L scratch, then differences; halves the sums): bit-identical,
  1 thread 76.8 -> 73.8 ms but 8 threads 20.3 -> 24.1 ms (extra 20 MB array + barrier) -> REVERTED, one-pass kept.
Totals, step_cartan R 16 (N 178k), best of 5 vs the HEAD build: 1 thread 94.9 -> 76.4 ms (-20 %),
8 threads 23.2 -> 20.0 ms (-14 %). test_continuous_cpp OK after the final build; test_continuous /
test_continuous_spheromak OK on the two-pass build (same arithmetic).
Next if wanted: 8-thread profile (memory-bound now?), step_cartan load / store copies through the slab (~8 %).
