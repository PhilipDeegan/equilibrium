# Entries for 2026-10-18 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-18 — reframes: no matter/EM split (E=mc^2); radiation is contextual (sinks direct it); tired light / redshift open again (expansion unprovable, periodic universe); offloading is quantized and transactional (Wheeler-Feynman-like), not random; a lump is not a structure. Switching E <-> M core = two independent radiators in the linear sector; saturating-throughput lumps are held by self-slowdown but leak, are scale-free and lose momentum; rounded-integer DEC is reversible but pumps noise; structure zoo harness built (step() + Cartan), R 9 resolution-dominated, R 20 batch running

1. **Option 1 (anapole + non-rotating part) closed on paper**: the hop / triangle sector is linear, so the field is a
   superposition; the wake cannot know the rotation; a luminal non-rotating field never settles (pancake keeps forming).
2. **Philip's reframes** (memory): no "matter sector" -- E=mc^2 says there is no mass / matter, one rule only; radiation
   is contextual (lightning -> ground, laser), intrinsic nonradiation is the wrong criterion (Purcell, Wheeler-Feynman);
   redshift may be tired-light-like, expansion unprovable / nonsense if periodic -> photon loss allowed if fractional,
   achromatic, coherent, frequency drop, packet stretch (supersedes 10-16 "no tired light"); offloading is quantized
   and not random (W-F analogue), valve may be hard in the continuum; a lump has no reason to remain bounded.
3. **Switching core** (exp/2026/10/18/switch/status.md; sw.py = nr.py + triangle magnetization): E <-> M quadrature
   core leaks exactly the sum of its parts (ratio 0.15-0.16 / unit distance); in-phase m perp p (Huygens) kills the
   on-axis backward wake 10x but total only -3 % (leak is sideways). Quadrature = cavity / standing-wave phase.
4. **Wolfram "atoms of space"**: particles = locally stable hypergraph structures, core + spread activity, rule-specific,
   found by search; rest mass = node "reuse". No derived photon. -> search, don't impose.
5. **Glider search** (exp/2026/10/18/glider/status.md; gl.py): DEC + saturating per-cell energy f(u) = b^2(sqrt(1 + 2u /
   b^2) - 1), Stormer-Verlet. Above a ~ 30 every seed collapses to a stationary lump (Ewin 0.4-0.9), held by its own
   throughput drop (fp 1-8 %); bulk converges under refinement (x, t scaling), central peak sharpens to the grid; leaks
   steadily (no plateau in 330 sig / c), slower at higher amplitude, no threshold; loses momentum to the lattice
   (Pz -70 to -80 %, not cured by refinement). Wheeler-geon-like. Closed: lumps aren't structures (Philip).
6. **Integer DEC** (gl.py --q): kicks rounded before the +-1 curl -> integers, charge exact, bit-reversible; but energy
   grows ~t^2 q^2 at every q (2^-12: +17 % by t 60; 2^-8: x38), grid noise everywhere. Rounding amplitudes = noise
   source, not a valve; integers must be MOVED counts (conserved), not rounded amplitudes. Open: what the integer counts.
7. **Structure zoo** (exp/2026/10/18/zoo/status.md; zoo.py, checkpoint / resume): closed vortex curves with integer
   circulation per curve (ring, coaxial pairs, concentric, Hopf link, trefoil, coils 1-3 / 1-5 / 3-2) under step() +
   Cartan, one scorecard (travel, rms2 shape, leak, sum|G|, E). R 9 screen: everything degrades 49-163 %, ranked by
   smallest feature = resolution, not physics. R 20 round 1 (ring, pair_co, trefoil, hopf_link) finishing ~15:47;
   round 2 (coil_1_3, coil_3_2, pair_counter, concentric) to be killed + resumed later (checkpoints auto).
8. /bk/dst cleanup: rm list printed (~73 GB approved / done-run caches; exp/2026/10/18/bk_cleanup.sh); Philip to run.


# Entries for 2026-10-18 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-18 (later today / next) — read `now` 10-18 + exp/2026/10/18/{zoo,glider}/status.md

State: one rule, no matter/EM split. Held structures need an organising reason (circulation / winding / topology);
lumps don't count. Offloading must be quantized + transactional (moved counts, receiver-triggered), never a steady
trickle. Radiation and redshift are contextual (photon loss allowed if fractional, achromatic, coherent).

1. **Finish the R 20 zoo**: round 2 resumes from checkpoints (seeds + _ckpt next to them in /bk/dst/2026/10/18/zoo):
   cd /home/p/git/eq && export OMP_NUM_THREADS=1 PYTHONPATH=. && D=exp/2026/10/18/zoo/r20 && (for s in coil_1_3 coil_3_2
   pair_counter concentric; do echo $s; done) | xargs -P 4 -L 1 sh -c 'python3 -u exp/2026/10/18/zoo/zoo.py --struct $0
   --P 140 --R 20 --smooth 4.938 --T 30 --seed /bk/dst/2026/10/18/zoo/seed_$0_R20.npz > '$D'/$0.txt 2>&1'
   (~45 min; checkpoints at t 11 / 9 / 5 / 0, partial logs r20/*_part1.txt -- the resumed log overwrites *.txt, RESULT uses
   the full history from the checkpoint). Round 1 done: ring / pair_co / trefoil / hopf_link all converge (status.md).
   Then compare R 9 -> R 20 per structure (residual shrink ~30x = discretisation; anything that does NOT shrink is
   structural). Fix rms2 for rotating / precessing structures (best rigid motion, not shift only) before ranking knots.
2. **Valve on structures**: multi-winding ring (n 2, 3) under Cartan: does it shed only in whole windings (reconnection =
   topology change), and is the threshold fixed under refinement (structure-scale), unlike the 10-16 zclock (floor)?
3. **What the integer counts** (Philip's call): moved quanta conserving sum n exactly (walk / lattice-gas lineage risk)
   vs a boundary-only valve (imposed). Rounded amplitudes are out (10-18 int/: noise pump).
4. **Glider diagnostic doubt**: D x B momentum loss in the saturating rule should vanish in the continuum for a local
   H(D, B) -- check the diagnostic (cell reconstruction of E, B) before quoting "lattice absorbs momentum".
5. Carried: tick smooth limit == step() test; farfield fast-fail test; barrier closure check; walk beta-clock removal
   (asked 10-08); /bk/dst: Philip to run exp/2026/10/18/bk_cleanup.sh (~73 GB) + decide the other ~25 GB of walk states
   (03/pbox, 05/coil N4 R160, rest of 06/, 07/slosh); MEMORY.md trim (over limit); lit search (anapole / luminal
   nonradiating sources; discrete Lie derivative; geons).

---
