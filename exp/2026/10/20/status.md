# Entries for 2026-10-20 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-20 (recorded 10-21; shutdown before write-up) — finite holding capacity (form 2, per cell + stiffening) built in C++ + tested; a finite store sheds once to its ceiling, then the ring is held AT the ceiling; the early "shed" was entirely readjustment of a linear-relaxed seed; a held ring is not a receiver (a wave has no circulation, passes through); a sigma well gives a baroclinic wave <-> circulation channel but only in +- pairs (net zero) -> every smooth coupling only reshapes / steers; net transfer needs a change in what is held (reconnection / topology change)

All in exp/2026/10/20/cap/status.md (scripts r9.sh, r9p.sh, relax.sh, push.sh, well.sh; logs r9/ relax/ push/ well/, all
runs finished; states /bk/dst/2026/10/20/cap/{end,relaxed}_*.npz + ckpt_* ~1.7 GB). Code uncommitted: adaptive_lattice.hpp,
lattice.cpp, zoo.py, test_continuous_cpp.py.

1. **Design** (Philip chose per cell + stiffening): S_j = 1/4 sum_k phi_jk^2, H = sum u^2 / 2 + sum_j G(S_j),
   G' = 1 + (S / S_c)^p. dH/dphi_f = phi_f (G'_j + G'_n) / 2 drives u in step() pass 1; Cartan as a Lie-Poisson bracket
   (omega = loop sum of a, velocity slot b = dH/da) -> energy exact per triangle; linear limit == old code.
   alat.set_cell_capacity(S_c, p); test TestCartanTransport.test_cell_capacity (off == plain, orthogonality 1e-13,
   u update == formula, H drift O(dt^2)). zoo.py --scap / --cap_p / --Sc / --relax / --relax_tau / --pulse / --ediv /
   --smax; dt / sqrt(1 + (2p+1) / scap^p).
2. **R 9 screen, p 1** (lone ring P 96 T 20): Eu jumps by t ~1 in proportion to the excess (2 / 5 / 10 / 20 % of E at
   scap 1 / 0.5 / 0.25 / 0.1), then flat; outer shell grows. But the core never settles below S_c and p 1 is a soft
   store everywhere (G' 1.7 at 0.7 S_c): saturated core carried faster, shears, winds up (sum|G| x9 at 0.1) = tangle.
3. **Steep store** (p 2 / 4): p 4 scap 0.7 clips once, sheds into u, ends with no cell above S_c, speed and sum|G| back
   at linear values -> held ring at the ceiling. p 4 sheds 41 % of the store's excess, p 2 29 %, p 1 17-27 %.
4. **Relaxed under the store** (--relax, step() only, u damped, kappa 0): seed 1.43 -> 0.99 S_c. Then lone P 96 ==
   P 144 to 1e-3, Eu 1.1-1.3 % (linear dressing ~1 %), no trickle, no cycling; equal pair = no exchange (symmetry).
   -> items 2-3's burst was readjustment to the new H; a finite store gives "excess leaves once, then held".
5. **Push** (plane u pulse at a ring held at the ceiling, AMP 0.1 / 0.3 / 0.5, up to 1.7x the ring's energy): core
   peaks at 1.02 S_c, no capture, no re-shed; store == linear to ~2 %; div-free energy dips <= 1.5 % and returns.
   Structural: held content is circulation (Kelvin under step() + Cartan), a wave is a gradient (zero loop sums) ->
   cannot add / remove it. Not a receiver -> not an emitter (Philip 10-19: emitter = receiver).
6. **Sigma well** (dip / bump +-0.3, width 6, on the ring; linear store): baroclinic term (-W sigma_f grad u, loop sums
   where sigma varies) exists and changes the pulse's reversible div-free dip by +-15 % (odd in depth), but the
   circulation it makes is even in depth = +- pairs, net zero (~1 % sum|G|). Ring's net circulation untouched.
7. **Conclusion**: store, sigma well, rate / carrier bounds (10-19) -- every smooth route only reshapes or steers. A net
   transfer between bodies needs a change in what is held: reconnection / topology change (10-18 "moved counts").


# Entries for 2026-10-20 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-20 — read `now` 10-19 items 8-14 + exp/2026/10/19/{csig,recv,vcap}/status.md

State: held structures (step() + Cartan) hold exactly and offload nothing -- they have no capacity limit (circulation is
a continuous label, energy smooth in size) = Philip's "proton storing infinite energy -> no photons". Philip 10-19:
systems prefer the lowest energy; offload because a body's capacity is finite; thermalization IS equilibrating with the
surroundings; any emitter is also a receiver, raw vacuum is not (photons would decay); offload can cycle A -> B -> A,
the space between holding up to 2x (superposition); lam is a side effect of a structure (rate, its gradient steers),
charge capacitance is also structure-caused but distinct from lam, and reaches outward (Coulomb). Rate bounds shed
nothing (csig = steering only, vcap = slows + less Eu). Periodic images are a lone body's receivers. 4-core cap unless
Philip says otherwise (ask at session start); zoo.py diagnostics now ~3x cheaper (byte-identical).

1. **Finite holding capacity (form 2)** -- the main item. Per-face store F(phi): phi^2 / 2 below a ceiling, rising
   steeply past it (smooth, no clamp). u update with F'(phi) in step(); Cartan re-weighted so sum F'(phi) out = 0
   (energy exact in the new H = sum u^2 + F); design on paper first, then C++ (set_face_capacity or similar), test
   (energy orthogonality, F -> phi^2 / 2 limit == current), R 9 dE/E check. Reading to test: uniform per-cell ceiling
   (substrate) + the structure's field shape = structure-dependent capacitance, nearest cells saturate first
   (Schwinger-like) -> excess leaves. Ceiling picked for now (R 9 core |a| max 0.61, p99.9 0.38; R 20 0.28), derive later.
2. **Lone ring above capacity**: ceiling below the core value -> does it shed into u (Eu, shells), settle at a held
   state (which?), and does the settled content depend on box size (images as receivers)? R 9 screen, then R 20.
3. **Two bodies with capacity**: equal rings (recv/r20 setup, P 192, side / side_opp), one above capacity: does the
   excess cycle A -> B -> A, and does the in-flight energy approach 2x (superposition)? Then unequal bodies at equal
   speed (who gives, who takes).
4. Lone-ring Eu "cycle" (old item 3): largely answered -- first rise is local dressing (once), later swings are the
   periodic images (recv/status.md); remaining: refinement of the dressing level (~0.87 % at R 9, ~0.25 % at R 20).
5. **Precession scan** (optional): (p, q) torus coils at R 20 T 30 -- which embeddings precess, rate vs p, q, tube radius.
6. **Rauch-Werner 4 pi as an observation**: needs a resolvable precessing structure over several turns (~9 h at R 20:
   think before running).
7. Carried: farfield fast-fail test; barrier closure check; walk beta-clock removal (walk is reification: remove?);
   tick smooth limit test (moot?). /bk/dst: Philip to run exp/2026/10/18/bk_cleanup.sh (~73 GB) + decide ~25 GB walk
   states; deletable ckpts: /bk/dst/2026/10/19/{grad,csig,recv,vcap}/ckpt_* (~2.9 GB), /bk/dst/2026/10/18/zoo/*_ckpt.npz
   (1.2 GB); MEMORY.md trim (over limit, truncated); lit search (Purcell / inhibited emission as context-set offload;
   capacitance of an isolated body vs field reach; anapole; discrete Lie derivative).

---
