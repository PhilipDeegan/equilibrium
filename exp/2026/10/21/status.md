# Entries for 2026-10-21 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-21 — 10-20 written up; R 9 too small for reconnection (resolution-dominated) and the rule's continuum limit freezes topology (Lie-Poisson / Kelvin) -> any reconnection is lattice; R 20 tilted ring pair: steady approach, anti-parallel flattening, then a 1-2 cell hot spot at t ~75 and an accelerating grid-scale tangle (sum|G| +55 %, peak x16), no clean reconnection; event cost ~R^5; 8 cores again

1. **Valve run** (exp/2026/10/21/valve/status.md; zoo.py --struct side_tilt --ang --gap; slice.py plots): R 20 P 192 T 120,
   61 min at 8 threads (0.70 s / tick). Approach ~0.043 / t per ring, contact sheet by t ~55, grid-scale spot t ~75, then
   lattice tangle; dE/E -2.6e-8. Seed / end: /bk/dst/2026/10/21/valve/ (0.8 GB incl ckpt).
2. Prior stated: step() + Cartan (+ store) is Lie-Poisson -> Kelvin exact in the continuum for any H -> no reconnection
   in the limit; R 40 ~ 32x R 20 per event (no overnight) -> the valve needs a smooth frozen-in-breaking mechanism.
3. **Philip: the valve is probably not local** -- substrate update limits act like a valve but are just the limits of
   operation; the valve we want may be downstream of particle aggregates. -> stop hunting a local gate / threshold in
   step() + Cartan or in lone bodies / pairs; quantized offload is an aggregate property.
4. **Bind check on the 10-19 R 20 equal pairs** (valve/bind.py, seed vs end T 30): separation 55.426 -> 55.426 (same
   sense) / 55.416 (opposite) -- but sep = side / 2 puts each ring equidistant from its partner and the partner's image,
   so the x forces cancel by symmetry: that setup cannot show binding. A bind test needs sep != side / 2 (or a much
   larger box). Cartan unittests (incl test_cell_capacity, 8 tests) pass on the uncommitted 10-20 C++.


# Entries for 2026-10-21 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-21 — read `now` 10-20 + exp/2026/10/20/cap/status.md

State: finite per-cell store (alat.set_cell_capacity, p 4) works: over capacity -> sheds once to the ceiling -> held
ring AT the ceiling. A held circulation body is neither receiver nor emitter under any smooth coupling (store, sigma
well, csig, vcap): waves carry no circulation, the baroclinic channel only makes +- pairs. Net transfer needs the held
content itself to change (reconnection / topology change). 8 cores available again (Philip 10-21).

1. **Valve on structures** (old 10-18 item 2, now the main item): can a held body give / take circulation only by
   reconnection? Multi-winding ring (n 2, 3) or two rings brought into contact with the store on (p 4, relaxed seed):
   does reconnection change sum Gamma in whole units, is the threshold fixed under refinement (structure scale), and
   does the store's ceiling set when it happens?
2. **What the integer counts** (Philip's call, carried): moved quanta conserving sum n vs a boundary-only valve.
3. **Refinement of S_c**: per-cell ceiling clips a different fraction at R 20 unless S_c scales with h (substrate floor
   vs structure ceiling) -- decide before any R 20 capacity run.
4. Unequal bodies with the store (who gives, who takes) -- only once item 1 gives a transfer channel.
5. Carried: precession scan; Rauch-Werner 4 pi; farfield fast-fail test; barrier closure check; walk beta-clock removal;
   commit-ready check of the 10-20 C++ (test_cell_capacity); /bk/dst cleanup (bk_cleanup.sh ~73 GB, 25 GB walk states,
   ckpts /bk/dst/2026/10/{18,19,20}/ ~5.8 GB); MEMORY.md trim; lit search.

---
