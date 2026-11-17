# Where could the integer count live? Lattice geometry, 2026-10-19

Purpose: before deciding what the integer counts (next 10-18, item 3), lay out the geometry we actually have, where a
whole count can sit, and how "quantized and probably rotating" could be realized on it. Figures come from
`geom_plots.py` in this directory. Fig 4 is computed; figs 1-3 are exact geometry; fig 5's right panel is zoo data.

Constraints carried in (Philip):
- no internal angle space, no vectors, no complex numbers: rotation must be something going round actual cells;
- one rule (no matter/EM split);
- counts are moved, never rounded (10-18 int/);
- every element has a base floor > 0 (Casimir: the substrate transmits with zero particles, 10-19);
- a receiver is felt through the tension-gradient path built by all scales (10-19).

## 1. The cell and its relations (fig1_cell.png)

![cell](fig1_cell.png)

BCC lattice. The Voronoi cell is a truncated octahedron with 8 hexagonal faces (<111>, nearest neighbours) and 6
square faces (<100>, next nearest, 2 / sqrt(3) = 1.155 times farther). Each face is one relation shared by two cells,
so there are 7 relations (hops) per cell. This is the existing `D_HAT` / `W` (W 1.414 on <111>, 1 on <100>).

## 2. The four element types (fig2_elements.png)

![elements](fig2_elements.png)

| element | per cell | dual (Voronoi) | a whole count here would be | in current code |
|---|---|---|---|---|
| cell (vertex) | 1 | cell volume | quanta in a cell (lattice-gas lineage) | u, lam |
| hop (edge) | 7 = 4 <111> + 3 <100> | face | an exchange across one face | phi, E / d |
| triangle | 12 (2 <111> + 1 <100> edge each) | Voronoi edge | a circulation quantum (loop sum) | B / b (relay, glider) |
| tetrahedron | 6 (4 <111> + 2 <100> edges) | Voronoi vertex | a junction: in = out (div b = 0) is checked here | -- |

1 - 7 + 12 - 6 = 0 (Euler: the counts are consistent).

## 3. The smallest quantized rotation: one count on a hop (fig3_elementary_rings.png)

![elementary rings](fig3_elementary_rings.png)

The triangles around a hop are exactly the edges of its dual face:
- **6 around a <111> hop** (hexagonal face);
- **4 around a <100> hop** (square face).

One unit on a hop (e = 1) gives curl e = b = 1 on each triangle of its fan. Read on the dual, that is a closed ring of
circulation going round the hop, lying in the face. This is the smallest possible closed vortex:
- integer by construction;
- div-free automatically (curl of anything is div-free);
- the only cell-scale "rotation" the lattice has, with exactly two periods: **6 steps** (hex) or **4 steps** (square).

So "a count on a relation" (hop) and "a whole turn of circulation round that relation" (triangle fan) are the same
thing, seen from the two sides. A face exchange IS an elementary rotation.

## 4. A structure-scale ring is a sum of elementary rings (fig4_ring_from_hops.png, computed)

![ring](fig4_ring_from_hops.png)

Put e = 1 on every hop that crosses a flat disk of radius 4.2 (325 hops, both <111> and <100>). Then:
- every interior triangle has two crossing edges with opposite orientation, so b = 0 there;
- b survives only on 108 triangles on the rim, all |b| = 1, at radius 4.01-4.33 (one cell thick);
- the result is one whole turn of circulation round the rim.

This is the zoo's whole-turn ring seed before smoothing. The lattice version of Stokes is exact and integer, and the
spanning surface (the hop counts) is invisible except at its boundary.

Consequence: integer circulation per loop (what the zoo already has) and integer counts on hops (face exchanges) are
two descriptions of one thing. The hops carry the surface, the triangles carry the edge. Moving the structure =
adding or removing elementary face rings at the rim. Smoothing (zoo seeds) spreads b over neighbouring triangles,
making it non-integer per triangle while keeping it an exact whole turn per loop.

## 5. How could it rotate? (fig5_rotation.png)

![rotation](fig5_rotation.png)

Candidates, from smallest scale up:

(a) **A count stepping round a face** (fig 3): period 4 or 6 ticks.
- This is a clock fixed by the tick, the same for every structure.
- A structure-specific frequency (Compton frequency differs between e and p) can't come from this alone.
- If used, it is the floor's own activity (Casimir background?), not the particle's rotation.

(b) **Circulation itself**: the flow going round the core, in whole turns per loop.
- Already held exactly by step() + Cartan (zoo: total circulation changes by at most 3.4 % over T 30 at R 20, energy to 1e-8).
- The rate is set by the circulation over the core size, i.e. by the structure, not by the tick.

(c) **The whole structure precesses** (measured today, nothing imposed):

  | structure | rotation rate |
  |---|---|
  | coil_1_3 | 0.24 deg / t |
  | coil_3_2 | 1.0 deg / t (1.2 before t 9) |
  | rings, pairs, concentric | ~0 |

- Correction (10-19): coil_3_2 (3 toroidal, 2 poloidal, tube 0.35 R) and the zoo "trefoil" (2 toroidal, 3 poloidal,
  tube 0.45 R) are the SAME knot (T(3, 2) = T(2, 3)), yet trefoil turns 0.5 deg by t 23 vs ~25 deg for coil_3_2. So
  the rate is set by the embedding (which winding goes round the core vs the tube, tube radius), not the knot type.
- Steady, rigid rotation set by the two integers (p, q) in order: rotation that is quantized by
  winding numbers and emerges from the transport.
- Removing the rotation, coil_1_3 keeps its shape to +2.8 % (ring +2.4 %).

(d) **Two windings locked** (torus knot (p, q): p toroidal, q poloidal whole turns).
- Your 720 deg question ("one component at twice the other's rate") is the (1, 2) curve: one toroidal turn per two
  poloidal turns.
- The tight-winding Hopf picture (~60 windings round a toroidal core) is (1, ~60) or a bundle of such curves.
- Both integers are topological, so they survive smoothing and are held exactly by Cartan transport.

My read: (b)-(d) are one family (integer windings on a torus), and only they give a structure-scale rotation. (a)
is the floor's clock.

## 6. Floors and the receiver (Philip 10-19), on this geometry

- **Floor:** every hop (and so every elementary face ring) carries a base count n0 > 0. The vacuum is all faces at the
  floor, still exchanging (Casimir: transmission with zero particles). A structure is a departure from the floor.
- **Receiver:** its pull reaches a face as a gradient of the floor / tension along the path built by all scales (two
  planets set a tension path between them). The local transfer rule can then read the receiver's sink strength, so it
  doesn't need to see the receiver directly.
- **Valve candidate on this geometry:** a structure gives up a whole elementary face ring at its rim (one count moved
  across one face) only when its own excess plus the local floor gradient on that face exceeds a threshold.
  - Open: whether that threshold can be structure-scale and not the face floor. The 10-16 zclock failed exactly there.

## 7. To decide

1. **Which element carries the count?** My lean: the hop (face exchange). Its curl is the triangle circulation, so the
   zoo's integer windings come for free. The elementary move "one count across one face" equals adding one elementary
   ring.
2. **Is the 4 / 6-step face clock (a) wanted at all**, e.g. as the floor's activity, or only collective rotation (b-d)?
3. **Floors:** one n0 per hop, the same on hex and square? They differ in length and area (W 1.414 vs 1). That
   ratio is fixed by the geometry, not picked.
4. **Does the (p, q) precession rate matter?** It is the first structure-dependent frequency we have measured. The
   ratio coil_3_2 / coil_1_3 ~ 4.3 at equal R and tube radius; worth a scan in p, q.

## 8. Philip 10-19: the single-cell rotation never happens

The elementary face ring (section 3, candidate (a) in section 5) is too sharp at discrete scale. The background
dictates the future and direction and discourages discrete local occurrences. Consequences:
- Sections 3-4 stay as bookkeeping (lattice Stokes: hop counts <-> triangle circulation), not as events.
- Candidate (a) is out, and so is the valve in section 6 ("give up one elementary face ring at the rim").
- Quantized units must be collective: whole windings of the structure (b-d). The integer appears only in
  structure totals (winding / linking numbers), while the local field stays smooth and context-driven.
- The valve then has to be a whole-winding topology change (reconnection) whose threshold is set by the structure
  plus the background gradient. That is next 10-18 item 2: a multi-winding ring under Cartan -- does it shed only whole
  windings, and does the threshold stay fixed under refinement?

## 9. Philip 10-19: no isolated cell

What happens to one discrete cell is ~99.999999 % what happens to its neighbours. Measured: zoo R 20 end states
(coil_1_3, concentric) have a median neighbour relative contrast of 0.15-0.26 in cell energy (core max ~2), i.e.
>= 7 orders of magnitude coarser than the discrete scale. Our cells are coarse cells, so anything keyed to one
(sim) cell's state is a coarse-cell rule, not a discrete one.

## 10. coil_3_2 and 720 deg (discussion 10-19)

Philip: no spinors at all (no vectors -> no spinors). Spinor / fermion / belt-trick framing is dropped (the belt trick
is SU(2) topology by another name). What remains is the observation (Rauch-Werner 4pi: interference intensity after
a relative precession of one beam path has a 720 deg period), to reproduce from structure + context relations.
- A rigid curve returns after 360 deg (coil_3_2 after 120 deg, 3-fold symmetry); shape alone gives no 720 deg.
- coil_3_2 and the zoo trefoil are the same knot; precession depends on embedding, not knot type (section 5c).
- Flag, not evidence: coil_3_2's 3-fold symmetry + ratio 2 / 3 echo quark-like counts. Numerology unless derived.
- Cheap next step (optional): (p, q) scan at R 20, T 30, to see which embeddings precess and how the rate scales.
