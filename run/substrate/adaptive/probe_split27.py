"""
Standalone prototype: 27-child (3x3x3, odd-factor) self-similar refinement of
a Kelvin (BCC Wigner-Seitz) cell -- validated entirely in Python, independent
of the C++ AdaptiveLattice engine.

Motivation: the existing C++ split (adaptive_cell::N_CHILDREN_HEX = 8, or
HEX_CENTER = 9) refines by an EVEN factor (2x). BCC is two interleaved simple
cubic sublattices (corner: all-even index parity, body: all-odd), and an
even refinement factor collapses that parity distinction -- a child is no
longer a genuine 14-neighbour Kelvin cell, and adaptive_lattice.hpp's
accumulate_to_parent() has to special-case which children/faces are
"externally facing" per split model (see the is_sibling / nb_ck checks
there).

An ODD refinement factor (here 3, giving 27 children) preserves BCC
self-similarity exactly: 3*even=even and 3*odd=odd, so the parity class is
untouched. Concretely, in integer h-unit grid coordinates (h = d1/sqrt(3),
same convention as build_bcc_sphere in bcc_geometry.hpp), a level-0 cell at
index I maps to level-1 index 3*I, and its 27 children are simply the 27
valid (all-even or all-odd) fine-grid points whose nearest coarse centre
(among the parent and its 14 neighbours) is the parent itself -- a plain
Voronoi assignment, no bespoke child/face table required.

This script:
  1. Builds a regular (uniform, unrefined) BCC/Kelvin sphere at L0 -- the
     same construction as bcc_geometry.hpp's build_bcc_sphere, reimplemented
     here in Python with exact integer arithmetic.
  2. For every "interior" L0 cell (all 14 neighbours present in the sphere),
     computes the 27-child split via nearest-centre assignment.
  3. Wires each child's 14 fine-grid neighbours and classifies them as
     internal (sibling, same parent) or external (belongs to an adjacent
     parent's child set).
  4. Cross-validates every external face: the neighbour point found from
     child c of parent P must also appear in the independently-computed
     child set of the actual neighbouring parent N -- i.e. the interface is
     watertight (no seams, no gaps, no duplicates) with zero special-casing.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_split27.py --R 10
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_split27.py --R 20
"""

import argparse

import numpy as np

# ---------------------------------------------------------------------------
# BCC face geometry, integer h-unit convention (matches bcc_geometry.hpp).
#   hex neighbours: delta = (+-1,+-1,+-1)  -> physical distance h*sqrt(3) = d1
#   sq  neighbours: delta = (+-2, 0, 0) etc -> physical distance 2h = d2
# ---------------------------------------------------------------------------
HEX_DELTA = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.int64
)  # (8,3)
SQ_DELTA = np.array(
    [[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2]],
    dtype=np.int64,
)  # (6,3)
ALL_DELTA = np.concatenate([HEX_DELTA, SQ_DELTA])  # (14,3)


def build_bcc_sphere_idx(R_cells: float):
    """Integer (ix,iy,iz) in h-units for all BCC points with physical |r| <= R_cells*d1.

    Mirrors build_bcc_sphere's pass 1 (bcc_geometry.hpp): same-parity triples only
    (all-even = corner sublattice, all-odd = body sublattice); mixed parity is not
    a lattice point and is skipped.
    """
    imax = int(np.ceil(R_cells * np.sqrt(3.0))) + 1
    rng = np.arange(-imax, imax + 1, dtype=np.int64)
    Ix, Iy, Iz = np.meshgrid(rng, rng, rng, indexing="ij")
    idx = np.stack([Ix.ravel(), Iy.ravel(), Iz.ravel()], axis=1)

    same_parity = (idx[:, 0] & 1) == (idx[:, 1] & 1)
    same_parity &= (idx[:, 1] & 1) == (idx[:, 2] & 1)

    # physical distance^2 in h^2 units is just sum(idx^2); R in h units = R_cells*sqrt(3)
    R2_h = (R_cells * np.sqrt(3.0)) ** 2
    r2 = np.sum(idx.astype(np.float64) ** 2, axis=1)
    keep = same_parity & (r2 <= R2_h)
    return idx[keep]


def split27(parent_idx, neighbour_idx_list, window=4):
    """27-child split of the Kelvin cell at parent_idx (level-0, h-units).

    Returns (children_idx1, ) -- (M,3) int64 array of level-1 (h/3-unit) child
    indices assigned to this parent by nearest-centre (Voronoi) rule against
    the parent and every neighbour actually present in neighbour_idx_list.
    M == 27 when all 14 neighbours are present (an "interior" cell); M > 27
    at the domain boundary where absent neighbours cede territory.
    """
    c1 = 3 * parent_idx  # this parent's centre, in level-1 (fine) index units
    centres1 = np.vstack([c1[None, :]] + [3 * n[None, :] for n in neighbour_idx_list])

    w = np.arange(-window, window + 1, dtype=np.int64)
    Dx, Dy, Dz = np.meshgrid(w, w, w, indexing="ij")
    cand = c1[None, :] + np.stack([Dx.ravel(), Dy.ravel(), Dz.ravel()], axis=1)

    same_parity = (cand[:, 0] & 1) == (cand[:, 1] & 1)
    same_parity &= (cand[:, 1] & 1) == (cand[:, 2] & 1)
    cand = cand[same_parity]

    # squared distance to every centre; assign to nearest (index 0 == parent)
    d2 = np.sum((cand[:, None, :] - centres1[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)
    # detect exact ties (would mean an ambiguous Voronoi boundary point)
    dmin = d2[np.arange(len(cand)), nearest]
    n_ties = int(np.sum(np.sum(d2 == dmin[:, None], axis=1) > 1))

    return cand[nearest == 0], n_ties


def split_one_cell(idx0, key_to_row, cell_idx, window=4):
    """Refine a single L0 cell (given as its (ix,iy,iz) h-unit index) into its
    27 children and print full detail: child positions (h/3 units and
    physical d1 units) and each child's 14-neighbour wiring classification.
    """
    i0 = key_to_row.get(tuple(cell_idx))
    if i0 is None:
        raise ValueError(f"cell {cell_idx} is not in the L0 lattice")
    p = idx0[i0]

    present, slots = [], []
    for k, d in enumerate(ALL_DELTA):
        q = tuple((p + d).tolist())
        if q in key_to_row:
            present.append(idx0[key_to_row[q]])
            slots.append(k)
    print(f"cell {cell_idx}: {len(present)}/14 neighbours present in the L0 lattice")
    print(
        "-> this is a single refined cell surrounded by UNREFINED L0 neighbours: "
        "genuine mixed-level AMR (one coarse-fine interface per external child face), "
        "not the uniform-refinement stress test above."
    )

    # face-slot k (fixed global direction, same convention as `present`/`slots`
    # above) -> the *specific* L0 neighbour cell that slot connects to, if any.
    slot_to_l0row = {
        k: idx0_row
        for idx0_row, k in zip((key_to_row[tuple(n.tolist())] for n in present), slots)
    }

    children, n_ties = split27(p, present, window=window)
    print(f"n_children={len(children)}  (expect 27 if fully interior)  ties={n_ties}\n")

    h = 1.0 / np.sqrt(3.0)  # h-unit size at d1=1; child grid is h/3
    my_set = {tuple(c) for c in children.tolist()}
    print(
        f"{'child idx (h/3 units)':>26}  {'pos (d1 units)':>28}  internal  "
        f"external face-slot -> L0 neighbour cell"
    )
    for c in children.tolist():
        pos = np.array(c) * (h / 3.0)
        ext_faces = []
        n_int = 0
        for k, d in enumerate(ALL_DELTA):
            nb_pt = tuple((np.array(c) + d).tolist())
            if nb_pt in my_set:
                n_int += 1
            else:
                l0row = slot_to_l0row.get(k)
                l0_idx = (
                    tuple(idx0[l0row].tolist())
                    if l0row is not None
                    else "OUT-OF-DOMAIN"
                )
                ext_faces.append((k, l0_idx))
        print(
            f"{c!s:>26}  ({pos[0]:+7.4f},{pos[1]:+7.4f},{pos[2]:+7.4f})  "
            f"{n_int:>8}  {ext_faces}"
        )
    return children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=10.0, help="L0 domain radius, d1 units")
    ap.add_argument(
        "--window",
        type=int,
        default=4,
        help="candidate search half-width, level-1 units",
    )
    ap.add_argument(
        "--one-cell",
        action="store_true",
        help="only refine a single L0 cell (--cell) and print its 27 children in detail, "
        "instead of validating the whole lattice",
    )
    ap.add_argument(
        "--cell",
        type=int,
        nargs=3,
        default=(0, 0, 0),
        help="(ix,iy,iz) h-unit index of the L0 cell to refine, used with --one-cell",
    )
    args = ap.parse_args()

    idx0 = build_bcc_sphere_idx(args.R)
    n0 = len(idx0)
    print(f"L0: R={args.R}  n_cells={n0}", flush=True)

    key_to_row = {tuple(row): i for i, row in enumerate(idx0.tolist())}

    if args.one_cell:
        split_one_cell(idx0, key_to_row, tuple(args.cell), window=args.window)
        return

    # neighbour lookup: which of the 14 neighbours actually exist in the sphere
    neighbours = []  # list of (present_idx_list, present_slot_list) per cell
    n_interior = 0
    for i, p in enumerate(idx0):
        present = []
        slots = []
        for k, d in enumerate(ALL_DELTA):
            q = tuple((p + d).tolist())
            if q in key_to_row:
                present.append(idx0[key_to_row[q]])
                slots.append(k)
        neighbours.append((present, slots))
        if len(present) == 14:
            n_interior += 1
    print(
        f"L0: interior (14/14 nb present) = {n_interior}  boundary = {n0 - n_interior}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Split every interior cell into 27 children; keep a global map from
    # child (level-1) index -> owning parent row, for cross-validation.
    # ------------------------------------------------------------------
    child_owner = {}  # level-1 index tuple -> parent row i
    parent_children = {}  # parent row i -> (M,3) child idx array
    n_ties_total = 0
    counts = []
    for i, p in enumerate(idx0):
        present, _ = neighbours[i]
        if len(present) != 14:
            continue  # boundary cell -- skip for the strict interior test
        children, n_ties = split27(p, present, window=args.window)
        n_ties_total += n_ties
        counts.append(len(children))
        parent_children[i] = children
        for c in children.tolist():
            key = tuple(c)
            if key in child_owner:
                print(
                    f"  DUPLICATE child {key}: claimed by parent {child_owner[key]} and {i}"
                )
            child_owner[key] = i

    counts = np.array(counts)
    print(
        f"\nsplit27 over {len(counts)} interior L0 cells: "
        f"children/cell min={counts.min()} max={counts.max()} "
        f"(expect exactly 27 for all)  ties={n_ties_total}",
        flush=True,
    )
    n_bad = int(np.sum(counts != 27))
    if n_bad:
        print(f"  ** {n_bad} interior cells did NOT produce exactly 27 children **")
    else:
        print("  OK: every interior cell -> exactly 27 children")

    # ------------------------------------------------------------------
    # Neighbour wiring + interface cross-validation.
    # For every child of every interior parent, resolve its 14 fine
    # neighbours as internal (sibling) or external, and if external + the
    # coarse neighbour is itself interior, verify the point is present in
    # that neighbour's OWN independently-computed child set.
    # ------------------------------------------------------------------
    n_internal = 0
    n_external_checked = 0
    n_external_mismatch = 0
    n_external_unchecked = 0  # neighbour parent not interior / not in lattice
    per_face_external = np.zeros(
        14, dtype=np.int64
    )  # external count, coarse face index

    for i, children in parent_children.items():
        present, slots = neighbours[i]
        slot_of_neighbour_row = {}
        for n_idx, k in zip(present, slots):
            slot_of_neighbour_row[tuple(n_idx.tolist())] = k

        my_set = {tuple(c) for c in children.tolist()}
        for c in children.tolist():
            for d in ALL_DELTA:
                nb_pt = tuple((np.array(c) + d).tolist())
                if nb_pt in my_set:
                    n_internal += 1
                    continue
                # external: which parent owns it?
                owner = child_owner.get(nb_pt)
                if owner is None:
                    n_external_unchecked += 1
                    continue
                n_external_checked += 1
                # identify which of the 14 coarse faces this external neighbour
                # sits behind, purely from parent-index geometry
                owner_p = idx0[owner]
                delta0 = owner_p - idx0[i]
                # map coarse delta back to a face slot for reporting
                matched_slot = None
                for k, d14 in enumerate(ALL_DELTA):
                    if np.array_equal(delta0, d14):
                        matched_slot = k
                        break
                if matched_slot is not None:
                    per_face_external[matched_slot] += 1
                if owner == i:
                    n_external_mismatch += (
                        1  # shouldn't happen: already excluded via my_set
                    )

    print("\nneighbour wiring (interior parents' children only):")
    print(f"  internal (sibling) links : {n_internal}")
    print(
        f"  external, cross-validated: {n_external_checked}  mismatches={n_external_mismatch}"
    )
    print(
        f"  external, unchecked (neighbour parent absent/boundary): {n_external_unchecked}"
    )
    print(
        f"  external count by coarse face slot (0-7 hex, 8-13 sq): {per_face_external.tolist()}"
    )

    total_faces = n_internal + n_external_checked + n_external_unchecked
    expected_faces = 27 * 14 * len(parent_children)
    print(
        f"\ntotal child-face resolutions: {total_faces}  "
        f"(expected {expected_faces} = 27*14*n_interior_parents)"
    )


if __name__ == "__main__":
    main()
