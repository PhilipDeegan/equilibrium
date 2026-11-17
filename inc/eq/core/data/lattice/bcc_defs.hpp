#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_BCC_DEFS_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_BCC_DEFS_HPP_

// Pure BCC Kelvin-cell geometry primitives — no lattice dependencies.
// Included by both adaptive_lattice.hpp and bcc_geometry.hpp.

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace equilibrium::dst::discrete {
namespace bcc {

// ---------------------------------------------------------------------------
// Face direction tables
// ---------------------------------------------------------------------------

// hex_dir[k]: (sx,sy,sz) ∈ {-1,+1}³ in lex order (matches kelvin.py)
//   0=(-,-,-)  1=(-,-,+)  2=(-,+,-)  3=(-,+,+)
//   4=(+,-,-)  5=(+,-,+)  6=(+,+,-)  7=(+,+,+)
constexpr std::array<std::array<int8_t, 3>, 8> hex_dir = {{{-1, -1, -1},
                                                           {-1, -1, +1},
                                                           {-1, +1, -1},
                                                           {-1, +1, +1},
                                                           {+1, -1, -1},
                                                           {+1, -1, +1},
                                                           {+1, +1, -1},
                                                           {+1, +1, +1}}};

// sq_dir[k-8]: ±x, ±y, ±z  (faces 8-13)
constexpr std::array<std::array<int8_t, 3>, 6> sq_dir = {
    {{-1, 0, 0}, {+1, 0, 0}, {0, -1, 0}, {0, +1, 0}, {0, 0, -1}, {0, 0, +1}}};

// Face-k delta in integer index units, at ANY refinement level (BCC neighbour
// geometry is scale-invariant: hex_dir/sq_dir describe the same lattice steps
// whether the unit is a coarse h or a fine h/3) — hex k<8: hex_dir[k];
// sq k>=8: 2*sq_dir[k-8] (matches the `2 * d` convention used everywhere else
// nb[] is resolved from an integer grid, e.g. build_bcc_sphere/build_bcc_box).
constexpr std::array<int, 3> face_delta(std::size_t k) noexcept {
  if (k < 8) return {hex_dir[k][0], hex_dir[k][1], hex_dir[k][2]};
  auto const& d = sq_dir[k - 8];
  return {2 * d[0], 2 * d[1], 2 * d[2]};
}

// ---------------------------------------------------------------------------
// Odd27Table — 3x3x3 (ratio-3) self-similar BCC split geometry.
//
// An ODD refinement factor preserves BCC sublattice parity exactly
// (3*even=even, 3*odd=odd — unlike the ratio-2 splits below, which collapse
// the corner/body distinction), so every child is a genuine 14-neighbour
// Kelvin cell. Validated in pure Python (tst/substrate/adaptive/
// probe_split27.py / probe_split27_boundary.py): nearest-centre (Voronoi)
// assignment of fine-grid points against the parent + its 14 neighbour
// centres, first-occurrence-wins on ties (matches numpy argmin) — an
// exhaustive sweep (R=8, 1831 interior cells, 692118 face resolutions) found
// zero duplicate/mismatched assignments. This table ports that exact
// algorithm as a constexpr (compile-time) computation — no runtime geometry
// search — for the canonical case of a fully interior parent (all 14
// neighbours present).
//
// offset[c]      = fine (h/3-unit) integer offset of child c from the
//                  parent's own centre.
// face_target[c][k] = classification of child c's own face k (same 14
//                  face directions, at fine scale):
//                    < 27  → sibling child index (c's face k lands on
//                            another one of the 27 children)
//                    >= 27 → external: proxies through the PARENT's own
//                            face (face_target[c][k] - ODD27_EXTERNAL_BASE),
//                            same "coarse proxy" convention used by the
//                            ratio-2 splits (only child k is geometrically
//                            adjacent to the real fine neighbour, but until
//                            that neighbour is itself split, every
//                            externally-facing child reads the parent's own
//                            coarse neighbour).
// ---------------------------------------------------------------------------
inline constexpr std::uint8_t ODD27_EXTERNAL_BASE = 27;

struct Odd27Table {
  std::array<std::array<int, 3>, 27> offset{};
  std::array<std::array<std::uint8_t, 14>, 27> face_target{};
};

constexpr bool operator==(std::array<int, 3> const& a, std::array<int, 3> const& b) noexcept {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

constexpr int odd27_dist2(std::array<int, 3> const& a, std::array<int, 3> const& b) noexcept {
  int const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

// nearest-centre index (0=self, 1-14=neighbour face k=index-1) for point p,
// among the 15 centres — first-occurrence-wins on ties (matches numpy argmin).
constexpr int odd27_nearest_centre(std::array<int, 3> const& p,
                                   std::array<std::array<int, 3>, 15> const& centres) noexcept {
  int best = 0;
  int best_d2 = odd27_dist2(p, centres[0]);
  for (int ci = 1; ci < 15; ++ci) {
    int const d2 = odd27_dist2(p, centres[static_cast<std::size_t>(ci)]);
    if (d2 < best_d2) {
      best_d2 = d2;
      best = ci;
    }
  }
  return best;
}

constexpr Odd27Table build_odd27_table() {
  std::array<std::array<int, 3>, 15> centres{};
  centres[0] = {0, 0, 0};
  for (std::size_t k = 0; k < 14; ++k) {
    auto const d = face_delta(k);
    centres[1 + k] = {3 * d[0], 3 * d[1], 3 * d[2]};
  }

  Odd27Table t{};
  int n_children = 0;
  for (int x = -4; x <= 4; ++x) {
    for (int y = -4; y <= 4; ++y) {
      for (int z = -4; z <= 4; ++z) {
        if (((x & 1) != (y & 1)) || ((y & 1) != (z & 1))) continue;  // same-parity filter
        std::array<int, 3> const p{x, y, z};
        if (odd27_nearest_centre(p, centres) != 0) continue;
        if (n_children >= 27) throw "odd27: more than 27 children found";
        t.offset[static_cast<std::size_t>(n_children)] = p;
        ++n_children;
      }
    }
  }
  if (n_children != 27) throw "odd27: expected exactly 27 children";

  for (std::size_t c = 0; c < 27; ++c) {
    for (std::size_t k = 0; k < 14; ++k) {
      auto const d = face_delta(k);
      std::array<int, 3> const target{t.offset[c][0] + d[0], t.offset[c][1] + d[1],
                                      t.offset[c][2] + d[2]};

      int sib = -1;
      for (std::size_t j = 0; j < 27; ++j) {
        if (t.offset[j] == target) {
          sib = static_cast<int>(j);
          break;
        }
      }
      if (sib >= 0) {
        t.face_target[c][k] = static_cast<std::uint8_t>(sib);
        continue;
      }

      int const nearest = odd27_nearest_centre(target, centres);
      if (nearest == 0) throw "odd27: unresolved face target (window too small?)";
      t.face_target[c][k] = static_cast<std::uint8_t>(ODD27_EXTERNAL_BASE + (nearest - 1));
    }
  }
  return t;
}

inline constexpr Odd27Table odd27_table = build_odd27_table();

constexpr int odd27_count_internal(Odd27Table const& t) noexcept {
  int n = 0;
  for (std::size_t c = 0; c < 27; ++c)
    for (std::size_t k = 0; k < 14; ++k)
      if (t.face_target[c][k] < ODD27_EXTERNAL_BASE) ++n;
  return n;
}

constexpr int odd27_count_external(Odd27Table const& t) noexcept {
  int n = 0;
  for (std::size_t c = 0; c < 27; ++c)
    for (std::size_t k = 0; k < 14; ++k)
      if (t.face_target[c][k] >= ODD27_EXTERNAL_BASE) ++n;
  return n;
}

// Pins the table to the numbers already verified by the exhaustive Python
// sweep (probe_split27.py --R 8: 358876 internal / 1831 interior parents =
// 196 exactly; 378 - 196 = 182 external) — any future edit that breaks the
// geometry fails the build, not a unit test.
static_assert(odd27_count_internal(odd27_table) == 196,
             "Odd27Table: internal (sibling) face-link count diverged from the "
             "validated Python geometry (probe_split27.py)");
static_assert(odd27_count_external(odd27_table) == 182,
             "Odd27Table: external face-link count diverged from the "
             "validated Python geometry (probe_split27.py)");

// ---------------------------------------------------------------------------
// Child octant table for hex-split (N_CHILDREN_HEX = 8)
//
// child_octant[c] == hex_dir[c]: child c sits in the octant of hex face c.
// ---------------------------------------------------------------------------
constexpr std::array<std::array<int8_t, 3>, 8> child_octant = {{{-1, -1, -1},
                                                                {-1, -1, +1},
                                                                {-1, +1, -1},
                                                                {-1, +1, +1},
                                                                {+1, -1, -1},
                                                                {+1, -1, +1},
                                                                {+1, +1, -1},
                                                                {+1, +1, +1}}};

// ---------------------------------------------------------------------------
// Sibling helpers
// ---------------------------------------------------------------------------

// flip_child(c, j): sibling of child c reached via sq face on axis j (0=x,1=y,2=z).
// Index encoding: bit 2 = x-sign, bit 1 = y-sign, bit 0 = z-sign  (lex order).
// Flipping axis j flips bit (2-j).
constexpr std::uint8_t flip_child(std::uint8_t c, int j) noexcept {
  return static_cast<std::uint8_t>(c ^ (1u << (2 - j)));
}

// sq_face_for_axis[j][side]: sq face index for axis j, side 0=negative, 1=positive.
//   j=0 → x: 8=−x, 9=+x   j=1 → y: 10=−y, 11=+y   j=2 → z: 12=−z, 13=+z
constexpr std::array<std::array<std::uint8_t, 2>, 3> sq_face_for_axis = {{
    {8, 9},    // x
    {10, 11},  // y
    {12, 13}   // z
}};

// children_on_face_interior[k]: bitmask of which children (0..7) lie on the
// interior side of parent face k.  Used by wire_split_children.
constexpr std::array<std::uint8_t, 14> children_on_face_interior = []() {
  std::array<std::uint8_t, 14> m{};
  for (std::size_t k = 0; k < 8; ++k) m[k] = static_cast<std::uint8_t>(1u << k);
  for (std::size_t k = 8; k < 14; ++k) {
    auto& d = sq_dir[k - 8];
    for (std::uint8_t c = 0; c < 8; ++c)
      for (int ax = 0; ax < 3; ++ax)
        if (d[ax] != 0 && child_octant[c][ax] == d[ax])
          m[k] |= static_cast<std::uint8_t>(1u << c);
  }
  return m;
}();

// ---------------------------------------------------------------------------
// Sq plaquette tables (curl / magnetic sector)
// ---------------------------------------------------------------------------

// Number of sq plaquettes per cell (xy, xz, yz).
constexpr std::size_t NP = 3;

// sq_plaquette_faces[p]: the 4 face indices traversed in order around loop p.
// Loop: cell_0 →(k0)→ cell_1 →(k1)→ cell_2 →(k2)→ cell_3 →(k3)→ cell_0
//   p=0 P_xy: A →(8,-x)→ B →(10,-y)→ C →(9,+x)→ D →(11,+y)→ A
//   p=1 P_xz: A →(8,-x)→ B →(12,-z)→ C →(9,+x)→ D →(13,+z)→ A
//   p=2 P_yz: A →(10,-y)→ B →(12,-z)→ C →(11,+y)→ D →(13,+z)→ A
// B_sq[p] has normal axis: p=0→z, p=1→y, p=2→x.
constexpr std::array<std::array<std::uint8_t, 4>, NP> sq_plaquette_faces = {{
    {8, 10, 9, 11},    // P_xy  (B_z)
    {8, 12, 9, 13},    // P_xz  (B_y)
    {10, 12, 11, 13},  // P_yz  (B_x)
}};

// ---------------------------------------------------------------------------
// Position key for hash-map lookup during lattice construction
// ---------------------------------------------------------------------------

struct IntVec3 {
  std::int32_t x, y, z;
  bool operator==(IntVec3 const& o) const noexcept {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct IntVec3Hash {
  std::size_t operator()(IntVec3 const& v) const noexcept {
    std::size_t h = 14695981039346656037ULL;
    auto mix = [&](std::int32_t x) {
      h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(x));
      h *= 1099511628211ULL;
    };
    mix(v.x);
    mix(v.y);
    mix(v.z);
    return h;
  }
};

using PositionMap = std::unordered_map<IntVec3, std::int32_t, IntVec3Hash>;

}  // namespace bcc
}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_BCC_DEFS_HPP_
