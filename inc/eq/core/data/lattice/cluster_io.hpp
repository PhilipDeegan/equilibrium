#ifndef EQUILIBRIUM_CORE_DATA_LATTICE_CLUSTER_IO_HPP
#define EQUILIBRIUM_CORE_DATA_LATTICE_CLUSTER_IO_HPP

#include "eq/core/data/lattice/bc.hpp"
#include "eq/core/data/lattice/cluster.hpp"
#include "eq/core/phlop.hpp"
#include "highfive/H5File.hpp"
#include "eq/core/data/lattice/dst_info.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <cstddef>
#include <memory>
#include <utility>

namespace equilibrium::dst::discrete {

// ---------------------------------------------------------------------------
// save_cluster / load_cluster — HDF5 checkpoint via HighFive.
//
// HDF5 layout under `grp`:
//   Attributes : n_slabs, bc_type, bc_target, bc_weight
//   Datasets   : slab_levels, leaf_indices,
//                u, L, lam, fixed_u, fixed_phi,
//                n_children, parent, first_child,
//                phi, nb, pos
//   Group info : split_hi, split_lo, max_level
//
// Array flat layout (T = any scalar field):
//   T[s * CAPACITY + slot]              for scalar-per-cell fields
//   T[k * n_slabs * CAPACITY + s * C + slot]  for face/coord fields (phi, nb, pos)
// ---------------------------------------------------------------------------

inline void save_cluster(AdaptiveCellCluster const& cl, HighFive::Group& grp) {
  EQ_SCOPE_TIMER("equilibrium::dst::discrete::save_cluster");

  constexpr std::size_t C = CellSlab::CAPACITY;
  constexpr std::size_t NF = CellSlab::NF;
  std::size_t NS = cl.n_slabs();
  std::size_t total = NS * C;

  // --- slab-level map ---
  std::vector<std::uint8_t> slab_lv(NS);
  for (std::size_t s = 0; s < NS; ++s) slab_lv[s] = cl.slab_level(s);

  // --- write ---
  grp.createAttribute("n_slabs", NS);
  grp.createDataSet("slab_levels", slab_lv);
  grp.createDataSet("leaf_indices", cl.leaf_indices);

  // Create every dataset at its final size up front, then stream each slab's
  // own storage straight into its hyperslab — no O(n_slabs*CAPACITY)
  // intermediate buffer (the old flattened-vector approach held the whole
  // cluster twice at once: once in cl's own slabs, once in the flat vectors).
  auto ds_u   = grp.createDataSet<double>("u", HighFive::DataSpace({total}));
  auto ds_L   = grp.createDataSet<double>("L", HighFive::DataSpace({total}));
  auto ds_lam = grp.createDataSet<double>("lam", HighFive::DataSpace({total}));
  auto ds_fu  = grp.createDataSet<std::uint8_t>("fixed_u", HighFive::DataSpace({total}));
  auto ds_fp  = grp.createDataSet<std::uint16_t>("fixed_phi", HighFive::DataSpace({total}));
  auto ds_nc  = grp.createDataSet<std::uint8_t>("n_children", HighFive::DataSpace({total}));
  auto ds_par = grp.createDataSet<std::int32_t>("parent", HighFive::DataSpace({total}));
  auto ds_fc  = grp.createDataSet<std::int32_t>("first_child", HighFive::DataSpace({total}));
  auto ds_phi = grp.createDataSet<double>("phi", HighFive::DataSpace({NF * total}));
  auto ds_nb  = grp.createDataSet<std::int32_t>("nb", HighFive::DataSpace({NF * total}));
  auto ds_pos = grp.createDataSet<double>("pos", HighFive::DataSpace({3 * total}));

  for (std::size_t s = 0; s < NS; ++s) {
    auto const& slab = cl.slab(s);
    std::size_t off = s * C;
    ds_u.select({off}, {C}).write_raw(slab.u);
    ds_L.select({off}, {C}).write_raw(slab.L);
    ds_lam.select({off}, {C}).write_raw(slab.lam);
    ds_fu.select({off}, {C}).write_raw(slab.fixed_u);
    ds_fp.select({off}, {C}).write_raw(slab.fixed_phi);
    ds_nc.select({off}, {C}).write_raw(slab.n_children);
    ds_par.select({off}, {C}).write_raw(slab.parent);
    ds_fc.select({off}, {C}).write_raw(slab.first_child);
    for (std::size_t k = 0; k < NF; ++k) {
      std::size_t koff = (k * NS + s) * C;
      ds_phi.select({koff}, {C}).write_raw(slab.phi + k * C);
      ds_nb.select({koff}, {C}).write_raw(slab.nb + k * C);
    }
    for (std::size_t d = 0; d < 3; ++d)
      ds_pos.select({(d * NS + s) * C}, {C}).write_raw(slab.pos + d * C);
  }

  // --- DSTInfo ---
  auto ig = grp.createGroup("info");
  ig.createAttribute("split_hi", cl.info.split_hi);
  ig.createAttribute("split_lo", cl.info.split_lo);
  ig.createAttribute("max_level", static_cast<int>(cl.info.max_level));

  // --- BC (fn-based only; face_ghost not yet serialised) ---
  int bc_type = 0;
  double bc_target = 0.0, bc_weight = 0.5;
  if (cl.bc) {
    if (cl.bc->fn == BoundaryCondition::absorbing)
      bc_type = 1;
    else if (cl.bc->fn == BoundaryCondition::dirichlet)
      bc_type = 2;
    else if (cl.bc->fn == BoundaryCondition::sponge)
      bc_type = 3;
    bc_target = cl.bc->target;
    bc_weight = cl.bc->weight;
  }
  auto bg = grp.createGroup("bc");
  bg.createAttribute("type", bc_type);
  bg.createAttribute("target", bc_target);
  bg.createAttribute("weight", bc_weight);
}

inline AdaptiveCellCluster load_cluster(HighFive::Group const& grp) {
  constexpr std::size_t C = CellSlab::CAPACITY;
  constexpr std::size_t NF = CellSlab::NF;

  std::size_t NS = 0;
  grp.getAttribute("n_slabs").read(NS);

  std::vector<std::uint8_t> slab_lv;
  std::vector<std::int32_t> leaf_idx;
  grp.getDataSet("slab_levels").read(slab_lv);
  grp.getDataSet("leaf_indices").read(leaf_idx);

  // --- DSTInfo ---
  DSTInfo info;
  auto ig = grp.getGroup("info");
  ig.getAttribute("split_hi").read(info.split_hi);
  ig.getAttribute("split_lo").read(info.split_lo);
  int ml = 0;
  ig.getAttribute("max_level").read(ml);
  info.max_level = static_cast<std::uint8_t>(ml);

  AdaptiveCellCluster cl;
  if (NS == 0) {
    cl.restore_finalize(std::move(leaf_idx), std::move(info));
  } else {
    // Stream each slab's fields straight off disk into the slab's own
    // storage — no O(n_slabs*CAPACITY) intermediate buffer (the old
    // flattened-vector approach held the whole cluster twice at once: once
    // read off disk into flat vectors, once copied into cl's real slabs).
    auto ds_u   = grp.getDataSet("u");
    auto ds_L   = grp.getDataSet("L");
    auto ds_lam = grp.getDataSet("lam");
    auto ds_fu  = grp.getDataSet("fixed_u");
    auto ds_fp  = grp.getDataSet("fixed_phi");
    auto ds_nc  = grp.getDataSet("n_children");
    auto ds_par = grp.getDataSet("parent");
    auto ds_fc  = grp.getDataSet("first_child");
    auto ds_phi = grp.getDataSet("phi");
    auto ds_nb  = grp.getDataSet("nb");
    auto ds_pos = grp.getDataSet("pos");

    for (std::size_t s = 0; s < NS; ++s) {
      auto& slab = cl.restore_add_slab(slab_lv[s]);
      std::size_t off = s * C;
      ds_u.select({off}, {C}).read_raw(slab.u);
      ds_L.select({off}, {C}).read_raw(slab.L);
      ds_lam.select({off}, {C}).read_raw(slab.lam);
      ds_fu.select({off}, {C}).read_raw(slab.fixed_u);
      ds_fp.select({off}, {C}).read_raw(slab.fixed_phi);
      ds_nc.select({off}, {C}).read_raw(slab.n_children);
      ds_par.select({off}, {C}).read_raw(slab.parent);
      ds_fc.select({off}, {C}).read_raw(slab.first_child);
      for (std::size_t k = 0; k < NF; ++k) {
        std::size_t koff = (k * NS + s) * C;
        ds_phi.select({koff}, {C}).read_raw(slab.phi + k * C);
        ds_nb.select({koff}, {C}).read_raw(slab.nb + k * C);
      }
      for (std::size_t d = 0; d < 3; ++d)
        ds_pos.select({(d * NS + s) * C}, {C}).read_raw(slab.pos + d * C);

      cl.restore_finalize_slab(s);
    }

    cl.restore_finalize(std::move(leaf_idx), std::move(info));
  }

  // --- BC ---
  auto bg = grp.getGroup("bc");
  int bc_type = 0;
  double bc_target = 0.0, bc_weight = 0.5;
  bg.getAttribute("type").read(bc_type);
  bg.getAttribute("target").read(bc_target);
  bg.getAttribute("weight").read(bc_weight);
  if (bc_type != 0) {
    auto bc = std::make_unique<BoundaryCondition>();
    switch (bc_type) {
      case 1:
        bc->fn = BoundaryCondition::absorbing;
        bc->fn_type = BCType::ABSORBING;
        break;
      case 2:
        bc->fn = BoundaryCondition::dirichlet;
        bc->fn_type = BCType::FUNCTIONAL;
        break;
      case 3:
        bc->fn = BoundaryCondition::sponge;
        bc->fn_type = BCType::FUNCTIONAL;
        break;
    }
    bc->target = bc_target;
    bc->weight = bc_weight;
    cl.bc = std::move(bc);
  }

  return cl;
}

}  // namespace equilibrium::dst::discrete

#endif  // EQUILIBRIUM_CORE_DATA_LATTICE_CLUSTER_IO_HPP
