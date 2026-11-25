// python: damn::core::ExecutionModel, MPI start/stop, continuous::clusters (equilibrium.lattice)

#include "eq/core/data/lattice/continuous/damn_clusters.hpp"
#include "eq/core/data/lattice/time_stepper.hpp"

#include "damn/core/exec/execution_model.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
namespace ct = equilibrium::dst::continuous;
namespace cs = equilibrium::dst::continuous::clusters;
using damn::core::ExecutionModel;
using equilibrium::dst::discrete::CellSlab;
using equilibrium::dst::discrete::TimeStepper;

namespace {

// a lattice's clusters with the lattice: python keeps both, and exec, alive
struct PyClusters {
  ct::AdaptiveLattice* lat = nullptr;
  cs::Clusters c;
  std::vector<std::pair<std::size_t, std::int32_t>> owned;  // (cluster, global index), in order
  std::vector<cs::Key> keys;

  PyClusters(ct::AdaptiveLattice& l, cs::Clusters&& cl) : lat{&l}, c{std::move(cl)} {
    for (std::size_t i = 0; i < lat->clusters.size(); ++i)
      cs::for_each_owned(c, i, [&](std::int32_t const g, cs::Key const& k) {
        owned.emplace_back(i, g);
        keys.push_back(k);
      });
  }

  auto cell(std::size_t const j) { return lat->clusters[owned[j].first][owned[j].second]; }
};

template <typename T, std::size_t... shape>
auto array(std::size_t const n) {
  std::size_t constexpr per = (shape * ... * 1);
  auto* data = new T[n * per];
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  if constexpr (sizeof...(shape) == 0)
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(data, {n}, owner);
  else
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(data, {n, shape...}, owner);
}

using F64_1D_C = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;

auto owned_field(PyClusters& p, auto&& get) {
  auto a = array<double>(p.owned.size());
  for (std::size_t j = 0; j < p.owned.size(); ++j) a.data()[j] = get(p.cell(j));
  return a;
}

void set_owned_field(PyClusters& p, F64_1D_C const& v, auto&& set) {
  if (v.shape(0) != p.owned.size()) throw std::runtime_error("Clusters: one value per owned site");
  for (std::size_t j = 0; j < p.owned.size(); ++j) {
    auto cell = p.cell(j);
    set(cell, v.data()[j]);
  }
}

}  // namespace

void bind_clusters(nb::module_& top, nb::module_& continuous) {
  nb::class_<ExecutionModel>(
      top, "ExecutionModel",
      "What this rank runs on: MPI ranks (0: none), usable threads (per rank), "
      "the threads in use at once at most max_scheduled times them")
      .def(
          "__init__",
          [](ExecutionModel* t, std::size_t usable_threads, double max_scheduled, bool mpi) {
            new (t)
                ExecutionModel{.usable_threads = usable_threads, .max_scheduled = max_scheduled};
            if (mpi) {
#if EQ_KELVIN_LEVEL_MPI
              damn::mpi::set_ranks(*t);
              if (!t->mpi()) throw std::runtime_error("ExecutionModel: mpi=True before mpi_init()");
#else
              throw std::runtime_error("ExecutionModel: mpi=True in a build without MPI");
#endif
            }
            t->check();
          },
          nb::arg("usable_threads") = 1, nb::arg("max_scheduled") = 1.2, nb::arg("mpi") = false)
      .def_prop_ro("mpi_ranks", [](ExecutionModel const& e) { return e.mpi_ranks; })
      .def_rw("usable_threads", &ExecutionModel::usable_threads)
      .def_rw("max_scheduled", &ExecutionModel::max_scheduled)
      .def_prop_ro("threads_used", [](ExecutionModel const& e) { return e.threads_used.load(); })
      .def_prop_ro("schedulable", &ExecutionModel::schedulable)
      .def("mpi", &ExecutionModel::mpi)
      .def("threaded", &ExecutionModel::threaded)
      .def("mpi_only", &ExecutionModel::mpi_only);

#if EQ_KELVIN_LEVEL_MPI
  top.def("mpi_init", [] {
    if (!damn::mpi::is_init()) {
      int provided = 0;
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    }
  });
  top.def("mpi_finalize", [] {
    int done = 0;
    MPI_Finalized(&done);
    if (damn::mpi::is_init() && !done) MPI_Finalize();
  });
  top.def("mpi_rank", [] { return damn::mpi::is_init() ? damn::mpi::rank() : 0; });
  top.def("mpi_size", [] { return damn::mpi::is_init() ? damn::mpi::size() : 1; });
  top.def(
      "mpi_sum",
      [](double const x) {
        if (!damn::mpi::is_init()) return x;
        double sum = 0;
        MPI_Allreduce(&x, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return sum;
      },
      nb::arg("x"), "x summed over the ranks (collective), x without MPI");
#endif

  nb::class_<PyClusters>(continuous, "Clusters",
                         "Level 0 as one cluster per damn patch (this rank's), ghosts exchanged by "
                         "damn; replaces the lattice's clusters")
      .def_static(
          "periodic",
          [](ct::AdaptiveLattice& lat, std::array<int, 3> P, std::array<int, 3> patch_cells,
             ExecutionModel& exec, double d1, double u0, double lam0) {
            return new PyClusters(lat,
                                  cs::build_periodic(lat, P, exec, {patch_cells, d1, u0, lam0}));
          },
          nb::arg("lattice"), nb::arg("P"), nb::arg("patch_cells"), nb::arg("exec"),
          nb::arg("d1") = 1.0, nb::arg("u0") = 0.0, nb::arg("lam0") = 0.0,
          nb::rv_policy::take_ownership, nb::keep_alive<0, 1>(), nb::keep_alive<0, 4>(),
          "build_bcc_box_periodic(P)'s sites (P multiples of 4)")
      .def_static(
          "box",
          [](ct::AdaptiveLattice& lat, std::array<int, 3> N, std::array<int, 3> patch_cells,
             ExecutionModel& exec, double d1, double u0, double lam0) {
            return new PyClusters(lat, cs::build_box(lat, N, exec, {patch_cells, d1, u0, lam0}));
          },
          nb::arg("lattice"), nb::arg("N"), nb::arg("patch_cells"), nb::arg("exec"),
          nb::arg("d1") = 1.0, nb::arg("u0") = 0.0, nb::arg("lam0") = 0.0,
          nb::rv_policy::take_ownership, nb::keep_alive<0, 1>(), nb::keep_alive<0, 4>(),
          "build_bcc_box's sites, sides 2 N h: keys -N..N")
      .def(
          "step",
          [](PyClusters& p, double dt, std::size_t n_steps) {
            nb::gil_scoped_release release;
            auto const ts = TimeStepper::make_fixed(dt);
            for (std::size_t s = 0; s < n_steps; ++s) cs::step_all(*p.lat, p.c, ts);
          },
          nb::arg("dt"), nb::arg("n_steps") = 1, "collective with MPI")
      .def("exchange_u", [](PyClusters& p) { cs::exchange(p.c, cs::u_names); })
      .def(
          "exchange_lam", [](PyClusters& p) { cs::exchange(p.c, cs::lam_names); },
          "after setting lam: ghosts' from their owners (collective with MPI)")
      .def_prop_ro("rank", [](PyClusters& p) { return p.c.levels.rank; })
      .def_prop_ro("n_clusters", [](PyClusters& p) { return p.lat->clusters.size(); })
      .def_prop_ro("n_owned", [](PyClusters& p) { return p.owned.size(); })
      .def_prop_ro(
          "owned_keys",
          [](PyClusters& p) {
            auto a = array<std::int32_t, 3>(p.keys.size());
            for (std::size_t j = 0; j < p.keys.size(); ++j) {
              a.data()[3 * j] = p.keys[j][0];
              a.data()[3 * j + 1] = p.keys[j][1];
              a.data()[3 * j + 2] = p.keys[j][2];
            }
            return a;
          },
          nb::rv_policy::take_ownership,
          "(n_owned, 3) h-unit keys of this rank's sites (no ghosts), the order of every owned_*")
      .def_prop_ro(
          "owned_u", [](PyClusters& p) { return owned_field(p, [](auto c) { return c.u; }); },
          nb::rv_policy::take_ownership)
      .def_prop_ro(
          "owned_lam", [](PyClusters& p) { return owned_field(p, [](auto c) { return c.lam; }); },
          nb::rv_policy::take_ownership)
      .def_prop_ro(
          "owned_phi",
          [](PyClusters& p) {
            auto a = array<double, CellSlab::NF>(p.owned.size());
            for (std::size_t j = 0; j < p.owned.size(); ++j)
              for (std::size_t k = 0; k < CellSlab::NF; ++k)
                a.data()[j * CellSlab::NF + k] = p.cell(j).phi[k];
            return a;
          },
          nb::rv_policy::take_ownership)
      .def(
          "set_owned_u",
          [](PyClusters& p, F64_1D_C const& v) {
            set_owned_field(p, v, [](auto& c, double x) { c.u = x; });
            cs::exchange(p.c, cs::u_names);
          },
          nb::arg("u"), "this rank's sites' u, then the ghosts' (collective with MPI)")
      .def(
          "set_owned_lam",
          [](PyClusters& p, F64_1D_C const& v) {
            set_owned_field(p, v, [](auto& c, double x) { c.lam = x; });
            cs::exchange(p.c, cs::lam_names);
          },
          nb::arg("lam"), "this rank's sites' lam, then the ghosts' (collective with MPI)");
}
