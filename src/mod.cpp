

#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <vector>

#include "mkn/kul/span.hpp"
#include "mkn/avx/lazy.hpp"

#include "eq/nanobind.hpp"
#include "eq/substrate.hpp"

namespace eq::nb {

namespace nb = nanobind;

template <typename... Args>
auto shape(nb::ndarray<Args...> const& a) {
  std::vector<std::size_t> shape(a.ndim());
  for (size_t i = 0; i < a.ndim(); ++i) shape[i] = a.shape(i);
  return shape;
}

template <typename... Args>
auto shape_equals(nb::ndarray<Args...> const& a, nb::ndarray<Args...> const& b) {
  if (a.ndim() != b.ndim()) return false;
  std::vector<std::size_t> shape(a.ndim());
  for (size_t i = 0; i < a.ndim(); ++i)
    if (a.shape(i) != b.shape(i)) return false;
  return true;
}

template <typename OP, typename T, typename... Args>
auto __op__(nb::ndarray<T, Args...> const& a, nb::ndarray<T, Args...> const& b) {
  if (!shape_equals(a, b)) throw std::runtime_error("cannot compute different shapes");
  auto const s = a.size();
  OP const op;
  auto c = a;
  for (std::size_t i = 0; i < s; ++i) c.data()[i] = op(c.data()[i], b.data()[i]);
  return c;
}
template <typename OP, typename T, typename... Args>
auto& __iop__(nb::ndarray<T, Args...>& a, nb::ndarray<T, Args...> const& b) {
  if (!shape_equals(a, b)) throw std::runtime_error("cannot compute different shapes");
  auto const s = a.size();
  OP const op;
  for (std::size_t i = 0; i < s; ++i) a.data()[i] = op(a.data()[i], b.data()[i]);
  return a;
}

template <typename T, typename... Args>
auto __add__(nb::ndarray<T, Args...> const& a, nb::ndarray<T, Args...> const& b) {
  return __op__<std::plus<T>>(a, b);
}

template <typename T, typename... Args>
auto& __iadd__(nb::ndarray<T, Args...>& a, nb::ndarray<T, Args...> const& b) {
  return __iop__<std::plus<T>>(a, b);
}

NB_MODULE(cpp, m) {
  // Symplectic Euler step for the Kelvin substrate.
  // All arrays must be C-contiguous float64 / int32 / uint8.
  // n_threads=1 disables threading (single-threaded fast path).
  m.def(
      "substrate_step",
      [](nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> u,
         nb::ndarray<double,  nb::ndim<2>, nb::c_contig, nb::device::cpu> phi,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> ell,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> lam,
         nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> fixed_u,
         nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> fixed_phi,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> nb_idx,
         double dt, int n_threads) {
        nb::gil_scoped_release release;
        eq::substrate_step(
            u.data(), phi.data(), ell.data(), lam.data(),
            fixed_u.data(), fixed_phi.data(), nb_idx.data(),
            static_cast<int>(u.size()), dt, 0.0, n_threads);
      },
      nb::arg("u"), nb::arg("phi"), nb::arg("ell"), nb::arg("lam"),
      nb::arg("fixed_u"), nb::arg("fixed_phi"), nb::arg("nb_idx"),
      nb::arg("dt"), nb::arg("n_threads") = 1);

  m.def(
      "substrate_step_lam",
      [](nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> lam,
         nb::ndarray<double,  nb::ndim<2>, nb::c_contig, nb::device::cpu> phi,
         nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> fixed_u,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> nb_idx,
         double dt, double scale, int n_threads) {
        nb::gil_scoped_release release;
        eq::substrate_step_lam(
            lam.data(), phi.data(), fixed_u.data(), nb_idx.data(),
            static_cast<int>(lam.size()), dt, scale, n_threads);
      },
      nb::arg("lam"), nb::arg("phi"), nb::arg("fixed_u"), nb::arg("nb_idx"),
      nb::arg("dt"), nb::arg("scale") = 1.0, nb::arg("n_threads") = 1);

  m.def(
      "substrate_step_cross",
      [](nb::ndarray<double,  nb::ndim<2>, nb::c_contig, nb::device::cpu> phi,
         nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> fixed_phi,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> nb_idx,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> lam,
         double dt, double scale, int n_threads) {
        nb::gil_scoped_release release;
        eq::substrate_step_cross(
            phi.data(), fixed_phi.data(), nb_idx.data(), lam.data(),
            static_cast<int>(phi.shape(0)), dt, scale, n_threads);
      },
      nb::arg("phi"), nb::arg("fixed_phi"), nb::arg("nb_idx"), nb::arg("lam"),
      nb::arg("dt"), nb::arg("scale") = 1.0, nb::arg("n_threads") = 1);

  m.def(
      "substrate_step_cross_barrier",
      [](nb::ndarray<double,  nb::ndim<2>, nb::c_contig, nb::device::cpu> phi,
         nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> fixed_phi,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> nb_idx,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> lam,
         double dt, double scale, double m0, int n_threads) {
        nb::gil_scoped_release release;
        eq::substrate_step_cross_barrier(
            phi.data(), fixed_phi.data(), nb_idx.data(), lam.data(),
            static_cast<int>(phi.shape(0)), dt, scale, m0, n_threads);
      },
      nb::arg("phi"), nb::arg("fixed_phi"), nb::arg("nb_idx"), nb::arg("lam"),
      nb::arg("dt"), nb::arg("scale") = 1.0, nb::arg("m0"), nb::arg("n_threads") = 1);

  // Implicit midpoint with self-sourced lam (see eq::substrate_step_selflam_midpoint). (u0, phi0) read-only,
  // (u1, phi1) written, caller swaps; lam = lam at the last midpoint; um / sig are N-sized scratch.
  // Returns iterations (as self_lam_long.py step_midpoint), raises if not converged.
  m.def(
      "substrate_step_selflam_midpoint",
      [](nb::ndarray<double const, nb::ndim<1>, nb::c_contig, nb::device::cpu> u0,
         nb::ndarray<double const, nb::ndim<2>, nb::c_contig, nb::device::cpu> phi0,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> u1,
         nb::ndarray<double,  nb::ndim<2>, nb::c_contig, nb::device::cpu> phi1,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> lam,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> um,
         nb::ndarray<double,  nb::ndim<1>, nb::c_contig, nb::device::cpu> sig,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> nb_idx,
         double dt, double alpha, double eps, double tol, int max_iter, int n_threads) {
        std::size_t const N = u0.size();
        if (u1.size() != N || lam.size() != N || um.size() != N || sig.size() != N || phi0.shape(0) != N
            || phi1.shape(0) != N || nb_idx.shape(0) != N || phi0.shape(1) != 14 || phi1.shape(1) != 14
            || nb_idx.shape(1) != 14)
          throw std::runtime_error("substrate_step_selflam_midpoint: shape mismatch");
        int it;
        {
          nb::gil_scoped_release release;
          it = eq::substrate_step_selflam_midpoint(
              u0.data(), phi0.data(), u1.data(), phi1.data(), lam.data(), um.data(), sig.data(),
              nb_idx.data(), static_cast<int>(N), dt, alpha, eps, tol, max_iter, n_threads);
        }
        if (it < 0) throw std::runtime_error("substrate_step_selflam_midpoint: did not converge");
        return it;
      },
      nb::arg("u0"), nb::arg("phi0"), nb::arg("u1"), nb::arg("phi1"), nb::arg("lam"), nb::arg("um"),
      nb::arg("sig"), nb::arg("nb_idx"), nb::arg("dt"), nb::arg("alpha"), nb::arg("eps") = 0.0,
      nb::arg("tol") = 1e-13, nb::arg("max_iter") = 200, nb::arg("n_threads") = 1);

  m.def("__iadd__", __iadd__<float>, nb::rv_policy::automatic);
  m.def("__iadd__", __iadd__<double>, nb::rv_policy::automatic);
  m.def("__iadd__", __iadd__<std::uint32_t>);
  m.def("__iadd__", __iadd__<std::uint64_t>);
  m.def("__iadd__", __iadd__<std::int64_t>);
  m.def("__iadd__", __iadd__<std::int32_t>);

  m.def("__add__", __add__<float>, nb::rv_policy::copy);
  m.def("__add__", __add__<double>, nb::rv_policy::copy);
  m.def("__add__", __add__<std::uint32_t>);
  m.def("__add__", __add__<std::uint64_t>);
  m.def("__add__", __add__<std::int64_t>);
  m.def("__add__", __add__<std::int32_t>);

  m.def("make_aligned_like_d", aligned_like<double>);
  m.def("make_aligned_size_d", aligned_size<double>);

  m.def("make_span_d", make_span<double>);

  using Span_d = mkn::kul::Span<double>;
  using LzSpan_d = mkn::avx::LazyVal<Span_d>;
  nb::class_<Span_d>(m, "span_d").def(nb::init<double*, std::size_t>());

  nb::class_<LzSpan_d>(m, "lazy_val_d")
      .def(nb::init<mkn::kul::Span<double>&>())
      .def(nb::self + nb::self)
      .def(nb::self - nb::self)
      .def(nb::self * nb::self)
      .def(nb::self / nb::self);

  nb::class_<LazyNumpyArray<double>>(m, "lazy_np_arr_d")
      .def(nb::init<nb::ndarray<double>&>())
      .def(nb::self + nb::self)
      .def(nb::self - nb::self)
      .def(nb::self * nb::self)
      .def(nb::self / nb::self)
      .def("size", &LazyNumpyArray<double>::size)
      .def_ro("lz", &LazyNumpyArray<double>::lz);

  using LzEval_d = mkn::avx::LazyEvaluator<LzSpan_d>;
  m.def("lazy_eval_d", [](LazyNumpyArray<double>& ar) {
    auto p = aligned_like(ar.arr);
    LzEval_d eve{ar.lz};
    eve(p.data(), true);
    return p;
  });

  nb::class_<LzEval_d>(m, "LzEval_d")
      .def(nb::init<LzSpan_d&>())
      .def("compile", &LzEval_d::compile)
      .def("write_compilable", &LzEval_d::write_compilable);
}

}  // namespace eq::nb
