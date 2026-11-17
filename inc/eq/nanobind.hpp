#ifndef _EQUILIBRIUM_NANOBIND_HPP_
#define _EQUILIBRIUM_NANOBIND_HPP_

#include <nanobind/ndarray.h>
#include <algorithm>
#include <cstddef>

// #include <array>
// #include <vector>

#include "mkn/kul/alloc.hpp"
// #include "mkn/avx/span.hpp"
#include "mkn/avx/lazy.hpp"
#include "mkn/avx.hpp"
#include "mkn/kul/span.hpp"
#include "nanobind/nanobind.h"

namespace eq::nb {

namespace nb = nanobind;

template <typename T, typename... Args>
auto make_span(nb::ndarray<T, Args...>& a) {
  return mkn::kul::Span<T>{a.data(), a.size()};
}

// mkn::avx's SIMD width: 64 bytes on AVX-512 (a hardcoded 32 faulted there on
// the aligned __m512d loads), 32 on AVX2.
inline constexpr auto ALIGN = mkn::avx::Options::ALIGN();

template <typename T>
auto aligned_size(std::size_t const& s, T const& def = 0) {
  auto p = mkn::kul::AlignedAllocator<double, ALIGN>{}.allocate(s);
  nb::capsule owner(p, [](void* p) noexcept { delete[] (float*)p; });
  std::fill_n(p, s, def);
  return nb::ndarray<nb::numpy, double>(p, {s}, owner);
}

template <typename T, typename... Args>
auto aligned_like(nb::ndarray<T, Args...> const& a) {
  auto p = mkn::kul::AlignedAllocator<double, ALIGN>{}.allocate(a.size());
  nb::capsule owner(p, [](void* p) noexcept { delete[] (float*)p; });
  if (a.ndim() == 1) return nb::ndarray<nb::numpy, double>(p, {a.shape(0)}, owner);
  if (a.ndim() == 2) return nb::ndarray<nb::numpy, double>(p, {a.shape(0), a.shape(1)}, owner);
  return nb::ndarray<nb::numpy, double>(p, {a.shape(0), a.shape(1), a.shape(2)}, owner);
}

template <typename T, typename... Args>
auto aligned_like_fill(nb::ndarray<T, Args...> const& a, T const& def) {
  auto ar = aligned_like(a);
  std::fill_n(ar.data(), ar.size(), def);
  return ar;
}

template <typename T, typename... Args>
auto aligned_copy(nb::ndarray<T, Args...> const& a) {
  auto ar = aligned_like(a);
  std::copy(a.data(), a.data() + a.size(), ar.data());
  return ar;
}

// hidden: its nb::ndarray / mkn::avx members are hidden-visibility types
// (-Wattributes "declared with greater visibility than the type of its field")
template <typename T>
struct __attribute__((visibility("hidden"))) LazyNumpyArray {
  using This = LazyNumpyArray<T>;

  LazyNumpyArray(nb::ndarray<double>& a) : arr{a} {}

  auto operator+(This const& that) const {
    lz + that.lz;
    return *this;
  }
  auto operator-(This const& that) const {
    lz - that.lz;
    return *this;
  }

  auto operator*(This const& that) const {
    lz* that.lz;
    return *this;
  }
  auto operator/(This const& that) const {
    lz / that.lz;
    return *this;
  }

  auto size() const { return arr.size(); }

  nb::ndarray<double> arr;
  mkn::kul::Span<T> span = make_span(arr);
  mkn::avx::LazyVal<mkn::kul::Span<T>> lz{span};
};

}  // namespace eq::nb

#endif  // _EQUILIBRIUM_NANOBIND_HPP_
