#ifndef _EQUILIBRIUM_CORE_MATH_FINITE_DIFFERENCE_HPP_
#define _EQUILIBRIUM_CORE_MATH_FINITE_DIFFERENCE_HPP_

namespace equilibrium::math::continuous {

auto derivative(auto const& v, auto const fn, auto const diff = 1e-5) {
  return (fn(v + diff) - fn(v - diff)) / (2 * diff);
}

}  // namespace equilibrium::math::continuous

namespace equilibrium::math::discrete {

auto derivative(auto const& v, auto const fn) { return fn(v + 1) - fn(v); }

}  // namespace equilibrium::math::discrete

#endif  // _EQUILIBRIUM_CORE_MATH_FINITE_DIFFERENCE_HPP_
