#ifndef _EQUILIBRIUM_CORE_MATH_SOLID_ANGLE_HPP_
#define _EQUILIBRIUM_CORE_MATH_SOLID_ANGLE_HPP_

// Solid angle of a closed polygon seen from many points, summed over translated copies of the polygon.
// Each segment (A, B) contributes the signed solid angle of the triangle (apex, A, B) (Van Oosterom-Strackee),
// so the sum is the polygon's solid angle mod 4 pi with the branch cut on the cone from the apex.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

namespace equilibrium::math {

inline void solid_angle(double const* P, std::size_t M, double const* X, std::size_t N, double const* apex,
                        double const* shifts, std::size_t S, double* out, std::size_t n_threads) {
  auto work = [&](std::size_t i0, std::size_t i1) {
    std::vector<double> b(3 * M), lb(M);
    for (std::size_t i = i0; i < i1; ++i) {
      double const x0 = X[3 * i], x1 = X[3 * i + 1], x2 = X[3 * i + 2];
      double const a0 = apex[0] - x0, a1 = apex[1] - x1, a2 = apex[2] - x2;
      double const la = std::sqrt(a0 * a0 + a1 * a1 + a2 * a2);
      double sum = 0;
      for (std::size_t s = 0; s < S; ++s) {
        double const s0 = shifts[3 * s] - x0, s1 = shifts[3 * s + 1] - x1, s2 = shifts[3 * s + 2] - x2;
        for (std::size_t j = 0; j < M; ++j) {
          double const v0 = P[3 * j] + s0, v1 = P[3 * j + 1] + s1, v2 = P[3 * j + 2] + s2;
          b[3 * j] = v0, b[3 * j + 1] = v1, b[3 * j + 2] = v2;
          lb[j] = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
        }
        for (std::size_t j = 0; j < M; ++j) {
          std::size_t const k = j + 1 == M ? 0 : j + 1;
          double const *u = &b[3 * j], *v = &b[3 * k];
          double const c0 = u[1] * v[2] - u[2] * v[1], c1 = u[2] * v[0] - u[0] * v[2], c2 = u[0] * v[1] - u[1] * v[0];
          double const num = a0 * c0 + a1 * c1 + a2 * c2;
          double const den = la * lb[j] * lb[k] + (a0 * u[0] + a1 * u[1] + a2 * u[2]) * lb[k]
                             + (a0 * v[0] + a1 * v[1] + a2 * v[2]) * lb[j]
                             + (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) * la;
          sum += std::atan2(num, den);
        }
      }
      out[i] = 2 * sum;
    }
  };
  n_threads = std::max<std::size_t>(1, std::min(n_threads, N));
  std::size_t const chunk = (N + n_threads - 1) / n_threads;
  std::vector<std::thread> pool;
  for (std::size_t t = 0; t < n_threads; ++t) {
    std::size_t const i0 = t * chunk, i1 = std::min(N, i0 + chunk);
    if (i0 < i1) pool.emplace_back(work, i0, i1);
  }
  for (auto& th : pool) th.join();
}

} // namespace equilibrium::math

#endif
