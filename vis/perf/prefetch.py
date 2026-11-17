import argparse
import itertools

import cppyy

SRC = r"""
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace prefetch_bench {

inline double func(double v) { return std::cos(std::sqrt(v)); }

template <bool PREFETCH, std::size_t GROUP>
double sum(double const* table, std::size_t const* idx, std::size_t size) {
    double result = 0.;
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (PREFETCH) {
            if (i + 1 < size) __builtin_prefetch(table + idx[i + 1], 0, 1);
        }
        for (std::size_t j = 0; j < GROUP; ++j) result += func(table[idx[i] + j]);
    }
    return result;
}

double run(std::size_t size, bool prefetch, bool random, bool group8) {
    std::vector<double> data(size);
    for (auto& d : data) d = 1. * std::rand();
    std::size_t const n = group8 ? size / 8 : size;
    std::size_t const stride = group8 ? 8 : 1;
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = (random ? std::rand() % n : i) * stride;

    auto const start = std::chrono::high_resolution_clock::now();
    double res = 0;
    if (group8) res = prefetch ? sum<true, 8>(data.data(), idx.data(), n) : sum<false, 8>(data.data(), idx.data(), n);
    else        res = prefetch ? sum<true, 1>(data.data(), idx.data(), n) : sum<false, 1>(data.data(), idx.data(), n);
    auto const end = std::chrono::high_resolution_clock::now();
    volatile double sink = res; (void)sink;
    return std::chrono::duration<double>(end - start).count();
}

}
"""

_defined = False


def run(size=1024 * 1024 * 64, prefetch=False, random=True, group8=False):
    global _defined
    if not _defined:
        cppyy.cppdef(SRC)
        _defined = True
    return cppyy.gbl.prefetch_bench.run(size, prefetch, random, group8)


def sweep(size=1024 * 1024 * 64):
    rows = []
    for group8, random, prefetch in itertools.product((False, True), repeat=3):
        rows.append((group8, random, prefetch, run(size, prefetch, random, group8)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2-size", type=int, default=26)
    a = ap.parse_args(argv)
    print(f"{'group8':>6} {'random':>6} {'prefetch':>8} {'seconds':>8}")
    for g, r, p, t in sweep(1 << a.log2_size):
        print(f"{g:>6} {r:>6} {p:>8} {t:8.3f}")


if __name__ == "__main__":
    main()
