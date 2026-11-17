#pragma once

#include <cstdint>

namespace equilibrium::dst {

struct Cell {
    double  u     = 0.0;
    double  L     = 1.0;
    int32_t lam   = 0;
    bool    fixed = false;
};

} // namespace equilibrium::dst
