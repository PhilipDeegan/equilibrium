#ifndef EQUILIBRIUM_CORE_DATA_LATTICE_DST_INFO_HPP
#define EQUILIBRIUM_CORE_DATA_LATTICE_DST_INFO_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "dict.hpp"

namespace equilibrium::dst::discrete
{

// Initialisation callable: receives x/y/z coordinate arrays, returns field values.
// Set once from Python before the first step.
using InitFn = std::function<std::vector<double>(std::vector<double> const&,
                                                  std::vector<double> const&,
                                                  std::vector<double> const&)>;

// Typed dict accepted by DSTInfo::FROM.
using DSTDict = cppdict::Dict<bool, int, double, std::string, InitFn>;


struct DSTInfo
{
    // --- per-step physics ---
    double       split_hi       = 0.0;  // criterion_max > split_hi  →  refine
    double       split_lo       = 0.0;  // criterion_max < split_lo  →  coarsen
    std::uint8_t max_level      = 0;
    std::uint32_t split_interval = 1;   // adapt() only does work every Nth call (1 = every call)

    // --- initialisation callables (called once, set from Python) ---
    std::optional<InitFn> u_init;

    static DSTInfo FROM(DSTDict const& dict)
    {
        return {
            cppdict::get_value(dict, "split_hi",  0.0),
            cppdict::get_value(dict, "split_lo",  0.0),
            static_cast<std::uint8_t>(cppdict::get_value(dict, "max_level", 0)),
            static_cast<std::uint32_t>(cppdict::get_value(dict, "split_interval", 1)),
            dict.contains("u_init")
                ? std::optional<InitFn>{dict["u_init"].template to<InitFn>()}
                : std::nullopt,
        };
    }
};

} // namespace equilibrium::dst::discrete

#endif // EQUILIBRIUM_CORE_DATA_LATTICE_DST_INFO_HPP
