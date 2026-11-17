#ifndef EQUILIBRIUM_CORE_DEF_PHLOP_HPP
#define EQUILIBRIUM_CORE_DEF_PHLOP_HPP

#if __has_include("phlop/timing/mpi_scope_timer.hpp") and \
                  (!defined(EQUILIBRIUM_HAVE_PHLOP) || EQUILIBRIUM_HAVE_PHLOP == 1)

#if __has_include("mpi.h")
#include "phlop/timing/mpi_scope_timer.hpp"  // IWYU pragma: keep
#else
#include "phlop/timing/scope_timer.hpp"  // IWYU pragma: keep
#endif

#define EQUILIBRIUM_HAVE_PHLOP 1
#define EQUILIBRIUM_WITH_PHLOP(...) __VA_ARGS__
#define EQUILIBRIUM_WITHOUT_PHLOP(...)
#define EQ_SCOPE_TIMER PHLOP_SCOPE_TIMER
#define EQ_SCOPE PHLOP_SCOPE

#else

#define EQUILIBRIUM_HAVE_PHLOP 0
#define EQUILIBRIUM_WITH_PHLOP(...)
#define EQUILIBRIUM_WITHOUT_PHLOP(...) __VA_ARGS__
#define EQ_SCOPE_TIMER(...)
#define EQ_SCOPE(...)

#endif  // __has_include(...)

#endif /* EQUILIBRIUM_CORE_DEF_PHLOP_HPP */
