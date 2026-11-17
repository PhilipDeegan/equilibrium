#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "eq/dst/lattice.hpp"

namespace nb = nanobind;
using namespace equilibrium::dst;

namespace {

template<typename T>
nb::ndarray<nb::numpy, T> own1d(std::vector<T> vec) {
    auto* p = new std::vector<T>(std::move(vec));
    nb::capsule cap(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
    std::size_t sh[1] = {p->size()};
    return nb::ndarray<nb::numpy, T>(p->data(), 1, sh, cap);
}

} // namespace

NB_MODULE(dst, m) {
    nb::class_<DSTLattice>(m, "DSTLattice")
        .def(nb::init<double, double, double>(),
             nb::arg("cell_size"), nb::arg("radius_cells"), nb::arg("K") = 1.0)
        .def("set_proton", &DSTLattice::set_proton,
             nb::arg("u_val"), nb::arg("radius_cells") = 0.0)
        .def("set_moat", &DSTLattice::set_moat,
             nb::arg("u_val"), nb::arg("radius_cells"), nb::arg("width_cells") = 1.0)
        .def("set_u", [](DSTLattice& lat,
                         nb::ndarray<nb::numpy, double, nb::ndim<1>> u) {
            lat.set_u(u.data(), u.size());
        }, nb::arg("u"))
        .def("set_perimeter_pressure", &DSTLattice::set_perimeter_pressure,
             nb::arg("u0") = 0.0, nb::arg("shell_width_cells") = 1.0)
        .def("stable_dt", &DSTLattice::stable_dt)
        .def("step", &DSTLattice::step, nb::arg("dt"))
        .def("run", &DSTLattice::run,
             nb::arg("n_steps"), nb::arg("dt"), nb::arg("n_threads") = 1)
        .def("size", &DSTLattice::size)
        .def("u_array", [](DSTLattice& lat) {
            std::size_t shape[1] = {lat.size()};
            return nb::ndarray<nb::numpy, double>(lat.u_data(), 1, shape, nb::handle{});
        })
        // (N, 3) cell positions — valid for the lifetime of the DSTLattice object
        .def("positions", [](DSTLattice const& lat) {
            std::size_t shape[2] = {lat.size(), 3};
            return nb::ndarray<nb::numpy, double const>(lat.pos_data(), 2, shape, nb::handle{});
        })
        // (N, 14) neighbour indices (-1 = boundary) — valid for lifetime of object
        .def("neighbours", [](DSTLattice const& lat) {
            std::size_t shape[2] = {lat.size(), 14};
            return nb::ndarray<nb::numpy, int32_t const>(lat.nb_data(), 2, shape, nb::handle{});
        })
        // (F, 2) int32 array of unique face pairs (i < j); ordering matches solve_lsqr
        .def("face_pairs", [](DSTLattice const& lat) {
            auto* p = new auto(lat.face_pairs());
            nb::capsule cap(p, [](void* q) noexcept {
                delete static_cast<std::vector<std::array<int32_t, 2>>*>(q);
            });
            std::size_t sh[2] = {p->size(), 2};
            return nb::ndarray<nb::numpy, int32_t>(
                p->data()->data(), 2, sh, cap);
        })
        // Nuclear Hamiltonian: harmonic oscillator + configurable face weights
        .def("build_nuclear_hamiltonian", [](DSTLattice const& lat,
                                             double omega, bool use_dst_weights) {
            std::vector<int32_t> rows, cols;
            std::vector<double>  vals;
            lat.build_nuclear_hamiltonian(rows, cols, vals, omega, use_dst_weights);
            return nb::make_tuple(own1d(std::move(rows)),
                                  own1d(std::move(cols)),
                                  own1d(std::move(vals)));
        }, nb::arg("omega") = 1.0, nb::arg("use_dst_weights") = false)
        // Sparse Hamiltonian: returns (rows, cols, vals) as 1-D numpy arrays
        .def("build_hamiltonian", [](DSTLattice const& lat) {
            std::vector<int32_t> rows, cols;
            std::vector<double>  vals;
            lat.build_hamiltonian(rows, cols, vals);
            return nb::make_tuple(own1d(std::move(rows)),
                                  own1d(std::move(cols)),
                                  own1d(std::move(vals)));
        })
        // LSQR winding solver: b_face shape (F,), returns u shape (N,)
        .def("solve_lsqr", [](DSTLattice const& lat,
                               nb::ndarray<nb::numpy, double, nb::ndim<1>> b_face,
                               std::size_t iter_lim) {
            auto u = lat.solve_lsqr(b_face.data(), b_face.size(), iter_lim);
            return own1d(std::move(u));
        }, nb::arg("b_face"), nb::arg("iter_lim") = 50000);
}
