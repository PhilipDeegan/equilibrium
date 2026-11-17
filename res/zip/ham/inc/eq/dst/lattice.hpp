#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace equilibrium::dst {

class DSTLattice {
public:
    DSTLattice(double cell_size, double radius_cells, double K = 1.0);

    void set_proton(double u_val, double radius_cells = 0.0);
    void set_moat(double u_val, double radius_cells, double width_cells = 1.0);

    // Fix the outermost shell of cells at tension u0 (vacuum reference).
    // Provides the "inward pressure from without" that confines a winding or moat.
    // u0 = 0.0 is the interplanetary vacuum reference (λ=0, no knots).
    // Calibrate K so that perturbations propagate at c in this state — see
    // equilibrium/physical/dst/constants.py for CODATA guides.
    void set_perimeter_pressure(double u0 = 0.0, double shell_width_cells = 1.0);

    double      stable_dt() const;
    void        step(double dt);
    void        run(std::size_t n_steps, double dt, std::size_t n_threads = 1);

    double*     u_data()    { return _u[_cur].data(); }
    std::size_t size()      const { return _N; }
    double      cell_size() const { return _cell_size; }

    // Write u values for all non-fixed cells into both ping-pong buffers.
    // Fixed cells (set by set_proton / set_perimeter_pressure) are left unchanged.
    void set_u(double const* vals, std::size_t n);

    // Raw pointers to (N, 3) and (N, 14) contiguous data
    double const*  pos_data() const { return reinterpret_cast<double const*>(_pos.data()); }
    int32_t const* nb_data()  const { return _nb.empty() ? nullptr : _nb[0].data(); }

    // Sparse Hamiltonian assembly for Bohr-level convergence study (Coulomb, isotropic weights).
    void build_hamiltonian(std::vector<int32_t>& rows,
                           std::vector<int32_t>& cols,
                           std::vector<double>&  vals) const;

    // Nuclear shell Hamiltonian: harmonic oscillator potential, configurable face weights.
    // use_dst_weights=false → isotropic BCC (w1=2/3, w2=1/3), gives magic 2,8,20
    // use_dst_weights=true  → actual DST face geometry (FU_HEX/FU_SQ, ratio 3:1), targets 28,50
    void build_nuclear_hamiltonian(std::vector<int32_t>& rows,
                                   std::vector<int32_t>& cols,
                                   std::vector<double>&  vals,
                                   double omega           = 1.0,
                                   bool   use_dst_weights = false) const;

    // All unique faces (i < j, j >= 0) as a flat list of pairs.
    // Ordering is canonical and matches the b_face argument of solve_lsqr.
    std::vector<std::array<int32_t, 2>> face_pairs() const;

    // LSQR least-squares solve for the winding inverse problem.
    // b_face must have length == face_pairs().size().
    // Returns u solution of length N (mean-zero, max-abs normalised).
    std::vector<double> solve_lsqr(double const* b_face, std::size_t n_faces,
                                   std::size_t iter_lim = 50000) const;

private:
    void _build(double radius_cells);

    std::size_t _N         = 0;
    double      _cell_size = 1.0;
    double      _K         = 1.0;
    int         _cur       = 0;

    std::vector<std::array<double, 3>>   _pos;
    std::vector<double>                  _u[2];
    std::vector<double>                  _L;
    std::vector<int32_t>                 _lam;
    std::vector<bool>                    _fixed;
    std::vector<std::array<int32_t, 14>> _nb;
};

} // namespace equilibrium::dst
