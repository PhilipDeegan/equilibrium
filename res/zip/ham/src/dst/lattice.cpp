#include "eq/dst/lattice.hpp"

#include <algorithm>
#include <barrier>
#include <cmath>
#include <thread>
#include <unordered_map>

namespace equilibrium::dst {

namespace {

// BCC unit-lattice constants (d1 = hex-face distance = 1)
//   a = 2/sqrt(3)  — BCC conventional cell side
//   h = a/2        — body-diagonal half-step
static const double A = 2.0 / std::sqrt(3.0);
static const double H = A / 2.0;

// 14 neighbour offsets in unit-lattice coordinates:
//   [0..7]  hex faces:  (±H, ±H, ±H)  — 8 body-diagonal directions
//   [8..13] sq  faces:  (±A, 0, 0), (0, ±A, 0), (0, 0, ±A)
static const std::array<std::array<double, 3>, 14> OFFSETS = [] {
    std::array<std::array<double, 3>, 14> o{};
    int k = 0;
    for (int sx : {-1, 1})
        for (int sy : {-1, 1})
            for (int sz : {-1, 1})
                o[k++] = {sx * H, sy * H, sz * H};
    o[8]  = { A,  0,  0};  o[9]  = {-A,  0,  0};
    o[10] = { 0,  A,  0};  o[11] = { 0, -A,  0};
    o[12] = { 0,  0,  A};  o[13] = { 0,  0, -A};
    return o;
}();

// face_unit[k] = A_face[k] / D[k] at unit L=1
//   hex: sqrt(3)/4 / 1.0      = sqrt(3)/4   ≈ 0.4330
//   sq : (1/6) / (2/sqrt(3))  = sqrt(3)/12  ≈ 0.1443
static const double FU_HEX = std::sqrt(3.0) / 4.0;
static const double FU_SQ  = std::sqrt(3.0) / 12.0;

static const std::array<double, 14> FACE_UNIT = [] {
    std::array<double, 14> f{};
    for (int k = 0; k <  8; ++k) f[k] = FU_HEX;
    for (int k = 8; k < 14; ++k) f[k] = FU_SQ;
    return f;
}();

// Integer key for neighbour lookup: scale unit-lattice coords by 2/A = sqrt(3),
// round to int32. Maps corner atoms to even integers, body-centres to odd.
using Key = std::array<int32_t, 3>;

Key make_key(std::array<double, 3> const& p) {
    double s = 2.0 / A;
    return {int32_t(std::lround(p[0] * s)),
            int32_t(std::lround(p[1] * s)),
            int32_t(std::lround(p[2] * s))};
}

struct KeyHash {
    std::size_t operator()(Key const& k) const noexcept {
        std::size_t h = 17;
        h = h * 31 + std::hash<int32_t>{}(k[0]);
        h = h * 31 + std::hash<int32_t>{}(k[1]);
        h = h * 31 + std::hash<int32_t>{}(k[2]);
        return h;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DSTLattice::DSTLattice(double cell_size, double radius_cells, double K)
    : _cell_size{cell_size}, _K{K}
{
    _build(radius_cells);
}

void DSTLattice::_build(double radius_cells) {
    double L     = _cell_size;
    double a     = A * L;
    double r_max = radius_cells * L;
    double r2    = r_max * r_max;

    std::vector<std::array<double, 3>> pts;
    int n = int(std::ceil(r_max / a)) + 1;
    for (int i = -n; i <= n; ++i)
    for (int j = -n; j <= n; ++j)
    for (int k = -n; k <= n; ++k) {
        std::array<double, 3> pc = {i * a, j * a, k * a};
        if (pc[0]*pc[0] + pc[1]*pc[1] + pc[2]*pc[2] <= r2)
            pts.push_back(pc);

        std::array<double, 3> pb = {(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a};
        if (pb[0]*pb[0] + pb[1]*pb[1] + pb[2]*pb[2] <= r2)
            pts.push_back(pb);
    }

    _N   = pts.size();
    _pos = pts;

    std::unordered_map<Key, int32_t, KeyHash> idx_map;
    idx_map.reserve(_N);
    for (std::size_t i = 0; i < _N; ++i) {
        std::array<double, 3> unit = {pts[i][0] / L, pts[i][1] / L, pts[i][2] / L};
        idx_map[make_key(unit)] = int32_t(i);
    }

    _nb.resize(_N);
    for (std::size_t i = 0; i < _N; ++i) {
        std::array<double, 3> unit = {pts[i][0] / L, pts[i][1] / L, pts[i][2] / L};
        for (int k = 0; k < 14; ++k) {
            std::array<double, 3> nb_unit = {unit[0] + OFFSETS[k][0],
                                             unit[1] + OFFSETS[k][1],
                                             unit[2] + OFFSETS[k][2]};
            auto it       = idx_map.find(make_key(nb_unit));
            _nb[i][k]     = (it != idx_map.end()) ? it->second : int32_t(-1);
        }
    }

    _u[0].assign(_N, 0.0);
    _u[1].assign(_N, 0.0);
    _L.assign(_N, L);
    _lam.assign(_N, 0);
    _fixed.assign(_N, false);
    _cur = 0;
}

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

void DSTLattice::set_proton(double u_val, double radius_cells) {
    double r_max  = radius_cells * _cell_size;
    double min_r2 = 1e300;
    int    min_i  = 0;
    bool   any    = false;

    for (std::size_t i = 0; i < _N; ++i) {
        auto& p  = _pos[i];
        double r2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        if (r2 < min_r2) { min_r2 = r2; min_i = int(i); }
        if (std::sqrt(r2) <= r_max) {
            _u[0][i] = _u[1][i] = u_val;
            _fixed[i] = true;
            any = true;
        }
    }
    if (!any) {
        _u[0][min_i] = _u[1][min_i] = u_val;
        _fixed[min_i] = true;
    }
}

void DSTLattice::set_u(double const* vals, std::size_t n) {
    if (n != _N) throw std::runtime_error("set_u: size mismatch");
    for (std::size_t i = 0; i < _N; ++i) {
        if (!_fixed[i]) {
            _u[0][i] = vals[i];
            _u[1][i] = vals[i];
        }
    }
}

void DSTLattice::set_perimeter_pressure(double u0, double shell_width_cells) {
    double r_max = 0.0;
    for (auto const& p : _pos) {
        double r = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        if (r > r_max) r_max = r;
    }
    double r_inner = r_max - shell_width_cells * _cell_size;
    for (std::size_t i = 0; i < _N; ++i) {
        auto const& p = _pos[i];
        double r = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        if (r >= r_inner) {
            _u[0][i] = _u[1][i] = u0;
            _fixed[i] = true;
        }
    }
}

void DSTLattice::set_moat(double u_val, double radius_cells, double width_cells) {
    double L    = _cell_size;
    double r_lo = (radius_cells - width_cells * 0.5) * L;
    double r_hi = (radius_cells + width_cells * 0.5) * L;
    for (std::size_t i = 0; i < _N; ++i) {
        auto& p = _pos[i];
        double r = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        if (r >= r_lo && r <= r_hi)
            _u[0][i] = _u[1][i] = u_val;
    }
}

// ---------------------------------------------------------------------------
// Time evolution
// ---------------------------------------------------------------------------

double DSTLattice::stable_dt() const {
    double max_sigma = 0.0;
    for (std::size_t i = 0; i < _N; ++i) {
        double s = _L[i] / (1.0 + _lam[i]);
        if (s > max_sigma) max_sigma = s;
    }
    double total_geom = 8.0 * FU_HEX + 6.0 * FU_SQ;
    return 0.4 / (_K * max_sigma * total_geom);
}

void DSTLattice::step(double dt) {
    auto const& src = _u[_cur];
    auto&       dst = _u[1 - _cur];
    for (std::size_t i = 0; i < _N; ++i) {
        if (_fixed[i]) { dst[i] = src[i]; continue; }
        double sigma = _L[i] / (1.0 + _lam[i]);
        double du    = 0.0;
        for (int k = 0; k < 14; ++k) {
            int32_t j  = _nb[i][k];
            double  uj = (j >= 0) ? src[j] : src[i];
            du += FACE_UNIT[k] * (uj - src[i]);
        }
        dst[i] = src[i] + dt * _K * sigma * du;
    }
    _cur ^= 1;
}

void DSTLattice::run(std::size_t n_steps, double dt, std::size_t n_threads) {
    if (n_threads <= 1) {
        for (std::size_t s = 0; s < n_steps; ++s) step(dt);
        return;
    }

    std::size_t chunk = (_N + n_threads - 1) / n_threads;

    auto completion = [this]() noexcept { _cur ^= 1; };
    std::barrier sync{std::ptrdiff_t(n_threads), completion};

    auto worker = [&](std::size_t start, std::size_t end) {
        for (std::size_t s = 0; s < n_steps; ++s) {
            auto const& src = _u[_cur];
            auto&       dst = _u[1 - _cur];
            for (std::size_t i = start; i < end; ++i) {
                if (_fixed[i]) { dst[i] = src[i]; continue; }
                double sigma = _L[i] / (1.0 + _lam[i]);
                double du    = 0.0;
                for (int k = 0; k < 14; ++k) {
                    int32_t j  = _nb[i][k];
                    double  uj = (j >= 0) ? src[j] : src[i];
                    du += FACE_UNIT[k] * (uj - src[i]);
                }
                dst[i] = src[i] + dt * _K * sigma * du;
            }
            sync.arrive_and_wait();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (std::size_t t = 0; t < n_threads; ++t) {
        std::size_t start = t * chunk;
        std::size_t end   = std::min(start + chunk, _N);
        threads.emplace_back(worker, start, end);
    }
    for (auto& t : threads) t.join();
}

// ---------------------------------------------------------------------------
// Hamiltonian assembly
// ---------------------------------------------------------------------------

void DSTLattice::build_hamiltonian(std::vector<int32_t>& rows,
                                    std::vector<int32_t>& cols,
                                    std::vector<double>&  vals) const {
    // BCC Laplacian isotropic weights: hex k<8 → 2/3, sq k>=8 → 1/3
    static constexpr double W[14] = {
        2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3,
        1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3
    };
    // t = 1 / (2 a²),  a = A * cell_size,  a² = 4/3 * cell_size²
    double a2 = _cell_size * _cell_size * (4.0 / 3.0);
    double t  = 0.5 / a2;
    double r_min = _cell_size * 0.5;

    rows.clear(); cols.clear(); vals.clear();
    rows.reserve(_N * 15);
    cols.reserve(_N * 15);
    vals.reserve(_N * 15);

    for (std::size_t i = 0; i < _N; ++i) {
        auto const& p = _pos[i];
        double r = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        double diag = -1.0 / std::max(r, r_min);   // Coulomb V_i

        for (int k = 0; k < 14; ++k) {
            int32_t j = _nb[i][k];
            if (j < 0) continue;     // Dirichlet BC: ψ = 0 outside sphere
            double hop = t * W[k];
            rows.push_back(int32_t(i));
            cols.push_back(j);
            vals.push_back(-hop);
            diag += hop;
        }
        rows.push_back(int32_t(i));
        cols.push_back(int32_t(i));
        vals.push_back(diag);
    }
}

// ---------------------------------------------------------------------------
// Nuclear Hamiltonian (harmonic oscillator + configurable face weights)
// ---------------------------------------------------------------------------

void DSTLattice::build_nuclear_hamiltonian(std::vector<int32_t>& rows,
                                            std::vector<int32_t>& cols,
                                            std::vector<double>&  vals,
                                            double omega,
                                            bool   use_dst_weights) const {
    // Isotropic BCC weights (w1=2/3, w2=1/3) or actual DST face geometry
    static constexpr double W_ISO[14] = {
        2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3,
        1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3
    };
    // DST: A/d per face type at unit L=1
    static const double W_DST[14] = {
        FU_HEX, FU_HEX, FU_HEX, FU_HEX, FU_HEX, FU_HEX, FU_HEX, FU_HEX,
        FU_SQ,  FU_SQ,  FU_SQ,  FU_SQ,  FU_SQ,  FU_SQ
    };
    double const* W = use_dst_weights ? W_DST : W_ISO;

    double a2 = _cell_size * _cell_size * (4.0 / 3.0);
    double t  = 0.5 / a2;
    double w2 = omega * omega;

    rows.clear(); cols.clear(); vals.clear();
    rows.reserve(_N * 15);
    cols.reserve(_N * 15);
    vals.reserve(_N * 15);

    for (std::size_t i = 0; i < _N; ++i) {
        auto const& p = _pos[i];
        double r2   = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        double diag = 0.5 * w2 * r2;   // harmonic oscillator V = ½ω²r²

        for (int k = 0; k < 14; ++k) {
            int32_t j = _nb[i][k];
            if (j < 0) continue;
            double hop = t * W[k];
            rows.push_back(int32_t(i));
            cols.push_back(j);
            vals.push_back(-hop);
            diag += hop;
        }
        rows.push_back(int32_t(i));
        cols.push_back(int32_t(i));
        vals.push_back(diag);
    }
}

// ---------------------------------------------------------------------------
// Face list
// ---------------------------------------------------------------------------

std::vector<std::array<int32_t, 2>> DSTLattice::face_pairs() const {
    std::vector<std::array<int32_t, 2>> faces;
    faces.reserve(_N * 7);
    for (std::size_t i = 0; i < _N; ++i)
        for (int k = 0; k < 14; ++k) {
            int32_t j = _nb[i][k];
            if (j > int32_t(i))
                faces.push_back({int32_t(i), j});
        }
    return faces;
}

// ---------------------------------------------------------------------------
// LSQR solver
// ---------------------------------------------------------------------------

std::vector<double> DSTLattice::solve_lsqr(double const* b_face,
                                             std::size_t   /*n_faces*/,
                                             std::size_t   iter_lim) const {
    auto faces = face_pairs();
    std::size_t F = faces.size();
    std::size_t N = _N;

    // G*v  → for each face (i,j): out[k] = v[j] - v[i]
    // G^T*w → for each face (i,j): out[i] -= w[k], out[j] += w[k]
    auto Av = [&](std::vector<double> const& v, std::vector<double>& out) {
        for (std::size_t k = 0; k < F; ++k) {
            int32_t i = faces[k][0], j = faces[k][1];
            out[k] = v[j] - v[i];
        }
    };
    auto ATw = [&](std::vector<double> const& w, std::vector<double>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (std::size_t k = 0; k < F; ++k) {
            int32_t i = faces[k][0], j = faces[k][1];
            out[i] -= w[k];
            out[j] += w[k];
        }
    };

    auto norm2 = [](std::vector<double> const& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return std::sqrt(s);
    };

    // --- Initialise LSQR (Paige & Saunders 1982) ---
    std::vector<double> u(b_face, b_face + F);
    double beta = norm2(u);
    if (beta < 1e-15) return std::vector<double>(N, 0.0);
    for (auto& x : u) x /= beta;

    std::vector<double> v(N), w(N), x(N, 0.0), tmp_f(F), tmp_n(N);
    ATw(u, v);
    double alpha = norm2(v);
    if (alpha < 1e-15) return std::vector<double>(N, 0.0);
    for (auto& y : v) y /= alpha;
    w = v;

    double phi_bar = beta, rho_bar = alpha;

    for (std::size_t it = 0; it < iter_lim; ++it) {
        // Bidiagonalisation: beta_new * u_new = A*v - alpha*u
        Av(v, tmp_f);
        for (std::size_t k = 0; k < F; ++k) tmp_f[k] -= alpha * u[k];
        double beta_new = norm2(tmp_f);
        if (beta_new < 1e-15) break;
        for (auto& y : tmp_f) y /= beta_new;
        std::swap(u, tmp_f);

        // alpha_new * v_new = A^T*u_new - beta_new*v
        ATw(u, tmp_n);
        for (std::size_t n = 0; n < N; ++n) tmp_n[n] -= beta_new * v[n];
        double alpha_new = norm2(tmp_n);
        if (alpha_new < 1e-15) break;
        for (auto& y : tmp_n) y /= alpha_new;
        std::swap(v, tmp_n);

        // Plane rotation
        double rho   = std::sqrt(rho_bar * rho_bar + beta_new * beta_new);
        double c     = rho_bar / rho;
        double s     = beta_new / rho;
        double theta = s * alpha_new;
        rho_bar      = -c * alpha_new;
        double phi   = c * phi_bar;
        phi_bar      = s * phi_bar;

        // x += (phi/rho) * w;   w = v - (theta/rho) * w
        double sc1 = phi / rho, sc2 = theta / rho;
        for (std::size_t n = 0; n < N; ++n) {
            x[n] += sc1 * w[n];
            w[n]  = v[n] - sc2 * w[n];
        }

        alpha = alpha_new;
    }

    // Mean-zero, unit max-abs normalise
    double mean = 0.0;
    for (double y : x) mean += y;
    mean /= double(N);
    for (auto& y : x) y -= mean;
    double mx = 0.0;
    for (double y : x) mx = std::max(mx, std::abs(y));
    if (mx > 0.0) for (auto& y : x) y /= mx;

    return x;
}

} // namespace equilibrium::dst
