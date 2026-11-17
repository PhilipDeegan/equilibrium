#ifndef _EQUILIBRIUM_SUBSTRATE_HPP_
#define _EQUILIBRIUM_SUBSTRATE_HPP_
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include <cstddef>

namespace eq {

// Per-leg face weights: 8 hex (W = sqrt3) then 6 sq (W = 1), so the Hodge star
// W^2 is the Kelvin geometry's 3:1 (before 2026-09-28: W = 3, star 9:1) --
// matches Python lattice._W and bcc::face_weight.
static constexpr double S3 = 1.7320508075688772;
static constexpr double FACE_W[14] = {S3, S3, S3, S3, S3, S3, S3, S3, 1, 1, 1, 1, 1, 1};

// Throughput bounds — anchored to Sgr A* / S2 periapsis (30 s/day clock deficit)
static constexpr int    N_SAGA   = 2880;               // BH tick ratio
static constexpr double LAM_MAX  = N_SAGA - 1;         // = 2879 — maximum winding
static constexpr double SIGMA_MIN = 1.0 / N_SAGA;      // ≈ 3.47e-4 — BH core throughput
static constexpr double SIGMA_MAX = 1.0;               // void

// Symplectic Euler step for the Kelvin substrate.
//
// Array layout (all C-contiguous / row-major):
//   u         (N,)       essence field
//   phi       (N, 14)    directed face flux
//   ell       (N,)       always 1.0 — kept for ABI compat; sigma = 1/(1+lam)
//   lam       (N,)       winding occupation — primary state variable
//   fixed_u   (N,)       uint8 mask: 1 = u is frozen
//   fixed_phi (N, 14)    uint8 mask: 1 = face flux frozen
//   nb_idx    (N, 14)    neighbour tile indices, -1 = boundary (no neighbour)
//
// Three passes separated by std::barrier (threads spawned once per call):
//   1. Compute du = -dt*sigma*div(phi), then u_new = u + du (or keep if fixed)
//   2. Update phi from u_new — sigma read-only, no cross-write between threads
//   3. Apply u_new -> u
//
// At thread-range junctions: tile i (thread A) writes phi[i,k], tile j (thread B)
// writes phi[j,mirror_k] — different rows, no race.  u_new[j] is read-only in pass 2.
inline void substrate_step(
    double* __restrict__ u,
    double* __restrict__ phi,
    double* __restrict__ /*ell*/,  // kept for ABI compat; sigma = 1/(1+lam), ell unused
    double const* __restrict__ lam,
    uint8_t const* __restrict__ fixed_u,
    uint8_t const* __restrict__ fixed_phi,
    int32_t const* __restrict__ nb_idx,
    int N,
    double dt,
    double /*alpha*/,              // ignored: no ell update in discrete model
    int n_threads)
{
    std::vector<double> u_new(N), du(N);

    // ---- pass 1: u update ----
    auto pass1 = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            double const si = 1.0 / (1.0 + lam[i]);
            double acc = 0.0;
            int32_t const* nb_i  = nb_idx + i * 14;
            double  const* phi_i = phi    + i * 14;
            for (int k = 0; k < 14; ++k) {
                int32_t const j = nb_i[k];
                if (j < 0) continue;
                double const sface = 0.5 * (si + 1.0 / (1.0 + lam[j]));  // same as pass 2
                acc += sface * FACE_W[k] * phi_i[k];
            }
            du[i]    = -dt * acc;
            u_new[i] = fixed_u[i] ? u[i] : u[i] + du[i];
        }
    };

    // ---- pass 2: phi update ----
    auto pass2 = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            double const si     = 1.0 / (1.0 + lam[i]);
            double const ui     = u_new[i];
            int32_t const* nb_i = nb_idx    + i * 14;
            uint8_t const* fp_i = fixed_phi + i * 14;
            double*        phi_i = phi      + i * 14;

            for (int k = 0; k < 14; ++k) {
                int32_t const j = nb_i[k];
                if (j < 0 || fp_i[k]) continue;
                double const sj    = 1.0 / (1.0 + lam[j]);
                double const sface = 0.5 * (si + sj);
                phi_i[k] += -dt * sface * FACE_W[k] * (u_new[j] - ui);
            }
        }
    };

    // ---- pass 3: apply u_new ----
    auto pass3 = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i)
            u[i] = u_new[i];
    };

    if (n_threads <= 1) {
        pass1(0, N);
        pass2(0, N);
        pass3(0, N);
        return;
    }

    int const chunk = (N + n_threads - 1) / n_threads;
    int const n_actual = (N + chunk - 1) / chunk;
    std::barrier sync(n_actual);

    auto worker = [&](int lo, int hi) {
        pass1(lo, hi);
        sync.arrive_and_wait();
        pass2(lo, hi);
        sync.arrive_and_wait();
        pass3(lo, hi);
    };

    std::vector<std::thread> tv;
    tv.reserve(n_actual - 1);
    for (int t = 1; t < n_actual; ++t)
        tv.emplace_back(worker, t * chunk, std::min(t * chunk + chunk, N));
    worker(0, std::min(chunk, N));
    for (auto& th : tv) th.join();
}

// Face adjacency for step_cross: face k's edge-adjacent faces (6-wide, -1 = unused).
// Matches Python _FACE_ADJ in equilibrium/substrate/lattice.py.
static constexpr int FACE_ADJ[14][6] = {
    { 3,  5,  6,  8, 10, 12},  // 0 (-,-,-)
    { 2,  4,  7,  8, 10, 13},  // 1 (-,-,+)
    { 1,  4,  7,  8, 11, 12},  // 2 (-,+,-)
    { 0,  5,  6,  8, 11, 13},  // 3 (-,+,+)
    { 1,  2,  7,  9, 10, 12},  // 4 (+,-,-)
    { 0,  3,  6,  9, 10, 13},  // 5 (+,-,+)
    { 0,  3,  5,  9, 11, 12},  // 6 (+,+,-)
    { 1,  2,  4,  9, 11, 13},  // 7 (+,+,+)
    { 0,  1,  2,  3, -1, -1},  // 8  -x sq
    { 4,  5,  6,  7, -1, -1},  // 9  +x sq
    { 0,  1,  4,  5, -1, -1},  // 10 -y sq
    { 2,  3,  6,  7, -1, -1},  // 11 +y sq
    { 0,  2,  4,  6, -1, -1},  // 12 -z sq
    { 1,  3,  5,  7, -1, -1},  // 13 +z sq
};

// Opposite face (mirror) for each face k — matches Python _K_REV.
static constexpr int FACE_MIRROR[14] = {7,6,5,4,3,2,1,0,9,8,11,10,13,12};

// Number of edge-adjacent faces per face type: 6 for hex, 4 for sq.
static constexpr int FACE_N_ADJ[14]  = {6,6,6,6,6,6,6,6,4,4,4,4,4,4};

// Lam transport: Δlam_i = -dt·scale·σ_i·lam_i·Σ_k W_k·φ_{ik}
//
// Single pass — each cell reads its own lam and phi only (no cross-cell
// dependency), so no barrier is needed between threads.
// fixed_u cells are skipped (lam frozen, consistent with Python step_lam).
// lam is clamped to [0, LAM_MAX] after each update.
inline void substrate_step_lam(
    double* __restrict__ lam,
    double const* __restrict__ phi,
    uint8_t const* __restrict__ fixed_u,
    int32_t const* __restrict__ nb_idx,
    int N,
    double dt,
    double scale,
    int n_threads)
{
    auto pass = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            if (fixed_u[i]) continue;
            double const li = lam[i];
            if (li == 0.0) continue;
            double const si = 1.0 / (1.0 + li);
            double acc = 0.0;
            int32_t const* nb_i  = nb_idx + i * 14;
            double  const* phi_i = phi    + i * 14;
            for (int k = 0; k < 14; ++k)
                if (nb_i[k] >= 0) acc += FACE_W[k] * phi_i[k];
            double nl = li + (-dt * scale * si * li * acc);
            lam[i] = nl < 0.0 ? 0.0 : (nl > LAM_MAX ? LAM_MAX : nl);
        }
    };

    if (n_threads <= 1) { pass(0, N); return; }
    int const chunk  = (N + n_threads - 1) / n_threads;
    int const n_avail = (N + chunk - 1) / chunk;
    std::vector<std::thread> tv;
    tv.reserve(n_avail - 1);
    for (int t = 1; t < n_avail; ++t)
        tv.emplace_back(pass, t * chunk, std::min(t * chunk + chunk, N));
    pass(0, std::min(chunk, N));
    for (auto& th : tv) th.join();
}

// Cross step: inward phi on face k is rerouted to edge-adjacent faces.
//
// Conservative formulation — for each inward face k at cell i:
//   delta = -dt * scale * sigma(lam_i)^2 * phi[i,k]   (≥ 0)
//   phi[i, k]           += n_adj * delta      drain source
//   phi[nb_k, mirror_k] -= n_adj * delta      antisymmetry of drain
//   phi[i, j]           += delta    for each adj j   redistribute
//   phi[nb_j, mirror_j] -= delta    for each adj j   antisymmetry
//
// sigma(lam_i) = 1/(1+lam_i) is the same per-cell time-dilation throttle used
// by substrate_step_lam; squared here so a high-lam knot resists topology-
// opening (a real proton-scale lam freezes this almost entirely, giving mass
// a stability-against-decay effect that was previously absent from this step).
//
// Global sum of dphi is zero — no phi is created or destroyed.
// Phase 1 (compute dphi) is single-threaded to avoid scatter-write races.
// Phase 2 (apply) is threaded.
inline void substrate_step_cross(
    double* __restrict__ phi,
    uint8_t const* __restrict__ fixed_phi,
    int32_t const* __restrict__ nb_idx,
    double const* __restrict__ lam,
    int N,
    double dt,
    double scale,
    int n_threads)
{
    std::vector<double> dphi_buf(static_cast<std::size_t>(N) * 14, 0.0);
    double* dphi = dphi_buf.data();

    // Phase 1 — single-threaded (scatter writes touch neighbour cells)
    for (int i = 0; i < N; ++i) {
        double  const* phi_i = phi    + i * 14;
        int32_t const* nb_i  = nb_idx + i * 14;
        double const si  = 1.0 / (1.0 + lam[i]);
        double const si2 = si * si;

        for (int k = 0; k < 14; ++k) {
            double inward = phi_i[k] < 0.0 ? phi_i[k] : 0.0;
            if (inward == 0.0) continue;
            double const delta = -dt * scale * si2 * inward;  // ≥ 0
            int const n_adj = FACE_N_ADJ[k];
            int const mk    = FACE_MIRROR[k];
            int const nb_k  = nb_i[k];

            // Drain source face k
            dphi[i * 14 + k] += n_adj * delta;
            // Mirror of drain at neighbour nb_k
            if (nb_k >= 0)
                dphi[nb_k * 14 + mk] -= n_adj * delta;

            for (int m = 0; m < 6; ++m) {
                int const j = FACE_ADJ[k][m];
                if (j < 0) break;
                int const mj   = FACE_MIRROR[j];
                int const nb_j = nb_i[j];
                // Redistribute to adjacent face j
                dphi[i * 14 + j] += delta;
                // Mirror at neighbour nb_j
                if (nb_j >= 0)
                    dphi[nb_j * 14 + mj] -= delta;
            }
        }
    }

    // Phase 2 — apply dphi (no cross-cell deps, safe to thread)
    auto apply = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            uint8_t const* fp_i  = fixed_phi + i * 14;
            double*        phi_i = phi        + i * 14;
            double const*  dp_i  = dphi       + i * 14;
            for (int j = 0; j < 14; ++j)
                if (!fp_i[j]) phi_i[j] += dp_i[j];
        }
    };

    if (n_threads <= 1) { apply(0, N); return; }
    int const chunk   = (N + n_threads - 1) / n_threads;
    int const n_avail = (N + chunk - 1) / chunk;
    std::vector<std::thread> tv;
    tv.reserve(n_avail - 1);
    for (int t = 1; t < n_avail; ++t)
        tv.emplace_back(apply, t * chunk, std::min(t * chunk + chunk, N));
    apply(0, std::min(chunk, N));
    for (auto& th : tv) th.join();
}

// Cross step, barrier gate: stateless, instantaneous, all-or-nothing per cell
// (doc/dst.md: "the crossing either happens or it doesn't: a gate condition
// on the instantaneous phi state").
//
// Pooled per-cell drive drive_i = Σ_k W_k·max(-phi_i[k],0) (invalid/boundary
// faces contribute 0) is compared against a fixed threshold m0 — no default,
// see equilibrium/substrate/lattice.py step_cross(gate="barrier") / doc/
// dst_next.md sec 8 for calibration. If drive_i <= m0 the cell contributes
// nothing this tick (all 14 deltas are 0); otherwise every inward face drains
// and redistributes via the same conservative kernel as substrate_step_cross
// (gated by sigma(lam_i)^2, same drain+mirror discipline), matching
// equilibrium/substrate/lattice.py step_cross(gate="barrier") and
// tst/substrate/cross_v2_uniform.py CrossV2(gate="barrier") bit-for-bit.
//
// Global sum of dphi is zero -- no phi is created or destroyed.
// Phase 1 (compute dphi) is single-threaded to avoid scatter-write races.
// Phase 2 (apply) is threaded.
inline void substrate_step_cross_barrier(
    double* __restrict__ phi,
    uint8_t const* __restrict__ fixed_phi,
    int32_t const* __restrict__ nb_idx,
    double const* __restrict__ lam,
    int N,
    double dt,
    double scale,
    double m0,
    int n_threads)
{
    std::vector<double> dphi_buf(static_cast<std::size_t>(N) * 14, 0.0);
    double* dphi = dphi_buf.data();

    // Phase 1 — single-threaded (scatter writes touch neighbour cells)
    for (int i = 0; i < N; ++i) {
        double  const* phi_i = phi    + i * 14;
        int32_t const* nb_i  = nb_idx + i * 14;
        double const si  = 1.0 / (1.0 + lam[i]);
        double const si2 = si * si;

        double drive = 0.0;
        for (int k = 0; k < 14; ++k) {
            if (nb_i[k] < 0) continue;
            double const inward_k = phi_i[k] < 0.0 ? -phi_i[k] : 0.0;
            drive += FACE_W[k] * inward_k;
        }
        if (drive <= m0) continue;  // gate closed -- cell contributes nothing

        for (int k = 0; k < 14; ++k) {
            if (nb_i[k] < 0) continue;
            double const inward_k = phi_i[k] < 0.0 ? -phi_i[k] : 0.0;
            if (inward_k == 0.0) continue;
            double const delta = dt * scale * si2 * inward_k;  // ≥ 0
            int const n_adj = FACE_N_ADJ[k];
            int const mk    = FACE_MIRROR[k];
            int const nb_k  = nb_i[k];

            // Drain source face k
            dphi[i * 14 + k] += n_adj * delta;
            // Mirror of drain at neighbour nb_k
            dphi[nb_k * 14 + mk] -= n_adj * delta;

            for (int m = 0; m < 6; ++m) {
                int const j = FACE_ADJ[k][m];
                if (j < 0) break;
                int const mj   = FACE_MIRROR[j];
                int const nb_j = nb_i[j];
                // Redistribute to adjacent face j
                dphi[i * 14 + j] += delta;
                // Mirror at neighbour nb_j
                if (nb_j >= 0)
                    dphi[nb_j * 14 + mj] -= delta;
            }
        }
    }

    // Phase 2 — apply dphi (no cross-cell deps, safe to thread)
    auto apply = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            uint8_t const* fp_i  = fixed_phi + i * 14;
            double*        phi_i = phi        + i * 14;
            double const*  dp_i  = dphi       + i * 14;
            for (int j = 0; j < 14; ++j)
                if (!fp_i[j]) phi_i[j] += dp_i[j];
        }
    };

    if (n_threads <= 1) { apply(0, N); return; }
    int const chunk   = (N + n_threads - 1) / n_threads;
    int const n_avail = (N + chunk - 1) / chunk;
    std::vector<std::thread> tv;
    tv.reserve(n_avail - 1);
    for (int t = 1; t < n_avail; ++t)
        tv.emplace_back(apply, t * chunk, std::min(t * chunk + chunk, N));
    apply(0, std::min(chunk, N));
    for (auto& th : tv) th.join();
}

// Implicit-midpoint step with self-sourced lam: lam = eps + alpha (u^2 + 1/2 sum_k phi_k^2), sigma = 1/(1+lam),
// face sigma = mean of the two cells (as substrate_step). Solves
//   x1 = x0 + dt f(x_mid),  x_mid = (x0 + x1)/2,  f = (-sum sface W phi, -sface W (u_j - u_i))
// by fixed-point iteration; continuous-time H is conserved exactly (skew J(x), quadratic H), and so is the discrete
// one (exp/2026/09/30/electron_survival). Mirrors self_lam_long.py step_midpoint: the first iterate is the explicit
// Euler guess (x_mid = x0), then iterate until max |x_new - x1| < tol.
//
// Double-buffered, no copies: (u0, phi0) read-only, (u1, phi1) written; the caller swaps. lam receives the lam of the
// last midpoint. um / sig are N-sized scratch. Boundary faces (nb < 0) carry no flux and phi1 = phi0 there. fixed
// masks are ignored (as the numpy reference); apply sponges / BCs after. Returns the iteration count, or -1 if not
// converged in max_iter (u1/phi1 then hold the last iterate).
inline int substrate_step_selflam_midpoint(
    double const* __restrict__ u0,
    double const* __restrict__ phi0,
    double* __restrict__ u1,
    double* __restrict__ phi1,
    double* __restrict__ lam,
    double* __restrict__ um,
    double* __restrict__ sig,
    int32_t const* __restrict__ nb_idx,
    int N,
    double dt,
    double alpha,
    double eps,
    double tol,
    int max_iter,
    int n_threads)
{
    int const n_t = std::max(1, std::min(n_threads, N));
    int const chunk = (N + n_t - 1) / n_t;
    int const n_actual = (N + chunk - 1) / chunk;
    std::vector<double> err_t(n_actual, 0.0);
    bool first = true, done = false, converged = false;
    int iters = 0;

    // pass A: midpoint u, lam, sigma (own rows only)
    auto passA = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            double const* p0 = phi0 + i * 14;
            double const* p1 = phi1 + i * 14;
            int32_t const* nb_i = nb_idx + i * 14;
            double const umi = first ? u0[i] : 0.5 * (u0[i] + u1[i]);
            double s2 = 0.0;
            for (int k = 0; k < 14; ++k) {
                if (nb_i[k] < 0) continue;
                double const pm = first ? p0[k] : 0.5 * (p0[k] + p1[k]);
                s2 += pm * pm;
            }
            um[i] = umi;
            lam[i] = eps + alpha * (umi * umi + 0.5 * s2);
            sig[i] = 1.0 / (1.0 + lam[i]);
        }
    };

    // pass B: new iterate from the midpoint (reads um / sig of neighbours, writes own rows of u1 / phi1)
    auto passB = [&](int t, int lo, int hi) {
        double err = 0.0;
        for (int i = lo; i < hi; ++i) {
            double const* p0 = phi0 + i * 14;
            double* p1 = phi1 + i * 14;
            int32_t const* nb_i = nb_idx + i * 14;
            double const si = sig[i], umi = um[i];
            double acc = 0.0;
            for (int k = 0; k < 14; ++k) {
                int32_t const j = nb_i[k];
                double pn = p0[k];
                if (j >= 0) {
                    double const sw = 0.5 * (si + sig[j]) * FACE_W[k];
                    double const pm = first ? p0[k] : 0.5 * (p0[k] + p1[k]);
                    acc += sw * pm;
                    pn = p0[k] - dt * sw * (um[j] - umi);
                }
                double const old = first ? p0[k] : p1[k];
                err = std::max(err, std::abs(pn - old));
                p1[k] = pn;
            }
            double const un = u0[i] - dt * acc;
            double const old = first ? u0[i] : u1[i];
            err = std::max(err, std::abs(un - old));
            u1[i] = un;
        }
        err_t[t] = err;
    };

    // the numpy reference measures err from the second iterate on (its first iterate is the Euler guess)
    auto finish = [&]() noexcept {
        ++iters;
        double e = 0.0;
        for (double v : err_t) e = std::max(e, v);
        if (!first && e < tol) done = converged = true;
        if (iters > max_iter) done = true;
        first = false;
    };

    if (n_actual <= 1) {
        while (!done) {
            passA(0, N);
            passB(0, 0, N);
            finish();
        }
    } else {
        std::barrier syncA(n_actual);
        std::barrier syncB(n_actual, finish);
        auto worker = [&](int t, int lo, int hi) {
            while (true) {
                passA(lo, hi);
                syncA.arrive_and_wait();
                passB(t, lo, hi);
                syncB.arrive_and_wait();  // finish() runs once here, before anyone proceeds
                if (done) break;
            }
        };
        std::vector<std::thread> tv;
        tv.reserve(n_actual - 1);
        for (int t = 1; t < n_actual; ++t)
            tv.emplace_back(worker, t, t * chunk, std::min(t * chunk + chunk, N));
        worker(0, 0, std::min(chunk, N));
        for (auto& th : tv) th.join();
    }
    // iterations counted as in the numpy reference (Euler guess not counted)
    return converged ? iters - 1 : -1;
}

}  // namespace eq

#endif  // _EQUILIBRIUM_SUBSTRATE_HPP_
