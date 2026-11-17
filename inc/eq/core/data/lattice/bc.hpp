#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_BC_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_BC_HPP_

// Boundary-condition dispatch for faces with no local neighbour (nb[k] < 0).
//
// Two complementary mechanisms (applied in order):
//
//   1. face_ghost_by_level — wire a specific boundary face to any real cell in
//                    the cluster; that cell's u/L/sigma are used directly.
//                    Use for periodic BCs, coarse-fine ghosts, or any topology
//                    where "no neighbour" should redirect to another cell.
//                    Stored as one sorted vector per refinement level so that
//                    step_subcycle() can filter by level and binary search is
//                    fast within each (typically small) per-level set.
//                    Call sort_face_ghosts() once after all wire_face() calls.
//
//   2. fn          — plain C++ function pointer called for any boundary face
//                    not covered by face_ghost_by_level.  Signature:
//                        double fn(double u_i, double target, double weight)
//                    Built-in implementations live as static members:
//                        BoundaryCondition::absorbing   ghost u = u_i (zero flux)
//                        BoundaryCondition::dirichlet   ghost u = target
//                        BoundaryCondition::sponge      ghost u = (1-w)*u_i + w*target
//
// Ghost u for face_ghost entries uses the actual cell's sigma and L, giving a
// proper two-cell flux calculation.  For fn entries, ghost L = L_i is assumed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <iterator>
#include <utility>

namespace equilibrium::dst::discrete {

// Classifies how a boundary face (nb[k] < 0) contributes to the u-update (pass 1).
// Periodic faces are NOT tracked here — their inclusion is determined by
// find_face_ghost() directly (ghost entry present → phi maintained → include).
//
//   NONE       — open / no BC; phi[k] frozen at IC (should be 0).  Exclude.
//   ABSORBING  — fn = absorbing (dphi = 0); phi[k] frozen.  Exclude.
//   FUNCTIONAL — fn = dirichlet / sponge / custom; phi[k] driven each step.  Include.
//   MPI_REMOTE — remote_flux Neumann or future halo exchange; phi[k] driven.  Include.
//
// Set fn_type alongside fn / remote_flux:
//   absorbing()  → ABSORBING    (exclude)
//   dirichlet()  → FUNCTIONAL   (include)
//   sponge()     → FUNCTIONAL   (include)
//   remote()     → MPI_REMOTE   (include)
//   neumann()    → NONE         (exclude — dphi=0 on all boundary faces)
//
// Default is NONE (safe: zero-flux, exclude from u).
enum class BCType : std::uint16_t { NONE = 0, ABSORBING, FUNCTIONAL, MPI_REMOTE };

struct BoundaryCondition {
  // Per-level sorted ghost wiring: face_ghost_by_level[level] holds
  // (encode(cell_g, face_k), ghost_global) pairs sorted by key.
  using FaceGhostEntry = std::pair<std::uint64_t, std::int32_t>;
  std::vector<std::vector<FaceGhostEntry>> face_ghost_by_level;

  // Cluster-wide fallback function pointer and its two scalar parameters.
  // nullptr = open BC (dphi = 0) for all boundary faces not in face_ghost.
  // Always set fn_type alongside fn so _step_pass1 can decide whether fn-handled
  // faces drive phi and should be included in the u-update (see BCType above).
  double (*fn)(double u_i, double target, double weight) = nullptr;
  BCType fn_type = BCType::NONE;  // ABSORBING=exclude; FUNCTIONAL/MPI_REMOTE=include
  double target  = 0.0;
  double weight  = 0.5;

  // Directional Neumann — far-field / remote-lattice gradient.
  // Models a distant cluster too far to simulate directly; its u-field appears
  // as a nearly-uniform gradient across the domain boundary.
  // ghost u = u_i + remote_flux * dot(face_dir[k], remote_axis).
  // When remote_flux != 0 this supersedes fn for all boundary faces not
  // covered by face_ghost.  Set remote_flux = 0 to disable (default).
  double remote_flux                  = 0.0;
  std::array<double, 3> remote_axis   = {0.0, 0.0, 1.0};  // unit vector toward remote source

  // ---------------------------------------------------------------------------
  // Predefined BC implementations (assign to fn)
  // ---------------------------------------------------------------------------

  static double absorbing(double u_i, double /*target*/, double /*weight*/) noexcept {
    return u_i;  // ghost u = u_i → dphi = 0
  }

  static double dirichlet(double /*u_i*/, double target, double /*weight*/) noexcept {
    return target;
  }

  static double sponge(double u_i, double target, double weight) noexcept {
    return (1.0 - weight) * u_i + weight * target;
  }

  // Volumetric sponge for step_sponge(dt), per leaf: vol_rate (1/time, preferred) or vol_damp
  // (legacy fixed per-call multiplier 0 < d <= 1, not dt-consistent). u relaxes toward `target`
  // with the leaf's factor; each face (phi, and E in the continuous model) is scaled by one
  // factor shared by both stored copies -- exp(-mean rate dt), or the mean of the two legacy
  // multipliers -- so phi_ki = -phi_ik survives (exp/2026/09/29/sponge_assess,
  // tst/substrate/test_bc_invariants.py).
  std::vector<double> vol_rate;
  std::vector<double> vol_damp;

  bool has_sponge() const { return !vol_rate.empty() || !vol_damp.empty(); }

  double sponge_leaf(std::size_t j, double dt) const {
    return vol_rate.empty() ? vol_damp[j] : std::exp(-vol_rate[j] * dt);
  }

  // jn < 0: no neighbour leaf (boundary face) -> the leaf's own factor.
  double sponge_face(std::size_t j, std::int32_t jn, double dt) const {
    if (jn < 0) return sponge_leaf(j, dt);
    auto const n = static_cast<std::size_t>(jn);
    return vol_rate.empty() ? 0.5 * (vol_damp[j] + vol_damp[n])
                            : std::exp(-0.5 * (vol_rate[j] + vol_rate[n]) * dt);
  }

  // Per-leaf exponential decay RATE (units 1/time, unlike vol_damp's fixed
  // per-call fraction). When non-empty, step_damped()/_step_pass1_damped/
  // _step_pass3_damped apply u_new = exp(-gamma*dt)*u_old + du and
  // phi_new = exp(-gamma*dt)*phi_old + dphi directly inside the normal
  // leapfrog update (decaying only the OLD state carried into the step, not
  // the freshly computed flux increment) instead of a separate post-step
  // multiply — avoids the operator-splitting reflection vol_damp has, and
  // is independent of how often the step is invoked per unit physical time.
  // Typically computed as gamma_max * t² with the same t as vol_damp.
  std::vector<double> gamma;

  // Per-(leaf, face) directional decay rate, flattened as k*N+j (same layout
  // as AdaptiveCellCluster::temp_phi). PML-lite: only damps the component of
  // flux crossing faces that point mostly radially outward (gamma_face ~
  // gamma_max*t(r)²*cos²(angle to radial)), leaving faces tangential to the
  // boundary — and `u` itself — completely undamped. A real PML never
  // attenuates propagation tangential to the absorbing boundary, only
  // normal to it; this is a cheap approximation of that idea on the BCC
  // mesh's 14 non-orthogonal face directions (no auxiliary split fields or
  // coordinate stretching, unlike a textbook Berenger PML). Applied by
  // _step_pass3_dir/step_dir_damped; u's own update is untouched (uses the
  // plain undamped _step_pass1). Size must equal NF*N when used.
  std::vector<double> gamma_face;

  // Split-field (Berenger-style) PML auxiliary state and geometry weights.
  // Unlike vol_damp/gamma/gamma_face above (each applies a single lumped
  // decay to the WHOLE field), a genuine PML splits u and each phi[k] into
  // a "radial" auxiliary sub-component (which decays) and an implicit
  // "tangential" remainder (which doesn't), recombining them only for the
  // physical field seen everywhere else — this bookkeeping is what actually
  // gives PML its near-zero-reflection property. The earlier gamma_face-
  // based "directional" attempt skipped it and measurably failed (see
  // [[project_poynting_perf_r150_farfield]]): it damped the ALREADY-
  // COMBINED field by a directional weight, rather than tracking two
  // separate accumulators whose sum reconstructs the field.
  //
  //   u_total (cell.u, unchanged storage)      = u_r (u_aux, decays) + u_p (implicit)
  //   phi_total (cell.phi[k], unchanged storage) = phi_r[k] (phi_aux, decays) + phi_p[k] (implicit)
  //
  // proj_r[k*N+j] = cos²(angle between face k's direction and the local
  // radial direction) — a pure DIMENSIONLESS geometry weight (0=tangential
  // face, 1=purely radial face) used to route a FRACTION of each face's
  // ordinary (undamped) flux increment into the decaying auxiliary; `gamma`
  // above is reused as the auxiliary's own decay rate (same per-leaf rate
  // as the integrated variant, not face-dependent).
  std::vector<double> proj_r;
  // Mutable auxiliary simulation state, evolved in place by
  // _step_pass1_pml/_step_pass3_pml — u_aux is per-leaf (size N), phi_aux is
  // per-(leaf,face) (size NF*N, same k*N+j layout as temp_phi). Zero-
  // initialized; stays at zero outside the sponge since gamma==0 there, so
  // it never diverges from the physical field in the interior. NOT
  // persisted by AdaptiveLattice::save()/load() or touched by AMR
  // split/merge — a resumed run's PML aux state restarts at zero (a small,
  // bounded transient, not a correctness bug, since it's re-derived from
  // the ordinary field's own flux each step).
  std::vector<double> u_aux;
  std::vector<double> phi_aux;

  // Outflow (first-order Mur/Sommerfeld one-way) boundary condition.
  //
  // Unlike fn/face_ghost/remote_flux above -- which only ever set phi[k] via
  // _step_pass2_bc and are structurally invisible to u, since _step_pass1's
  // divergence sum is gated on nb[k]>=0 and a genuine domain-edge face never
  // has that (see [[project_bc_fn_inert_on_u]]) -- this mechanism is wired
  // directly into _step_pass1 too, gated per-(leaf,face) by outflow_active
  // so every other BC path (periodic, vol_damp/gamma/proj_r sponge/PML, fn,
  // remote_flux) stays an EXACT no-op. That gating matters: a past session
  // found that naively including ghost-wired phi in pass1's u-accumulator
  // causes exponential instability ([[project_bc_enum]]) -- this is the
  // reason plain `fn`-based faces were never wired into pass1 at all, and
  // why this new mechanism uses a proper one-way EXTRAPOLATED ghost (not an
  // arbitrary target) plus its own persisted per-face history, rather than
  // just lifting the nb[k]>=0 guard for existing FUNCTIONAL/MPI_REMOTE faces.
  //
  // First-order one-way wave extrapolation along the face normal (Mur 1981),
  // derived by evaluating d(u)/dt + c*d(u)/dx = 0 at the midpoint between
  // the ghost and the owning cell, both in space and in time:
  //   r = (L_i - c_i*dt) / (L_i + c_i*dt)
  //   u_ghost^{n+1} = u_i^n + r*(outflow_ghost[k*N+j] - u_i^{n+1})
  // where u_i^{n+1} is this step's already-updated cell.u (pass1 runs first)
  // and u_i^n is recovered as u_i^{n+1} - temp_du[j] (pass1's own du for this
  // step, still valid when _step_pass2_bc runs later the same step).
  // outflow_ghost[k*N+j] holds last step's u_ghost (the "previous timestep"
  // a one-way condition needs in place of a real spatial neighbour) and is
  // overwritten with the new u_ghost each step. The resulting dphi uses the
  // SAME coefficient (sig_i * face_weight[k]/L_i) both existing fn-driven
  // ghost faces (_step_pass2_bc) and real same-level interior faces
  // (_face_flux_coefficient) already use -- there's no real neighbour, so
  // ghost sigma/L are assumed equal to the owning cell's own.
  //
  // c_i (local propagation speed along the face normal) is sigma_i *
  // outflow_c_lattice when outflow_c is empty (uniform override), else
  // outflow_c[j] directly. outflow_c_lattice has no fixed "correct" value in
  // this discrete scheme -- like sponge/PML's damp_max/gamma, it's tuned
  // empirically; a reasonable starting point is this project's existing
  // ad-hoc round-trip-time estimate (sqrt(10), see probe_sponge_quality_v3.py).
  //
  // Wired into _step_pass1/_step_pass1_damped/_step_pass1_pml (all consume
  // outflow_active identically -- an additional else-if branch alongside the
  // real-neighbour flux sum, gated the same way in every variant) plus
  // _step_pass2_bc (which persists outflow_ghost regardless of variant, see
  // step_pml/step_damped's shared call to it). _step_pass1_pml deliberately
  // does NOT feed the outflow contribution into acc_r/u_aux -- there's no
  // real neighbour to radially-project, it's genuine loss at the edge, not
  // sponge-tracked flux. Combining PML and outflow needs gamma==0 on
  // every true boundary (outflow_active) leaf -- co-locating PML decay with
  // outflow's ghost causes _step_pass2_bc's `u_i_old = u_i - temp_du[j]`
  // recovery to silently corrupt (it assumes temp_du IS the leaf's whole
  // u-update, which _step_pass1_pml's extra decay_loss subtraction breaks),
  // confirmed to cause exponential blowup starting right at the annulus's
  // own round-trip time when first tried without this exclusion. The
  // sponge still damps every real interior annulus cell; only the exact
  // handoff leaf at r_max stays gamma==0 so its Mur ghost sees a clean,
  // undamped du.
  // _step_pass1_lev still doesn't consume it (no AMR+outflow use case yet).
  // Populated via BoundaryCondition::outflow() / outflow_tapered() (lattice.cpp),
  // which need the cluster's own leaf_nb to know which faces are genuine
  // domain edges.
  std::vector<std::uint8_t> outflow_active;  // NF*N, per-(leaf,face); empty = disabled everywhere
  std::vector<double> outflow_c;             // N; empty = derive c_i = sigma_i*outflow_c_lattice
  double outflow_c_lattice = 0.0;
  std::vector<double> outflow_ghost;         // NF*N, persisted history, zero-initialized


  // Optional taper band (2026-09-16, [[project_bc_fn_inert_on_u]] idea 1):
  // an ADDITIVE relaxation sink applied to real interior leaves within
  // `taper_width` of the true domain edge, blending toward the same
  // one-way Mur-extrapolated target the true boundary ghost uses — meant
  // to give the outflow BC some absorbing thickness (delta>0 in the Nataf
  // sigma*delta sense) instead of the single zero-thickness interface at
  // r_max. Deliberately kept separate from outflow_active/_step_pass2_bc's
  // per-face ghost replacement: this only ADDS a small sink term to
  // _step_pass1's already-computed du for nb[k]>=0 cells, never touches the
  // real-neighbour flux sum itself, so it cannot reproduce the exponential
  // instability from wiring ghost phi into pass1's accumulator
  // ([[project_bc_enum]]). Empty = disabled (default; zero overhead).
  std::vector<double> outflow_taper_weight;  // N, per-leaf, in [0,1]; 0 outside the band
  std::vector<double> outflow_taper_ghost;   // N, per-leaf, persisted history, zero-initialized

  // ---------------------------------------------------------------------------
  // Key encoding: upper 60 bits = cell_g (as uint32), lower 4 bits = face_k
  // ---------------------------------------------------------------------------
  static constexpr std::uint64_t encode(std::int32_t g, std::size_t k) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(g)) << 4) |
           static_cast<std::uint64_t>(k & 0xFu);
  }

  // Add a ghost wiring for (g, face_k) at the given refinement level.
  // Entries are appended unsorted; call sort_face_ghosts() after all wire_face() calls.
  void wire_face(std::int32_t g, std::size_t k, std::int32_t ghost_g, std::uint8_t level) {
    if (level >= face_ghost_by_level.size()) face_ghost_by_level.resize(level + 1);
    face_ghost_by_level[level].push_back({encode(g, k), ghost_g});
  }

  // Sort each level's vector by key — call once after all wire_face() calls.
  void sort_face_ghosts() {
    for (auto& lv : face_ghost_by_level) std::sort(lv.begin(), lv.end());
  }

  // Returns ghost global index, or -1 if not found.
  std::int32_t find_face_ghost(std::int32_t g, std::size_t k, std::uint8_t level) const noexcept {
    if (level >= face_ghost_by_level.size()) return -1;
    auto const& lv  = face_ghost_by_level[level];
    auto const  key = encode(g, k);
    auto it = std::lower_bound(lv.begin(), lv.end(), FaceGhostEntry{key, 0});
    return (it != lv.end() && it->first == key) ? it->second : -1;
  }

  void unwire_face(std::int32_t g, std::size_t k, std::uint8_t level) {
    if (level >= face_ghost_by_level.size()) return;
    auto&      lv  = face_ghost_by_level[level];
    auto const key = encode(g, k);
    auto it = std::lower_bound(lv.begin(), lv.end(), FaceGhostEntry{key, 0});
    if (it != lv.end() && it->first == key) lv.erase(it);
  }

  void clear_face_ghosts() { face_ghost_by_level.clear(); }

  bool has_face_ghosts() const noexcept {
    for (auto const& lv : face_ghost_by_level)
      if (!lv.empty()) return true;
    return false;
  }
};

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_BC_HPP_
