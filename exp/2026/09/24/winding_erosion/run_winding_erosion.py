"""Erosion test for the octagon init_toroidal_winding candidate (electron only).

09-22's "persistence" run was 300 steps = t~1.5 (dt=0.4/78), under 1% of the
ring mode's own period (omega=0.0327 -> T~192 time units ~37k steps). This
runs long enough to see whether the winding keeps its energy localized.

Total energy is conserved by plain step(), but a localized structure can still
RADIATE: the discriminating quantity is the energy fraction remaining near the
ring path (within NEAR_D of any ring tile), plus the ring-mode amplitude
(cos/sin(theta) projections of u on ring tiles and phi on ring faces).

Arms:
  closed  -- init_toroidal_winding on the full closed octagon ring
  open    -- same seed, with a gap of GAP_FRAC of the ring zeroed (open arc,
             no closed winding) -- the non-winding control on the same geometry
  diffuse -- closed, plus phi face-diffusion (step_phi_diffuse ported to the
             cluster API, applied every DIFF_EVERY steps, operator-split)
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/deegan/git/eq")
sys.path.insert(0, "/home/deegan/git/eq/exp/2026/09/22/dipole_ic_gallery")

import numpy as np
from scipy.spatial import cKDTree

import plot_all_dipole_ics as gallery
from plot_per_ic_diagnostics import radial_profile
from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import (
    _MIRROR,
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
)

HOPS = 12
NEAR_D = 3.0
GAP_FRAC = 0.125
W_FACE = np.array([3.0] * 8 + [1.0] * 6)

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["closed", "open", "diffuse"], required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--report", type=int, default=500)
p.add_argument("--profile-every", type=int, default=2000)
p.add_argument("--eta", type=float, default=0.02)
p.add_argument("--diff-every", type=int, default=20)
p.add_argument("--lam", choices=["zero", "uniform1", "tube"], default="zero",
               help="fixed lam background: zero (vacuum), uniform1 (lam=1 everywhere, "
                    "no gradient), tube (lam=1/(1+(d/a)^2), d=distance to ring filament)")
p.add_argument("--tube-a", type=float, default=3.0)
args = p.parse_args()


def phi_diffuse(cl, nbj, dt, eta):
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)  # (14, N), lam=0 -> sigma=1
    dphi = np.zeros_like(phi)
    for k in range(14):
        nb = nbj[:, k]
        valid = nb >= 0
        safe = np.where(valid, nb, np.arange(len(nb)))
        d = phi[:, safe] - phi
        d[:, ~valid] = 0.0
        dphi += W_FACE[k] * d
    cl.set_leaf_phi(phi + dt * eta * dphi)


def main():
    t0 = time.time()
    alat, cl = gallery.build_cluster()
    pos = np.array(cl.leaf_pos)
    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="octagon", hops=HOPS)
    n_ring = len(ring_js)
    omega = init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0)

    ring_pos = pos[ring_js]
    centre = ring_pos.mean(axis=0)
    rel = ring_pos - centre
    theta = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    theta_next = np.append(theta[1:], theta[0] + 2 * np.pi)
    theta_mid = 0.5 * (theta + theta_next)

    if args.arm == "open":
        n_gap = int(round(GAP_FRAC * n_ring))
        u = cl.leaf_u.copy()
        phi = cl.leaf_phi.copy()
        for idx in range(n_gap):
            j, k = ring_js[idx], ring_faces[idx]
            u[j] = 0.0
            phi[k, j] = 0.0
            phi[int(_MIRROR[k]), ring_js[(idx + 1) % n_ring]] = 0.0
        cl.set_leaf_u(u)
        cl.set_leaf_phi(phi)
        print(f"  open arc: zeroed {n_gap}/{n_ring} ring tiles/faces", flush=True)

    d_ring, _ = cKDTree(ring_pos).query(pos)
    near = d_ring <= NEAR_D
    if args.lam == "uniform1":
        cl.set_leaf_lam(np.ones(cl.n_leaves))
    elif args.lam == "tube":
        cl.set_leaf_lam(1.0 / (1.0 + (d_ring / args.tube_a) ** 2))
    nbj = leaf_nb_j(cl) if args.arm == "diffuse" else None

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(
        f"arm={args.arm} n_leaves={cl.n_leaves} n_ring={n_ring} omega={omega:.6f}"
        f" dt={dt:.6f} T_mode={2*np.pi/omega:.1f} steps={args.steps} (t_end={args.steps*dt:.1f})"
        f" near_leaves={near.sum()} build_t={time.time()-t0:.1f}s",
        flush=True,
    )
    lam_now = np.asarray(cl.leaf_lam)
    print(f"  lam={args.lam} min={lam_now.min():.4f} max={lam_now.max():.4f}"
          f" mean_near={lam_now[near].mean():.4f}", flush=True)
    if args.arm == "diffuse":
        print(f"  eta={args.eta} diff_every={args.diff_every}"
              f" stab={args.eta*dt*args.diff_every*26:.4f} (<1)", flush=True)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_m, sin_m = np.cos(theta_mid), np.sin(theta_mid)
    face_j = np.array(ring_js)
    face_k = np.array(ring_faces)

    def measure():
        u = np.asarray(cl.leaf_u)
        phi = np.asarray(cl.leaf_phi)
        e = gallery.energy_density(cl)
        E = float(e.sum())
        E_near = float(e[near].sum())
        ur = u[face_j]
        pr = phi[face_k, face_j]
        a = [2 * np.mean(ur * cos_t), 2 * np.mean(ur * sin_t),
             2 * np.mean(pr * cos_m), 2 * np.mean(pr * sin_m)]
        return E, E_near, a, e

    rec = {"step": [], "t": [], "E": [], "E_near": [], "a_ucos": [], "a_usin": [],
           "a_pcos": [], "a_psin": []}
    profiles = {}

    def log(step_i):
        E, E_near, a, e = measure()
        for key, v in zip(["step", "t", "E", "E_near", "a_ucos", "a_usin", "a_pcos", "a_psin"],
                          [step_i, step_i * dt, E, E_near, *a]):
            rec[key].append(v)
        E0, En0 = rec["E"][0], rec["E_near"][0]
        print(f"  step {step_i:6d} t={step_i*dt:7.2f}  E/E0={E/E0:.6f}"
              f"  Enear/Enear0={E_near/En0:.6f}  near_frac={E_near/E:.4f}"
              f"  u(c,s)=({a[0]:+.4f},{a[1]:+.4f}) phi(c,s)=({a[2]:+.4f},{a[3]:+.4f})"
              f"  wall={time.time()-t0:.0f}s", flush=True)
        if step_i % args.profile_every == 0:
            r, prof, _ = radial_profile(cl, e, gallery.PLOT_R)
            profiles[step_i] = prof
            profiles["r"] = r

    log(0)
    for step_i in range(1, args.steps + 1):
        evolver.step(ts)
        if args.arm == "diffuse" and step_i % args.diff_every == 0:
            phi_diffuse(cl, nbj, dt * args.diff_every, args.eta)
        if step_i % args.report == 0:
            log(step_i)
            out = os.path.join(_HERE, f"erosion_{args.arm}_lam-{args.lam}.npz")
            np.savez(out, **{k: np.array(v) for k, v in rec.items()},
                     **{f"prof_{k}": v for k, v in profiles.items()},
                     omega=omega, dt=dt, near_d=NEAR_D)
    print(f"done total_t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
