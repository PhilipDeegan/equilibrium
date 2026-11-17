"""Which hex:square Hodge star ratio? Scan r = *1_hex / *1_sq on the BCC Delaunay / Kelvin complex.

Symbols from equilibrium.substrate.bcc_symbol (u sector: step() is r = 9; T sector: step_ae
curl='dec' is r = 9, 'dec_geo' r = 3). Reported: the spatial 2nd-order speed error dc/c / (kh)^2
(h = hex distance, no leapfrog term) over 404 directions: mean, spread max-min (anisotropy), and for
T the split of the two transverse branches (birefringence). Output: star_ratio.log.

Claims demonstrated (also asserted in tst/substrate/test_hodge_star.py):
  1. Any r > 0 is consistent: omega^2/k^2 is direction independent at k -> 0 in both sectors.
  2. The geometric star (dual face area / edge length) is exactly 3:1.
  3. The 2nd-order speed error is isotropic only at r = 2, in BOTH sectors (T: also no birefringence);
     this is doc/dst.md's w = 2/3 : 1/3 'maximal isotropy'. step()'s 9:1 is the most anisotropic tested.
  4. With the geometric *2 = 1, c_u^2 = (r+1)/4, c_T^2 = 4/(r+1): equal speeds need no extra *2 factor
     only at r = 3 (the Kelvin star is self-consistent DEC); any other r needs *2 = (r+1)^2/16.
"""
import numpy as np

from equilibrium.substrate import bcc_symbol as B


def main():
    sh, ss, vol = B.kelvin_star1()
    print(f"Kelvin geometry: *1 hex {sh:.6f}, square {ss:.6f}, ratio {sh / ss:.6f}; cell volume {vol:.6f}")
    dirs = B.directions()
    k0 = 1e-3
    print("\nk -> 0 consistency (spread of omega^2/k^2 over directions, relative):")
    for r in (1, 2, 3, 9):
        u = np.array([B.w2_u(k0 * n, r) / k0**2 for n in dirs[::20]])
        T = np.array([B.w2_T(k0 * n, r) / k0**2 for n in dirs[::20]])
        print(f"  r={r}: u c^2 {u.mean():.6f} [(r+1)/4 = {(r + 1) / 4:.6f}] spread {np.ptp(u) / u.mean():.1e};  "
              f"T c^2 {T.mean():.6f} [4/(r+1) = {4 / (r + 1):.6f}] spread {np.ptp(T) / T.mean():.1e};  "
              f"*2 for c_T = c_u: {u.mean() / T.mean():.6f}")
    print("\ndc/c / (kh)^2 (spatial), 404 directions: mean, spread (max-min); T: both branches pooled,"
          " birefringence = max |branch1 - branch2|")
    print(f"  {'r':>5} | {'u mean':>8} {'u spread':>8} | {'T mean':>8} {'T spread':>8} {'T biref':>8}")
    for r in (1, 1.5, 2, 2.5, 3, 4, 5, 9):
        res = [B.speed_error(n, r) for n in dirs]
        du = np.array([a for a, _ in res])
        dT = np.array([b for _, b in res])
        print(f"  {r:5.2f} | {du.mean():+8.4f} {np.ptp(du):8.4f} | {dT.mean():+8.4f} {np.ptp(dT):8.4f} "
              f"{np.abs(dT[:, 0] - dT[:, 1]).max():8.4f}")
    print("\ngapped 1-form modes at k -> 0 (omega^2, *2 = 1):")
    for r in (1, 2, 3, 9):
        ev = B.spectrum_T(1e-3 * dirs[0], r)
        print(f"  r={r}: " + " ".join(f"{v:.3f}" for v in ev[ev > 1]))
    rs = np.linspace(1, 9, 161)
    sp_u, sp_T = [], []
    for r in rs:
        res = [B.speed_error(n, r) for n in dirs[::8]]
        sp_u.append(np.ptp([a for a, _ in res]))
        sp_T.append(np.ptp([b for _, b in res]))
    for name, sp in (("u", sp_u), ("T", sp_T)):
        print(f"  most isotropic {name}: r = {rs[int(np.argmin(sp))]:.2f} (spread {min(sp):.2e})")


if __name__ == "__main__":
    main()
