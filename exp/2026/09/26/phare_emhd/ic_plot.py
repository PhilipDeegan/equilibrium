"""Meridional (y = mid) cut of PHARE's t=0 B for emhd_spheromak_p0: B_phi (= B_y here),
B_z, |B|, and poloidal field lines (Bx, Bz) with psi(rho, z) = int_0^rho B_z rho' drho' contours."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = np.load("B_steps.npy")[0]
g = np.arange(65) * 0.125 - 4.0
m = 32
bx, by, bz = (B[:, m, :, c] for c in range(3))
Bm = np.linalg.norm(B[:, m, :, :], axis=-1)
rho = np.abs(g)
psi = np.zeros_like(bz)
for k in range(65):
    for side in (slice(m, 65), slice(m, None, -1)):
        prof = bz[side, k] * rho[side]
        psi[side, k] = np.concatenate([[0], np.cumsum(0.5 * (prof[1:] + prof[:-1]) * 0.125)])
fig, ax = plt.subplots(2, 2, figsize=(12, 11))
ext = [g[0], g[-1], g[0], g[-1]]
for a, v, name, cm in ((ax[0, 0], by, "B_phi (B_y at y=0)", "RdBu_r"), (ax[0, 1], bz, "B_z", "RdBu_r"),
                        (ax[1, 0], Bm, "|B|", "viridis")):
    lim = float(np.abs(v).max())
    im = a.imshow(v.T, origin="lower", extent=ext, cmap=cm, vmin=0 if cm == "viridis" else -lim, vmax=lim)
    fig.colorbar(im, ax=a, shrink=0.8)
    a.add_patch(plt.Circle((0, 0), 1.5, fill=False, ls=":", color="k"))
    a.set_title(name); a.set_xlabel("x"); a.set_ylabel("z")
a = ax[1, 1]
a.contour(g, g, psi.T, levels=np.linspace(psi.min(), psi.max(), 30), colors="k", linewidths=0.6)
a.imshow(by.T, origin="lower", extent=ext, cmap="RdBu_r", alpha=0.5, vmin=-0.45, vmax=0.45)
a.add_patch(plt.Circle((0, 0), 1.5, fill=False, ls=":", color="g"))
a.set_title("poloidal flux psi contours (field lines) over B_phi")
a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("z")
fig.suptitle("PHARE emhd_spheromak_p0, t=0, meridional cut")
fig.tight_layout()
fig.savefig("ic_meridional.png", dpi=80)
