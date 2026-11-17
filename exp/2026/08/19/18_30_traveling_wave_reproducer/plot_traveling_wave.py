import sys

sys.path.insert(0, ".")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from equilibrium.substrate.lattice import DampedLattice, W_HEX, W_SQ

R, HALF_LEN, N_CELLS_PER_WAVELENGTH, ENVELOPE_WAVELENGTHS, AXIS = 12.0, 20.0, 8, 4.0, 2


def build_faces(lat):
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    offsets = lat.pos[lat.nb[c0]] - lat.pos[c0]
    w = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
    return offsets, w


def companion_phi(lat, offsets, w, axis, K, amplitude, envelope_sigma, z0, omega):
    theta = offsets[:, axis] * K
    phi = np.zeros_like(lat.phi)
    for k in range(14):
        Phi_k_amp = (2 * w[k] / omega) * np.sin(0.5 * theta[k])
        mid = lat.pos[:, axis] + 0.5 * offsets[k, axis]
        env = np.exp(-0.5 * ((mid - z0) / envelope_sigma) ** 2)
        phi[:, k] = amplitude * Phi_k_amp * env * np.cos(mid * K)
    return phi


def u_field(lat, axis, K, amplitude, envelope_sigma, z0):
    z = lat.pos[:, axis] - z0
    env = np.exp(-0.5 * (z / envelope_sigma) ** 2)
    return amplitude * env * np.cos(lat.pos[:, axis] * K)


lat = DampedLattice.cylinder(
    R, HALF_LEN, axis=AXIS, sponge_fraction=0.15, damp_max=0.003
)
offsets, w = build_faces(lat)
a_axial = float(abs(offsets[13, AXIS]))
wavelength = N_CELLS_PER_WAVELENGTH * a_axial
K = 2 * np.pi / wavelength
envelope_sigma = ENVELOPE_WAVELENGTHS * wavelength
z0 = 0.0
theta = offsets[:, AXIS] * K
omega = np.sqrt(float(np.sum(w**2 * (1 - np.cos(theta)))))
T = 2 * np.pi / omega
dt = lat.stable_dt()
n_steps = int(round(T / dt))

lat.u = u_field(lat, AXIS, K, 1.0, envelope_sigma, z0)
lat.phi = companion_phi(lat, offsets, w, AXIS, K, 1.0, envelope_sigma, z0, omega)
u0 = lat.u.copy()
phi0_face = lat.phi[:, 13].copy()  # +z sq-face component

pos0 = lat.pos.copy()
for _ in range(n_steps):
    lat.step(dt)
u_T = lat.u.copy()
phi_T_face = lat.phi[:, 13].copy()

r_perp = np.linalg.norm(np.delete(pos0, AXIS, axis=1), axis=1)
z_all = pos0[:, AXIS]
line_mask = r_perp < 0.6  # thin slice near the axis, for a clean 1D read

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
order = np.argsort(z_all[line_mask])
z_line = z_all[line_mask][order]
ax.plot(z_line, u0[line_mask][order], color="C0", lw=1.5, label="u(z, t=0)  [IC]")
ax.plot(
    z_line + wavelength,
    u0[line_mask][order],
    color="C0",
    ls="--",
    lw=1.5,
    label="u(z, t=0) shifted by 1 wavelength  [prediction]",
)
ax.plot(z_line, u_T[line_mask][order], color="C3", lw=1.5, label="u(z, t=T)  [evolved]")
ax.axvspan(-0.5 * HALF_LEN, 0.5 * HALF_LEN, color="gray", alpha=0.05)
ax.set_xlabel("z (axis position, cell units)")
ax.set_ylabel("u")
ax.set_title(
    f"u along the axis (r_perp<0.6) — wavelength={wavelength:.2f}, T={T:.3f}, n_steps={n_steps}"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

ax = axes[0, 1]
ax.plot(
    z_line,
    phi0_face[line_mask][order],
    color="C0",
    lw=1.5,
    label="phi[+z face](z, t=0)  [IC]",
)
ax.plot(
    z_line + wavelength,
    phi0_face[line_mask][order],
    color="C0",
    ls="--",
    lw=1.5,
    label="shifted by 1 wavelength  [prediction]",
)
ax.plot(
    z_line,
    phi_T_face[line_mask][order],
    color="C3",
    lw=1.5,
    label="phi[+z face](z, t=T)  [evolved]",
)
ax.set_xlabel("z (axis position, cell units)")
ax.set_ylabel("phi (+z square face)")
ax.set_title("companion phi (magnetic-like face flux) along the axis")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# 2D slice views (thin |y|<0.5) at t=0 and t=T
slice_mask = np.abs(pos0[:, 1]) < 0.5
x_s, z_s = pos0[slice_mask, 0], pos0[slice_mask, 2]
vmax = np.abs(u0[slice_mask]).max()

ax = axes[1, 0]
sc = ax.scatter(z_s, x_s, c=u0[slice_mask], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=8)
ax.set_xlabel("z")
ax.set_ylabel("x")
ax.set_title("u, x-z slice (|y|<0.5), t=0 [IC]")
plt.colorbar(sc, ax=ax, label="u")

ax = axes[1, 1]
sc = ax.scatter(z_s, x_s, c=u_T[slice_mask], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=8)
ax.set_xlabel("z")
ax.set_ylabel("x")
ax.set_title("u, x-z slice (|y|<0.5), t=T [evolved]")
plt.colorbar(sc, ax=ax, label="u")

plt.suptitle(
    "Traveling-wave reproducer: IC vs evolved state (derived companion-phi construction)"
)
plt.tight_layout()
out = "/tmp/claude-1000/-home-p-git-eq/252c5202-9bb1-41ea-a347-bd036a623abd/scratchpad/traveling_wave_ic_vs_evolved.png"
plt.savefig(out, dpi=140)
print("saved:", out)
