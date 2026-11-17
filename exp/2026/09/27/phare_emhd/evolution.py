"""History of PHARE emhd_spheromak_p0 (hybrid, frozen-ish ions): W_B, force-free residual, max|Vi|,
J-weighted centroid (drift), magnetic moment direction (tilt), plus meridional/equatorial cuts."""
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "emhd_spheromak_p0"
d = Path("/home/p/git/phare/mkn/phare_outputs/test") / tag
out = Path(__file__).parent / f"{tag}_plots"
out.mkdir(parents=True, exist_ok=True)
N, h, R = 65, 0.125, 1.5
g = np.arange(N) * h - 4.0
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
dV = h**3


def reader(name):
    f = h5py.File(d / f"{name}.vtkhdf", "r")
    L0, S = f["VTKHDF/Level0"], f["VTKHDF/Steps"]
    boxes, nbox = L0["AMRBox"][:], S["Level0/NumberOfAMRBox"][:]
    boff, poff = S["Level0/AMRBoxOffset"][:], S["Level0/PointDataOffset/data"][:]
    times, data = S["Values"][:], L0["PointData/data"]
    ncomp = 1 if data.ndim == 1 else data.shape[1]

    def load(s):
        A = np.full((N, N, N, ncomp), np.nan)
        p = poff[s]
        for b in boxes[boff[s]:boff[s] + nbox[s]]:
            lo, hi = b[0::2], b[1::2]
            sh = hi - lo + 2
            n = int(np.prod(sh))
            blk = np.asarray(data[p:p + n]).reshape(sh[2], sh[1], sh[0], ncomp).transpose(2, 1, 0, 3)
            A[lo[0]:lo[0] + sh[0], lo[1]:lo[1] + sh[1], lo[2]:lo[2] + sh[2]] = blk
            p += n
        return A

    return times, load


tB, loadB = reader("EM_B")
tV, loadV = reader("ions_bulkVelocity")
last = {float(t): s for s, t in enumerate(tB)}
steps = [last[t] for t in sorted(last)]
vstep = {float(t): s for s, t in enumerate(tV)}
print(f"{len(tB)} B steps, {len(steps)} unique times {min(last)}..{max(last)}")

hist = {k: [] for k in ("t", "WB", "fxb", "vmax", "cx", "cy", "cz", "tilt", "mabs", "Jint")}
cut_times = set(np.linspace(0, len(steps) - 1, 6).round().astype(int))
cuts = []
for i, s in enumerate(steps):
    t = float(tB[s])
    B = loadB(s)
    bx, by, bz = B[..., 0], B[..., 1], B[..., 2]
    jx = np.gradient(bz, h, axis=1) - np.gradient(by, h, axis=2)
    jy = np.gradient(bx, h, axis=2) - np.gradient(bz, h, axis=0)
    jz = np.gradient(by, h, axis=0) - np.gradient(bx, h, axis=1)
    J2 = jx**2 + jy**2 + jz**2
    B2 = bx**2 + by**2 + bz**2
    fx, fy, fz = jy * bz - jz * by, jz * bx - jx * bz, jx * by - jy * bx
    w = J2 / J2.sum()
    c = np.array([(w * X).sum(), (w * Y).sum(), (w * Z).sum()])
    rx, ry, rz = X - c[0], Y - c[1], Z - c[2]
    m = 0.5 * dV * np.array([(ry * jz - rz * jy).sum(), (rz * jx - rx * jz).sum(), (rx * jy - ry * jx).sum()])
    hist["t"].append(t)
    hist["WB"].append(0.5 * B2.sum() * dV)
    hist["fxb"].append(np.sqrt((fx**2 + fy**2 + fz**2).sum() / (J2 * B2).sum()))
    hist["cx"].append(c[0]); hist["cy"].append(c[1]); hist["cz"].append(c[2])
    hist["mabs"].append(np.linalg.norm(m))
    hist["tilt"].append(np.degrees(np.arccos(np.clip(m[2] / np.linalg.norm(m), -1, 1))))
    hist["Jint"].append(np.sqrt(J2.sum() * dV))
    vmax = np.nan
    if t in vstep:
        V = loadV(vstep[t])
        vmax = float(np.nanmax(np.linalg.norm(V, axis=-1)))
    hist["vmax"].append(vmax)
    print(f"t={t:6.2f} WB={hist['WB'][-1]:.5f} fxb={hist['fxb'][-1]:.4f} c=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) "
          f"tilt={hist['tilt'][-1]:6.2f} |m|={hist['mabs'][-1]:.4f} vmax={vmax:.2e}", flush=True)
    if i in cut_times:
        cuts.append((t, by[:, N // 2, :].copy(), bz[:, N // 2, :].copy(), np.sqrt(B2[:, :, N // 2]),
                     np.sqrt(J2[:, N // 2, :])))

np.savez(out / "history.npz", **{k: np.array(v) for k, v in hist.items()})
H = {k: np.array(v) for k, v in hist.items()}
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0, 0].plot(H["t"], H["WB"] / H["WB"][0]); ax[0, 0].set_title("W_B / W_B(0)")
ax[0, 1].semilogy(H["t"], H["fxb"]); ax[0, 1].set_title("|JxB| / (|J||B|)  (rms-weighted)")
ax[0, 2].semilogy(H["t"], H["vmax"]); ax[0, 2].set_title("max |Vi|")
for k in ("cx", "cy", "cz"):
    ax[1, 0].plot(H["t"], H[k], label=k)
ax[1, 0].legend(); ax[1, 0].set_title("|J|^2-weighted centroid (box centre 0)")
ax[1, 1].plot(H["t"], H["tilt"]); ax[1, 1].set_title("moment angle from +z [deg]")
ax[1, 2].plot(H["t"], H["mabs"] / H["mabs"][0], label="|m|")
ax[1, 2].plot(H["t"], H["Jint"] / H["Jint"][0], label="||J||"); ax[1, 2].legend(); ax[1, 2].set_title("relative to t=0")
for a in ax.flat:
    a.set_xlabel("t")
fig.suptitle(tag)
fig.tight_layout()
fig.savefig(out / "history.png", dpi=80)

ext = [g[0], g[-1], g[0], g[-1]]
fig, ax = plt.subplots(4, len(cuts), figsize=(3.2 * len(cuts), 12.5), squeeze=False)
lims = [max(np.abs(cc[k]).max() for cc in cuts) for k in range(1, 5)]
names = ["B_y (y=0, x-z)", "B_z (y=0, x-z)", "|B| (z=0, x-y)", "|J| (y=0, x-z)"]
for j, (t, *fields) in enumerate(cuts):
    for k, v in enumerate(fields):
        a = ax[k, j]
        cm = "RdBu_r" if k < 2 else "viridis"
        a.imshow(v.T, origin="lower", extent=ext, cmap=cm, vmin=-lims[k] if k < 2 else 0, vmax=lims[k])
        a.add_patch(plt.Circle((0, 0), R, fill=False, ls=":", color="k"))
        a.set_title(f"{names[k]}  t={t:.1f}", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(out / "cuts.png", dpi=80)
print("wrote", out)
