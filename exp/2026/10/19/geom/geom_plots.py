"""10-19 geometry figures for "what the integer counts": BCC cell, element types, elementary rings around a hop,
a structure-scale ring as a sum of elementary rings (computed), rotation measured in the zoo (coil precession).
Units: cubic a = 1 (<111> neighbour at sqrt(3)/2, <100> at 1; code units scale by 2 / sqrt(3))."""
import itertools
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

OUT = Path(__file__).parent
N111 = np.array([s for s in itertools.product((-0.5, 0.5), repeat=3)])
N100 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
VV = np.array(sorted({tuple(p) for b in itertools.permutations((0, 0.25, 0.5))
                      for s in itertools.product((-1, 1), repeat=3) for p in [np.array(b) * s]}))


def face(n):
    on = VV[np.isclose(VV @ n, np.max(VV @ n))]
    c = on.mean(0)
    u = on[0] - c
    u /= np.linalg.norm(u)
    w = np.cross(n / np.linalg.norm(n), u)
    return on[np.argsort(np.arctan2((on - c) @ w, (on - c) @ u))]


def setax(ax, lim=0.8):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


HEX, SQ, TRI, RING = "#d1495b", "#2e86ab", "#edae49", "#3b3b3b"

# fig1: the cell
fig = plt.figure(figsize=(11, 5.2))
ax = fig.add_subplot(121, projection="3d")
for n in N111:
    ax.add_collection3d(Poly3DCollection([face(n)], facecolor=HEX, alpha=0.35, edgecolor="k", lw=0.6))
for n in N100:
    ax.add_collection3d(Poly3DCollection([face(n)], facecolor=SQ, alpha=0.35, edgecolor="k", lw=0.6))
setax(ax, 0.6)
ax.set_title("cell = truncated octahedron\n8 hex faces (<111>) + 6 squares (<100>)")
ax = fig.add_subplot(122, projection="3d")
ax.scatter([0], [0], [0], c="k", s=40)
for n in N111:
    ax.plot(*np.stack([0 * n, n]).T, c=HEX, lw=2)
    ax.scatter(*n, c=HEX, s=25)
for n in N100:
    ax.plot(*np.stack([0 * n, n]).T, c=SQ, lw=2)
    ax.scatter(*n, c=SQ, s=25)
setax(ax, 1.0)
ax.set_title("14 relations (hops): 8 at sqrt(3)/2, 6 at 1\neach shared by 2 cells -> 7 per cell")
fig.tight_layout()
fig.savefig(OUT / "fig1_cell.png", dpi=130)
plt.close(fig)

# fig2: element types
o, x1, e1, e2 = np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, -0.5])
fig = plt.figure(figsize=(14, 4))
items = [("cell (vertex)\n1 per cell", [o], []),
         ("hop (edge)\n7 per cell: 4 <111> + 3 <100>", [o, e1], [(o, e1)]),
         ("triangle\n12 per cell: 2 <111> + 1 <100> edge", [o, x1, e1], [(o, x1), (x1, e1), (e1, o)]),
         ("tetrahedron\n6 per cell: 4 <111> + 2 <100> edges", [o, x1, e1, e2],
          [(o, x1), (e1, e2), (o, e1), (o, e2), (x1, e1), (x1, e2)])]
for i, (tt, P, E) in enumerate(items):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    P = np.array(P)
    ax.scatter(*P.T, c="k", s=40)
    for a, b in E:
        c = SQ if np.isclose(np.linalg.norm(b - a), 1) else HEX
        ax.plot(*np.stack([a, b]).T, c=c, lw=2.5)
    if len(P) == 3:
        ax.add_collection3d(Poly3DCollection([P], facecolor=TRI, alpha=0.5))
    if len(P) == 4:
        ax.add_collection3d(Poly3DCollection([P[[0, 1, 2]], P[[0, 1, 3]], P[[0, 2, 3]], P[[1, 2, 3]]],
                                             facecolor=TRI, alpha=0.25))
    ax.set_xlim(-0.2, 1.1); ax.set_ylim(-0.6, 0.7); ax.set_zlim(-0.6, 0.7)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title(tt, fontsize=10)
fig.suptitle("element types (1 - 7 + 12 - 6 = 0); red = <111>, blue = <100>", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig2_elements.png", dpi=130)
plt.close(fig)


# fig3: elementary rings around a hop (fan of triangles = edges of the dual face)
def fan(h):
    cand = np.concatenate([N111, N100])
    out = []
    for v in cand:
        a, b = np.linalg.norm(v), np.linalg.norm(v - h)
        if np.isclose(a, 0) or np.isclose(b, 0):
            continue
        lens = sorted([np.linalg.norm(h), a, b])
        if np.allclose(lens, [np.sqrt(3) / 2, np.sqrt(3) / 2, 1]):
            out.append(v)
    return np.array(out)


def to_z(h):
    a = h / np.linalg.norm(h)
    v = np.cross(a, [0, 0, 1.0])
    sn, cs = np.linalg.norm(v), a[2]
    if sn < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / sn
    th = np.arctan2(sn, cs)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


fig = plt.figure(figsize=(11, 5.6))
for i, (h, nm, col) in enumerate([(e1, "<111> hop: 6 triangles, hex face", HEX), (x1, "<100> hop: 4 triangles, square face", SQ)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    Rz = to_z(h)
    T = lambda P: (np.atleast_2d(P) - h / 2) @ Rz.T
    F = fan(h)
    ax.plot(*T(np.stack([o, h])).T, c=col, lw=4)
    for v in F:
        ax.add_collection3d(Poly3DCollection([T(np.stack([o, h, v]))], facecolor=TRI, alpha=0.3, edgecolor="grey", lw=0.6))
        ax.scatter(*T(v).T, c="k", s=12)
    fc = T(face(h / np.linalg.norm(h) * (1 if i == 0 else 1)))
    for k in range(len(fc)):
        a, b = fc[k], fc[(k + 1) % len(fc)]
        ax.quiver(*a, *(b - a), color=RING, lw=2.2, arrow_length_ratio=0.2)
        m = (a + b) / 2
        ax.text(*(1.25 * m), str(k + 1), fontsize=10)
    setax(ax, 0.75)
    ax.view_init(30, -60)
    ax.set_title(f"{nm}\none count on the hop -> b = 1 on each of its {len(F)} triangles\n"
                 f"= closed ring of circulation round the hop, {len(F)} steps per turn", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_elementary_rings.png", dpi=130)
plt.close(fig)

# fig4: disk of hop counts -> integer ring on the rim (Stokes on the lattice, computed)
L, Rd, z0 = 7, 4.2, 0.13
pts = np.array([p for p in itertools.product(np.arange(-L, L + 1), repeat=3)] +
               [np.array(p) + 0.5 for p in itertools.product(np.arange(-L, L), repeat=3)], float)
idx = {tuple(np.round(2 * p).astype(int)): i for i, p in enumerate(pts)}
HOPS = np.array([h for h in np.concatenate([N111, N100]) if tuple(h) > tuple(-h)])
e = {}
for i, p in enumerate(pts):
    for h in HOPS:
        q = p + h
        if tuple(np.round(2 * q).astype(int)) not in idx:
            continue
        lo, hi = min(p[2], q[2]), max(p[2], q[2])
        m = (p + q) / 2
        if lo < z0 < hi and np.hypot(m[0], m[1]) < Rd:
            e[(i, idx[tuple(np.round(2 * q).astype(int))])] = np.sign(q[2] - p[2])


def eget(a, b):
    return e.get((a, b), 0) - e.get((b, a), 0)


bt = {}
for i, p in enumerate(pts):
    if np.abs(p).max() > L - 1.5:
        continue
    for h in N100[[0, 2, 4]]:
        for v in fan(h):
            a, b, c = i, idx.get(tuple(np.round(2 * (p + h)).astype(int))), idx.get(tuple(np.round(2 * (p + v)).astype(int)))
            if b is None or c is None:
                continue
            key = tuple(sorted((a, b, c)))
            if key in bt:
                continue
            s = eget(a, b) + eget(b, c) + eget(c, a)
            if s != 0:
                bt[key] = (s, pts[[a, b, c]])
vals = np.array([s for s, _ in bt.values()])
cents = np.array([P.mean(0) for _, P in bt.values()])
rad = np.hypot(cents[:, 0], cents[:, 1])
fig = plt.figure(figsize=(11, 5.2))
ax = fig.add_subplot(121, projection="3d")
mids = np.array([(pts[a] + pts[b]) / 2 for (a, b) in e])
ax.scatter(*mids.T, s=4, c="#999999", alpha=0.5, label=f"{len(e)} hops with count 1 (crossing the disk)")
ax.add_collection3d(Poly3DCollection([P for s, P in bt.values()], facecolor=HEX, alpha=0.8, edgecolor="k", lw=0.3))
setax(ax, 5.5)
ax.view_init(28, -60)
ax.set_title(f"hop counts on a disk R {Rd} -> nonzero b only on the rim\n{len(bt)} triangles, |b| = {sorted(set(np.abs(vals).tolist()))} (sign = triangle orientation)",
             fontsize=10)
ax.legend(loc="lower left", fontsize=8)
ax = fig.add_subplot(122)
ax.hist(rad, bins=30, color=TRI, edgecolor="k")
ax.axvline(Rd, c="k", ls="--", label=f"disk radius {Rd}")
ax.set_xlabel("radius of triangles with b != 0")
ax.set_ylabel("count")
ax.set_title("interior fans cancel pairwise (curl of a step = 0);\nthe whole turn lives on the rim, integer per triangle", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig4_ring_from_hops.png", dpi=130)
plt.close(fig)
print(f"fig4: {len(e)} hops, {len(bt)} rim triangles, b values {sorted(set(vals.tolist()))}, radius {rad.min():.2f}-{rad.max():.2f}")

# fig5: rotation -- torus knots (two integer windings) + measured zoo precession
s = np.linspace(0, 2 * np.pi, 800)


def tk(p, q, r1=1.0, r2=0.35):
    rr = r1 + r2 * np.cos(q * s)
    return np.stack([rr * np.cos(p * s), rr * np.sin(p * s), r2 * np.sin(q * s)], 1)


fig = plt.figure(figsize=(15, 4.6))
for i, (p, q, tt) in enumerate([(1, 2, "(1, 2): poloidal at twice the toroidal rate\n(dotted = torus core)"),
                                (1, 3, "coil_1_3 (zoo)"), (3, 2, "coil_3_2 (zoo)")]):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    c = tk(p, q)
    seg = np.stack([c[:-1], c[1:]], 1)
    lc = Line3DCollection(seg, cmap="viridis", lw=2.5)
    lc.set_array(s[:-1])
    ax.add_collection3d(lc)
    ax.plot(np.cos(s), np.sin(s), 0 * s, c="grey", lw=1, ls=":")
    setax(ax, 1.3)
    ax.view_init(62, 20)
    ax.set_title(f"{tt}\nwhole turns: toroidal {p}, poloidal {q}", fontsize=9.5)
ax = fig.add_subplot(144)
zoo = Path(__file__).parents[1] / "zoo" / "r20"
for nm in ["coil_1_3", "coil_3_2", "concentric", "pair_counter", "ring", "pair_co", "trefoil", "hopf_link"]:
    f = zoo / f"{nm}.txt"
    if not f.exists():
        continue
    tr = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r"t\s+([\d.]+)\s+disp.*?rot\s+([\d.]+)", f.read_text())]
    if tr:
        tr = np.array(tr)
        ax.plot(tr[:, 0], tr[:, 1], "o-", ms=3, label=nm)
ax.set_xlabel("t")
ax.set_ylabel("fitted rigid rotation (deg)")
ax.set_title("zoo R 20: rotation that emerges\n(step() + Cartan, nothing imposed)", fontsize=9.5)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig5_rotation.png", dpi=130)
plt.close(fig)
