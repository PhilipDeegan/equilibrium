import numpy as np, sys
src = open("gradient.py").read(); src = src[: src.index("ax_ = np.linspace(-np.pi, np.pi, args.nq")]
sys.argv = ["x"]; exec(src)
Y0, om = plane_state(np.zeros(3))
for xi, qs in ((2.6, [-3.792, -1.6494, -1.6494]), (3.9, [-3.769, -1.6724, -1.6724])):
    gg = healing_g(m, xi)
    q = np.array(qs)
    mu = om + gg
    # g = 0 modes in the co-turning frame (phases of Rot(mu) H(q)), at q and -q
    pq = np.angle(np.linalg.eigvals(rot8(mu) @ H_of(np.zeros(3), q)))
    pm = np.angle(np.linalg.eigvals(rot8(mu) @ H_of(np.zeros(3), -q)))
    print(f"xi {xi}: mu {mu:.4f}; mode phases at q: {np.round(np.sort(pq), 4)}")
    print(f"           at -q: {np.round(np.sort(pm), 4)}")
    pairs = sorted(((abs(np.angle(np.exp(1j * (a + b)))), a, b) for a in pq for b in pm))[:3]
    print("   closest q/-q pairs to phase sum 0 (mod 2pi):", [(round(s, 4), round(a, 4), round(b, 4)) for s, a, b in pairs])
    # bare walk frequencies (no co-turning) of those modes
    w = np.angle(np.linalg.eigvals(H_of(np.zeros(3), q)))
    print(f"   bare tick phases at q (-omega): {np.round(np.sort(w), 4)}; omega0 {om:.4f}")
