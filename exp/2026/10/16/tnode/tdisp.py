"""10-16 tnode: transverse junction on the 14-face cell. Two amplitudes per face (polarisations e_k1, e_k2 perp to d_k,
shared by k and mirror(k) so a pulse keeps its polarisation across the face). Normalised amplitudes (energy sum a^2).
b = (2P - 1) a, P = projector onto span(sqrt(w_k) [e_kp, s (d_k x e_kp)]) (6 node quantities: 3 E-like, 3 H-like).
s = +-1 picks the E-H handedness. Bloch dispersion: count travelling branches, speed, isotropy, polarisation."""
import numpy as np

from equilibrium.substrate import continuous as C

D, MIR = C.D_HAT, C.K_MIRROR
DH = D / np.linalg.norm(D, axis=1)[:, None]
w = np.asarray(C.LAP_W, float)
w = w / w.sum()
E = np.zeros((14, 2, 3))
for k in range(14):
    if k > MIR[k]:
        continue
    n = DH[k]
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = t - (t @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    E[k] = E[MIR[k]] = (e1, e2)


def S_of(s):
    M = np.zeros((28, 6))
    for k in range(14):
        for p in range(2):
            M[2 * k + p, :3] = np.sqrt(w[k]) * E[k, p]
            M[2 * k + p, 3:] = s * np.sqrt(w[k]) * np.cross(DH[k], E[k, p])
    Q, _ = np.linalg.qr(M)
    return 2 * Q @ Q.T - np.eye(28), M


def T(q, S):
    P = np.zeros((28, 28), complex)
    for k in range(14):
        ph = np.exp(-1j * q @ D[k])
        for p in range(2):
            P[2 * MIR[k] + p, 2 * k + p] = ph
    return P @ S


if __name__ == "__main__":
    for s in (1, -1):
        S, M = S_of(s)
        print(f"s {s}: S orthogonal {np.abs(S @ S.T - np.eye(28)).max():.1e}, symmetric {np.abs(S - S.T).max():.1e}")
        for name, dv in (("[100]", np.array([1, 0, 0.])), ("[111]", np.ones(3) / np.sqrt(3)), ("[110]", np.array([1, 1, 0.]) / np.sqrt(2)), ("[123]", np.array([1, 2, 3.]) / np.sqrt(14))):
            out = []
            for qm in (0.02, 0.1, 0.5):
                q = qm * dv
                ev, vec = np.linalg.eig(T(q, S))
                om = np.angle(ev)
                mv = (np.abs(om) > 1e-6) & (np.abs(om) < 0.5 * np.pi)
                sp = np.sort(np.abs(om[mv]) / qm)
                pol = []
                for i in np.where(mv & (om > 0))[0]:
                    f = np.linalg.lstsq(M, vec[:, i], rcond=None)[0]
                    Ef = f[:3]
                    pol.append(abs(Ef @ dv) / max(np.linalg.norm(Ef), 1e-30))
                out.append(f"q{qm}: n_move {mv.sum()} speeds {' '.join(f'{v:.3f}' for v in sp[::2])} |E.q|/|E| {' '.join(f'{v:.2f}' for v in sorted(pol))}")
            nzero = np.sum(np.abs(np.angle(np.linalg.eigvals(T(0.02 * dv, S)))) < 1e-6)
            npi = np.sum(np.abs(np.abs(np.angle(np.linalg.eigvals(T(0.02 * dv, S)))) - np.pi) < 1e-6)
            print(f"  {name} static {nzero} flip {npi} | " + " | ".join(out))
