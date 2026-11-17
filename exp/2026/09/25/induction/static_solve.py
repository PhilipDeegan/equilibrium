"""Discrete static target for induction.py: solve C0SQ * curl_curl_sym(A) = J
by CG (op symmetric PSD on the antisymmetric anchors, null space = anchor
gradients + null(K)). J in range(K^T) (--src mag) converges; a J with a
null(K) part (--src sampled) stalls at a residual floor -- that floor is the
part no static A can balance. Reports Phi2/M, Phi1 of the solution."""
import os
import sys

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

sys.argv += [] if "--src" in sys.argv else ["--src", "mag"]
exec(open("induction.py").read().split("def holonomy")[0])


def holonomy(A, wt, a):
    return 2 * np.pi * a * float(((A * phat).sum(1) * wt).sum() / wt.sum())


k2 = 4 * args.a1 * args.a2 / ((args.a1 + args.a2) ** 2 + args.d**2)
k = np.sqrt(k2)
M = (1 / C.C0SQ) * np.sqrt(args.a1 * args.a2) * ((2 / k - k) * ellipk(k2) - (2 / k) * ellipe(k2))
valid = lat.nb >= 0
shape = lat.phi.shape


def op(v):
    phi = v.reshape(shape)
    return (C.C0SQ * (lat.curl_curl_sym(phi) if args.cpp else C.curl_curl_sym(lat, phi))).ravel()


L = LinearOperator((valid.size, valid.size), matvec=op, dtype=np.float64)
b = np.where(valid, Jf, 0.0).ravel()
it = [0]
hist = []


def cb(xk):
    it[0] += 1
    if it[0] % 25 == 0:
        r = np.linalg.norm(op(xk) - b) / np.linalg.norm(b)
        hist.append((it[0], r))
        print(f"  it={it[0]} rel_res={r:.3e}", flush=True)


x, info = cg(L, b, rtol=1e-8, atol=0.0, maxiter=int(os.environ.get("MAXITER", 1500)), callback=cb)
A = C.reconstruct_A(lat, x.reshape(shape))
r = np.linalg.norm(op(x) - b) / np.linalg.norm(b)
print(f"src={args.src} h={args.h} R={args.R} info={info} it={it[0]} rel_res={r:.3e}")
print(f"static Phi2/M = {holonomy(A, w2, args.a2)/M:.4f}  Phi1 = {holonomy(A, w1, args.a1):.4e}  M={M:.5e}")
np.savez(f"static_{args.src}_R{args.R / args.h:g}_h{args.h}{'_cpp' if args.cpp else ''}.npz", phi=x.reshape(shape), hist=np.array(hist))
