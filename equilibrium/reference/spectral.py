"""Periodic-box spectral references, numpy / scipy.fft only. Units: c = 1, mu0 = rho = 1 (B in Alfven units).

Box(n, side): cube [-side/2, side/2)^3, n^3 points, centred grid. Fields are real (3, n, n, n) arrays.

Situations (the reference ladder, exp/2026/10/24 in `now`):
  maxwell_vacuum   source-free transverse Maxwell from A0 at E = 0: each mode A_k cos(c k t), exact.
  maxwell_held     constant held current J switched on at t = 0 over A0: A_s + (A0 - A_s) cos(c k t), A_s = J / (c k)^2.
  diffuse          B_t = eta lap B (resistive decay, no flow), exact: B_k exp(-eta k^2 t).
  MHD              incompressible resistive MHD, pseudo-spectral, 2/3 dealiased, RK4 with integrating factor:
                     v_t = P[v x curl v + curl B x B] + nu lap v,   B_t = curl(v x B) + eta lap B.
Seeds: taylor_spheromak(box, a, smooth) = the Taylor field (curl B = lam B, lam a = 4.4934) inside a, zero outside,
smoothed by exp(-k^2 smooth) (the DST seeds' curl-curl smoothing on a div-free field), 2/3 truncated (no Nyquist
modes, which a real field cannot keep div-free), div-free projected."""
import numpy as np
import scipy.fft as sfft
from scipy.special import spherical_jn

LAM_A = 4.4934094579
WORKERS = 4


class Box:
    def __init__(self, n, side):
        self.n, self.side = n, float(side)
        self.dx = self.side / n
        x1 = (np.arange(n) - n // 2) * self.dx
        self.x = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"))
        k1 = 2 * np.pi * sfft.fftfreq(n, d=self.dx)
        kz = 2 * np.pi * sfft.rfftfreq(n, d=self.dx)
        self.k = np.stack(np.meshgrid(k1, k1, kz, indexing="ij"))
        self.k2 = np.sum(self.k**2, 0)
        self.k2_safe = np.where(self.k2 == 0, 1.0, self.k2)
        kmax = np.pi / self.dx
        self.dealias = np.all(np.abs(self.k) < (2.0 / 3.0) * kmax, axis=0)
        self.r = np.sqrt(np.sum(self.x**2, 0))
        self.dV = self.dx**3

    def fft(self, f):
        return sfft.rfftn(f, axes=(-3, -2, -1), workers=WORKERS)

    def ifft(self, fh):
        return sfft.irfftn(fh, s=(self.n,) * 3, axes=(-3, -2, -1), workers=WORKERS)

    def curl_h(self, fh):
        k = self.k
        return 1j * np.stack([k[1] * fh[2] - k[2] * fh[1], k[2] * fh[0] - k[0] * fh[2], k[0] * fh[1] - k[1] * fh[0]])

    def project_h(self, fh):
        return fh - self.k * (np.sum(self.k * fh, 0) / self.k2_safe)[None]

    def curl(self, f):
        return self.ifft(self.curl_h(self.fft(f)))

    def uncurl_h(self, Bh):
        """Coulomb-gauge A with curl A = B (B div-free, zero mean)."""
        return self.curl_h(Bh) / self.k2_safe[None]

    def shells(self, density, edges):
        """sum of density * dV in r bins [edges[i], edges[i+1]), plus beyond the last edge."""
        idx = np.searchsorted(edges, self.r.ravel(), side="right")
        return np.bincount(idx, weights=density.ravel() * self.dV, minlength=len(edges) + 1)


def taylor_spheromak(box, a, smooth=0.0, amp=1.0, axis=2, center=(0.0, 0.0, 0.0), hel=1, R=None):
    """hel = -1: mirror spheromak (toroidal winding flipped, helicity negated, curl B = -lam B).
    R (3x3 rotation): body frame -> box frame, overrides axis (body axis = R[:, 2])."""
    lam = LAM_A / a
    xc = box.x - np.asarray(center, float)[:, None, None, None]
    xc = (xc + box.side / 2) % box.side - box.side / 2
    rr = np.sqrt(np.sum(xc**2, 0))
    if R is not None:
        R = np.asarray(R, float)
        x = np.einsum("ji,j...->i...", R, xc)
        axis = 2
    else:
        x = np.roll(xc, 2 - axis, axis=0)
    r = np.maximum(rr, 1e-9)
    ct = x[2] / r
    rho = np.maximum(np.sqrt(x[0] ** 2 + x[1] ** 2), 1e-12)
    st, cp, sp = rho / r, x[0] / rho, x[1] / rho
    kr = lam * r
    j1, j1p = spherical_jn(1, kr), spherical_jn(1, kr, derivative=True)
    Br, Bt, Bp = 2 * j1 / kr * ct, -(j1 / kr + j1p) * st, hel * j1 * st
    B = np.stack([Br * st * cp + Bt * ct * cp - Bp * sp, Br * st * sp + Bt * ct * sp + Bp * cp, Br * ct - Bt * st])
    B = np.where(rr < a, B, 0.0)
    B = np.einsum("ij,j...->i...", R, B) if R is not None else np.roll(B, axis - 2, axis=0)
    Bh = box.project_h(box.fft(B) * (np.exp(-box.k2 * smooth) * box.dealias)[None])
    Bh[:, 0, 0, 0] = 0.0
    B = box.ifft(Bh)
    return amp * B / np.sqrt(np.sum(B**2, 0)).max()



def rot_x(theta):
    c, s_ = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def beltrami_ambient(box, B0, m=1):
    """Box-scale force-free ambient B0 (0, sin kx, cos kx), k = 2 pi m / side: |B| = B0 everywhere, along +z at x = 0,
    curl B = k B (steady in ideal MHD and, as a flow, in Euler)."""
    k = 2 * np.pi * m / box.side
    x = box.x[0]
    return B0 * np.stack([0 * x, np.sin(k * x), np.cos(k * x)])


def spheromak_pair(box, a, sep, smooth=0.0, amp=1.0, hel2=1, axis=2):
    """Two spheromaks on a common axis, centres +-sep/2 along it, the second with helicity sign hel2 (amp per body)."""
    c = np.zeros(3)
    c[axis] = sep / 2
    return (taylor_spheromak(box, a, smooth, amp, axis, c, 1) + taylor_spheromak(box, a, smooth, amp, axis, -c, hel2))


def energy(box, f):
    return 0.5 * np.sum(f**2) * box.dV


def helicity(box, B):
    Bh = box.fft(B)
    return np.sum(box.ifft(box.uncurl_h(Bh)) * B) * box.dV


def lam_eff(box, B):
    """sum B . curl B / sum B^2 (lam for a Beltrami field)."""
    return np.sum(B * box.curl(B)) / np.sum(B**2)


def maxwell_vacuum(box, A0, t, c=1.0):
    Ah = box.project_h(box.fft(A0))
    w = c * np.sqrt(box.k2)
    A = box.ifft(Ah * np.cos(w * t)[None])
    E = box.ifft(Ah * (w * np.sin(w * t))[None])
    return A, E


def maxwell_held(box, A0, J, t, c=1.0):
    Ah, Jh = box.project_h(box.fft(A0)), box.project_h(box.fft(J))
    w = c * np.sqrt(box.k2)
    As = Jh / (c**2 * box.k2_safe)[None]
    As[:, 0, 0, 0] = 0.0
    A = box.ifft(As + (Ah - As) * np.cos(w * t)[None])
    E = box.ifft((Ah - As) * (w * np.sin(w * t))[None])
    return A, E


def diffuse(box, B0, eta, t):
    return box.ifft(box.fft(B0) * np.exp(-eta * box.k2 * t)[None])


class MHD:
    def __init__(self, box, v, B, nu, eta):
        self.box, self.nu, self.eta = box, nu, eta
        self.vh = box.project_h(box.fft(v)) * box.dealias
        self.Bh = box.project_h(box.fft(B)) * box.dealias
        self.t = 0.0

    @property
    def v(self):
        return self.box.ifft(self.vh)

    @property
    def B(self):
        return self.box.ifft(self.Bh)

    def rhs(self, vh, Bh):
        b = self.box
        v, B = b.ifft(vh), b.ifft(Bh)
        w, J = b.ifft(b.curl_h(vh)), b.ifft(b.curl_h(Bh))
        fv = b.project_h(b.fft(np.cross(v, w, axis=0) + np.cross(J, B, axis=0))) * b.dealias
        fB = b.curl_h(b.fft(np.cross(v, B, axis=0))) * b.dealias
        return fv, fB

    def step(self, dt):
        """RK4 with integrating factor exp(-nu k^2 t) / exp(-eta k^2 t)."""
        k2 = self.box.k2
        ev = lambda s: np.exp(-self.nu * k2 * s)[None]
        eb = lambda s: np.exp(-self.eta * k2 * s)[None]
        v0, B0 = self.vh, self.Bh
        a1, b1 = self.rhs(v0, B0)
        v1, B1 = ev(dt / 2) * (v0 + dt / 2 * a1), eb(dt / 2) * (B0 + dt / 2 * b1)
        a2, b2 = self.rhs(v1, B1)
        v2, B2 = ev(dt / 2) * v0 + dt / 2 * a2, eb(dt / 2) * B0 + dt / 2 * b2
        a3, b3 = self.rhs(v2, B2)
        v3, B3 = ev(dt) * v0 + dt * ev(dt / 2) * a3, eb(dt) * B0 + dt * eb(dt / 2) * b3
        a4, b4 = self.rhs(v3, B3)
        self.vh = ev(dt) * v0 + dt / 6 * (ev(dt) * a1 + 2 * ev(dt / 2) * (a2 + a3) + a4)
        self.Bh = eb(dt) * B0 + dt / 6 * (eb(dt) * b1 + 2 * eb(dt / 2) * (b2 + b3) + b4)
        self.t += dt

    def cfl_dt(self, cfl=0.4):
        vmax = np.sqrt(np.sum(self.v**2, 0)).max() + np.sqrt(np.sum(self.B**2, 0)).max()
        return cfl * self.box.dx / max(vmax, 1e-12)


class VlasovPoisson1D:
    """Electron Vlasov-Poisson on a neutralising ion background, periodic x in [0, Lx), v in [-vmax, vmax), units
    omega_p = lambda_D = 1: f_t + v f_x - E f_v = 0, E_x = 1 - int f dv. Strang splitting, each advection an exact
    Fourier shift (x periodic; v periodic on a domain wide enough that f ~ 0 at the edges). Landau damping reference
    (rung 6, the collisionless offload: wave content -> resonant particles near v = omega / k, filamentation in v)."""

    def __init__(self, nx, nv, Lx, vmax, f0):
        self.nx, self.nv, self.Lx = nx, nv, float(Lx)
        self.x = np.arange(nx) * self.Lx / nx
        self.dv = 2 * vmax / nv
        self.v = -vmax + np.arange(nv) * self.dv
        self.kx = 2 * np.pi * sfft.fftfreq(nx, d=self.Lx / nx)
        self.kv = 2 * np.pi * sfft.fftfreq(nv, d=self.dv)
        self.f = f0(self.x[:, None], self.v[None, :])
        self.t = 0.0

    def efield(self):
        rho = 1.0 - self.f.sum(1) * self.dv
        rh = np.fft.fft(rho)
        Eh = np.where(self.kx == 0, 0.0, rh / (1j * np.where(self.kx == 0, 1.0, self.kx)))
        return np.real(np.fft.ifft(Eh))

    def _shift_x(self, dt):
        self.f = np.real(np.fft.ifft(np.fft.fft(self.f, axis=0) * np.exp(-1j * self.kx[:, None] * self.v[None, :] * dt), axis=0))

    def _shift_v(self, dt):
        a = -self.efield()
        self.f = np.real(np.fft.ifft(np.fft.fft(self.f, axis=1) * np.exp(-1j * self.kv[None, :] * a[:, None] * dt), axis=1))

    def step(self, dt):
        self._shift_x(dt / 2)
        self._shift_v(dt)
        self._shift_x(dt / 2)
        self.t += dt

    def field_energy(self):
        return 0.5 * np.sum(self.efield() ** 2) * self.Lx / self.nx

    def kinetic_energy(self):
        return 0.5 * np.sum(self.f * self.v[None, :] ** 2) * self.dv * self.Lx / self.nx


def landau_f0(k, alpha):
    return lambda x, v: (1 + alpha * np.cos(k * x)) * np.exp(-0.5 * v**2) / np.sqrt(2 * np.pi)
