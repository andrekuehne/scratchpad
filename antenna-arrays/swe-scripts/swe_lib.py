"""Minimal SWE helpers used by the check scripts.

Conventions follow ../swe-compression-embedded-element-patterns.md, except
that the 1/sqrt(2*pi) factor of the phi-part is dropped from G_m: with
Clenshaw-Curtis weights W, G_m^H W G_m = I exactly, and
int |K|^2 dOmega = 2*pi * sum |x|^2.
"""
import numpy as np


def theta_tables(th, L):
    """Fully normalised Theta_n^m (no Condon-Shortley phase), m, n = 0..L+1.

    Returns T[n, m, i] with int_0^pi |Theta_n^m|^2 sin(theta) dtheta = 1.
    """
    x, s = np.cos(th), np.sin(th)
    M = L + 2
    T = np.zeros((M, M, th.size))
    T[0, 0] = 1/np.sqrt(2)
    for m in range(1, M):
        T[m, m] = np.sqrt((2*m+1)/(2*m)) * s * T[m-1, m-1]
    for m in range(M):
        if m+1 < M:
            T[m+1, m] = np.sqrt(2*m+3) * x * T[m, m]
        for n in range(m+2, M):
            a = np.sqrt((4*n*n-1)/(n*n-m*m))
            b = np.sqrt(((n-1)**2-m*m)/(4*(n-1)**2-1))
            T[n, m] = a*(x*T[n-1, m] - b*T[n-2, m])
    return T


def AB(th, L, m):
    """A_nm, B_nm for n = max(1,m)..L (m >= 0). Poles via a tiny offset."""
    thc = np.clip(th, 1e-7, np.pi-1e-7)
    T = theta_tables(thc, L)
    ns = np.arange(max(1, m), L+1)
    A = np.zeros((ns.size, th.size)); B = np.zeros_like(A)
    for k, n in enumerate(ns):
        nu = np.sqrt(n*(n+1))
        A[k] = m*T[n, m]/np.sin(thc)/nu
        if m == 0:
            d = -np.sqrt(n*(n+1))*T[n, 1]
        else:
            d = 0.5*(np.sqrt((n+m)*(n-m+1))*T[n, m-1] - np.sqrt((n-m)*(n+m+1))*T[n, m+1])
        B[k] = d/nu
    return A, B


def G_of(th, L, m):
    """Design matrix G_m (2*Ntheta x 2*N_m): columns [TE a_nm | TM b_nm]."""
    A, B = AB(th, L, abs(m))
    A = np.sign(m)*A if m != 0 else A
    top = np.hstack([1j*A.T, B.T]); bot = np.hstack([-B.T, 1j*A.T])
    return np.vstack([top, bot])


def cc_weights(N):
    """Clenshaw-Curtis weights for x = cos(j*pi/N), j = 0..N (integrate dx on [-1, 1])."""
    th = np.arange(N+1)*np.pi/N
    w = np.zeros(N+1)
    for j in range(N+1):
        s = 0.0
        for k in range(1, N//2+1):
            bk = 1.0 if 2*k == N else 2.0
            s += bk*np.cos(2*k*th[j])/(4*k*k-1)
        cj = 1.0 if j in (0, N) else 2.0
        w[j] = cj/N*(1 - s)
    return w


class Grid:
    """Regular full-sphere grid: theta = 0..180 deg (N+1 points), phi = 0..360 deg (Nphi points)."""

    def __init__(self, N=180, Nphi=360):
        self.N, self.Nphi = N, Nphi
        self.th = np.arange(N+1)*np.pi/N
        self.ph = np.arange(Nphi)*2*np.pi/Nphi
        self.w_cc = cc_weights(N)
        self.TH, self.PH = np.meshgrid(self.th, self.ph, indexing="ij")
        TH, PH = self.TH, self.PH
        self.rhat = np.stack([np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)], -1)
        self.tvec = np.stack([np.cos(TH)*np.cos(PH), np.cos(TH)*np.sin(PH), -np.sin(TH)], -1)
        self.pvec = np.stack([-np.sin(PH), np.cos(PH), np.zeros_like(PH)], -1)
        self.dOmega = self.w_cc[:, None]*2*np.pi/Nphi * np.ones_like(PH)
        self._G = {}

    def G(self, L, m):
        if (L, m) not in self._G:
            self._G[(L, m)] = G_of(self.th, L, m)
        return self._G[(L, m)]

    def fit(self, Kt, Kp, L):
        """Kt, Kp: (Ntheta, Nphi, E) -> {m: x_m (2*N_m, E)}, CC quadrature projection."""
        Ct = np.fft.fft(Kt, axis=1)/self.Nphi
        Cp = np.fft.fft(Kp, axis=1)/self.Nphi
        W = np.concatenate([self.w_cc, self.w_cc])[:, None]
        return {m: self.G(L, m).conj().T @ (W*np.concatenate([Ct[:, m % self.Nphi], Cp[:, m % self.Nphi]], 0))
                for m in range(-L, L+1)}

    def synth(self, x, L):
        E = next(iter(x.values())).shape[1]
        Ct = np.zeros((self.N+1, self.Nphi, E), complex); Cp = np.zeros_like(Ct)
        for m, xm in x.items():
            c = self.G(L, m) @ xm
            Ct[:, m % self.Nphi], Cp[:, m % self.Nphi] = c[:self.N+1], c[self.N+1:]
        return np.fft.ifft(Ct, axis=1)*self.Nphi, np.fft.ifft(Cp, axis=1)*self.Nphi


def spectrum(x, L):
    """S[n, e] = sum_m |a_nm|^2 + |b_nm|^2."""
    E = next(iter(x.values())).shape[1]
    S = np.zeros((L+1, E))
    for m, xm in x.items():
        n0 = max(1, abs(m)); Nm = L-n0+1
        S[n0:] += abs(xm[:Nm])**2 + abs(xm[Nm:])**2
    return S


def recentre_closed_form(grid, Kt, Kp, L, k):
    """Closed-form global origin r0 = M^-1 n / k (step 0a), summed over elements.

    One fit -> D = div_s K, C = curl_s K from the coefficients -> 3x3 solve.
    """
    x = grid.fit(Kt, Kp, L)
    N, Nphi, E = grid.N, grid.Nphi, Kt.shape[2]
    T = theta_tables(np.clip(grid.th, 1e-7, np.pi-1e-7), L)
    Dm = np.zeros((N+1, Nphi, E), complex); Cm = np.zeros_like(Dm)
    for m, xm in x.items():
        ns = np.arange(max(1, abs(m)), L+1); Nm = ns.size
        sq = np.sqrt(ns*(ns+1))[:, None]
        a, b = xm[:Nm], xm[Nm:]
        Th = T[ns, abs(m)]
        Dm[:, m % Nphi] = Th.T @ (-sq*b)          # div_s  X = 0,  div_s Z = -sqrt(nu) Y
        Cm[:, m % Nphi] = Th.T @ (sq*a)           # curl_s X = sqrt(nu) Y, curl_s Z = 0
    D = np.fft.ifft(Dm, axis=1)*Nphi
    C = np.fft.ifft(Cm, axis=1)*Nphi
    tv, pv = grid.tvec[:, :, None], grid.pvec[:, :, None]
    Kc = Kt[..., None]*tv + Kp[..., None]*pv      # K       (Cartesian)
    Vc = Kt[..., None]*pv - Kp[..., None]*tv      # r x K   (Cartesian)
    dO = grid.dOmega[:, :, None]
    K2 = abs(Kt)**2 + abs(Kp)**2
    P = np.sum(dO*K2, axis=(0, 1))                # per-element normalisation
    Pt = np.eye(3) - grid.rhat[..., :, None]*grid.rhat[..., None, :]
    M = np.einsum("ije,ijab,ij->ab", K2/P, Pt, grid.dOmega)
    n = np.einsum("ije,ijec->c", dO/P, np.imag(np.conj(C)[..., None]*Vc - np.conj(D)[..., None]*Kc))
    return np.linalg.solve(M, n)/k
