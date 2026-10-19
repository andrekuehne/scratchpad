"""Step 10: minimum measurement grids for a band limit L.

- Gram matrix on equiangular rings (Clenshaw-Curtis, N = 2L) and on
  Gauss-Legendre rings (L+1 and L) with Nphi = 2L+1.
- Reconstruction test: band-limited random pattern sampled on the
  Gauss-Legendre grid, fitted, resynthesised on the 1 deg grid.
"""
import numpy as np
from swe_lib import G_of, cc_weights, Grid

L = 28


def gram_dev(th, w):
    W = np.concatenate([w, w])[:, None]
    dev = 0.0
    for m in range(-L, L+1):
        G = G_of(th, L, m)
        dev = max(dev, np.max(abs(G.conj().T @ (W*G) - np.eye(G.shape[1]))))
    return dev


N = 2*L
th_cc = np.arange(N+1)*np.pi/N
print(f"L = {L}, max|G^H W G - I| over all m")
print(f"  equiangular + Clenshaw-Curtis, {N+1} rings: {gram_dev(th_cc, cc_weights(N)):.1e}")
for Nr in (L+1, L):
    x, w = np.polynomial.legendre.leggauss(Nr)
    print(f"  Gauss-Legendre, {Nr} rings:              {gram_dev(np.arccos(x), w):.1e}")

# reconstruction from the Gauss-Legendre grid
rng = np.random.default_rng(2)
x_true = {}
for m in range(-L, L+1):
    n_cols = 2*(L-max(1, abs(m))+1)
    x_true[m] = rng.standard_normal((n_cols, 1)) + 1j*rng.standard_normal((n_cols, 1))

xg, wg = np.polynomial.legendre.leggauss(L+1)
thg, Nphi = np.arccos(xg), 2*L+1

# "measured" samples on the coarse grid
Ct = np.zeros((L+1, Nphi), complex); Cp = np.zeros_like(Ct)
for m, xm in x_true.items():
    c = G_of(thg, L, m) @ xm[:, 0]
    Ct[:, m % Nphi] += c[:L+1]; Cp[:, m % Nphi] += c[L+1:]
St, Sp = np.fft.ifft(Ct, axis=1)*Nphi, np.fft.ifft(Cp, axis=1)*Nphi

# fit: FFT in phi + Gauss-Legendre quadrature projection in theta
Ft, Fp = np.fft.fft(St, axis=1)/Nphi, np.fft.fft(Sp, axis=1)/Nphi
W = np.concatenate([wg, wg])[:, None]
x_fit = {m: G_of(thg, L, m).conj().T @ (W*np.concatenate([Ft[:, m % Nphi], Fp[:, m % Nphi]])[:, None])
         for m in range(-L, L+1)}

# resynthesise on the 1 deg grid and compare with the true pattern
fine = Grid(180, 360)
Kt, Kp = fine.synth(x_true, L)
Rt, Rp = fine.synth(x_fit, L)
err = np.sqrt(np.sum(abs(Rt-Kt)**2 + abs(Rp-Kp)**2)/np.sum(abs(Kt)**2 + abs(Kp)**2))
print(f"\nsamples: {L+1} x {Nphi} = {(L+1)*Nphi} directions "
      f"(minimum L(L+2) = {L*(L+2)}, 1 deg grid = {181*360})")
print(f"relative rebuild error on the 1 deg grid: {err:.1e}")
