"""Step 0a: data-driven re-centring on a synthetic shifted 8x8 array.

Compares the closed form (one fit) with a brute-force quadratic search over
J(r0) = sum_e <nu>_e, and shows the tail energy before / after re-centring.
"""
import numpy as np
from swe_lib import Grid, spectrum, recentre_closed_form

grid = Grid(180, 360)
k = 2*np.pi                                       # lambda = 1
rng = np.random.default_rng(1)

# local element patterns: random SWE coefficients up to n = 3 (dominant n = 1)
Lel = 3
xel = {}
for m in range(-Lel, Lel+1):
    ns = np.arange(max(1, abs(m)), Lel+1)
    amp = np.where(ns == 1, 1.0, 0.3)
    shape = (2*ns.size, 64)
    xel[m] = np.concatenate([amp, amp])[:, None]*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))
Kt0, Kp0 = grid.synth(xel, Lel)

# 8x8 at lambda/2, whole array shifted by r0_true; data referenced to the simulation origin
ix, iy = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
d = np.stack([(ix.ravel()-3.5)*0.5, (iy.ravel()-3.5)*0.5, np.zeros(64)], -1)
r0_true = np.array([0.7, -1.3, 0.4])
phase = np.exp(1j*k*np.einsum("ijc,ec->ije", grid.rhat, d + r0_true))
Kt, Kp = Kt0*phase, Kp0*phase

Lbig = 60


def rephased_spectrum(r0):
    g = np.exp(-1j*k*grid.rhat @ r0)[:, :, None]
    return spectrum(grid.fit(Kt*g, Kp*g, Lbig), Lbig)


def J(r0):
    S = rephased_spectrum(r0); n = np.arange(Lbig+1)[:, None]
    return np.sum(n*(n+1)*S/S.sum(0))


# closed form
r_cf = recentre_closed_form(grid, Kt, Kp, Lbig, k)

# brute force: J is exactly quadratic -> central differences give exact gradient / Hessian
h = 0.25; z = np.zeros(3); E3 = np.eye(3); J0 = J(z)
g = np.array([(J(h*E3[i])-J(-h*E3[i]))/(2*h) for i in range(3)])
H = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        H[i, j] = (J(h*(E3[i]+E3[j]))-J(h*(E3[i]-E3[j]))-J(h*(E3[j]-E3[i]))+J(-h*(E3[i]+E3[j])))/(4*h*h)
r_bf = -np.linalg.solve(H, g)
rt = np.array([0.3, 0.2, -0.5])
print("quadratic check: J(rt) =", J(rt), " model =", J0 + g@rt + 0.5*rt@H@rt)

print("r0_true     =", r0_true, " (lambda)")
print("closed form =", np.round(r_cf, 6))
print("brute force =", np.round(r_bf, 6), "  |diff| =", f"{np.linalg.norm(r_cf-r_bf):.1e}")

print("\nworst-element tail energy above L")
for name, r0 in [("no shift removed", z), ("re-centred", r_cf)]:
    S = rephased_spectrum(r0)
    cells = []
    for L in (20, 25, 30, 35, 40):
        frac = S[L+1:].sum(0)/S.sum(0)
        cells.append(f"L={L}: {10*np.log10(frac.max()):7.1f} dB")
    print(f"{name:17s}", "  ".join(cells))
