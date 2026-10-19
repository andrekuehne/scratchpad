"""Step 4 / 4a: discrete orthogonality (sin*dtheta vs Clenshaw-Curtis) and nested truncation."""
import numpy as np
from swe_lib import theta_tables, AB, G_of, cc_weights

N, L = 180, 28
th = np.arange(N+1)*np.pi/N
w_sin = np.sin(th)*np.pi/N; w_sin[[0, -1]] = 0
w_cc = cc_weights(N)

# sanity: dTheta/dtheta vs finite difference
A, B = AB(th[1:-1], 5, 2)
Tp = theta_tables(th[1:-1]+1e-6, 6); Tm = theta_tables(th[1:-1]-1e-6, 6)
fd = (Tp[4, 2]-Tm[4, 2])/2e-6/np.sqrt(20)
print("derivative check:", np.max(abs(fd - B[4-2])))

print(f"\nGram deviation max|G^H W G - I|, L={L}, 1 deg grid")
print(" m   sin*dtheta   Clenshaw-Curtis")
for m in [0, 1, 2, 3, 5, 10, 20, 28]:
    G = G_of(th, L, m)
    dev = []
    for wv in (w_sin, w_cc):
        W = np.concatenate([wv, wv])
        H = G.conj().T @ (W[:, None]*G)
        dev.append(np.max(abs(H - np.eye(H.shape[0]))))
    print(f"{m:3d}   {dev[0]:10.2e}   {dev[1]:10.2e}")

# synthetic pattern with modes to Ltrue and a decaying tail
rng = np.random.default_rng(0)
Ltrue, Lbig, Lsmall, ka = 50, 40, 28, 18
print(f"\nFit L={Lbig} then truncate to {Lsmall} vs fit L={Lsmall} directly (data has modes to {Ltrue})")
print(" m   weights   |dx|/|x|   resid_trunc   resid_direct")
for m in [0, 1, 2, 5]:
    Gt = G_of(th, Ltrue, m)
    n0 = max(1, m); ns = np.arange(n0, Ltrue+1)
    amp = np.where(ns <= ka, 1.0, 10**(-(ns-ka)/4))   # ~ -5 dB per order beyond ka
    xt = np.concatenate([amp, amp])*(rng.standard_normal(2*ns.size)+1j*rng.standard_normal(2*ns.size))
    c = Gt @ xt
    for name, wv in [("sin", w_sin), ("CC ", w_cc)]:
        W = np.concatenate([wv, wv]); sw = np.sqrt(W)[:, None]
        Gb, Gs = G_of(th, Lbig, m), G_of(th, Lsmall, m)
        xb = np.linalg.lstsq(sw*Gb, sw[:, 0]*c, rcond=None)[0]
        xs = np.linalg.lstsq(sw*Gs, sw[:, 0]*c, rcond=None)[0]
        nb, nsm = Lbig-n0+1, Lsmall-n0+1
        xb_tr = np.concatenate([xb[:nsm], xb[nb:nb+nsm]])
        rel = np.linalg.norm(xb_tr-xs)/np.linalg.norm(xs)
        nrm = lambda r: np.sqrt(np.sum(W*abs(r)**2))/np.sqrt(np.sum(W*abs(c)**2))
        print(f"{m:3d}   {name}     {rel:9.2e}   {nrm(c-Gs@xb_tr):10.2e}   {nrm(c-Gs@xs):10.2e}")
