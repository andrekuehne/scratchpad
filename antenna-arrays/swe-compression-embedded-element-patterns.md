# SWE Compression of Embedded Element Patterns — Process Summary

Assumption: full-sphere far-field data on a regular θ,φ grid.

Numerical checks (NumPy only): [`swe-scripts/`](swe-scripts/)
- `check_orthogonality.py` — Gram matrices for `sinθ·Δθ` vs
  Clenshaw–Curtis weights, nested truncation (steps 4, 4a)
- `check_recentring.py` — closed-form vs brute-force re-centring on a
  synthetic shifted 8×8 array (step 0a)
- `swe_lib.py` — shared helpers (tables, design matrices, fit,
  synthesis, closed-form r₀)

## 0. Conventions

- Far field: `E(r,θ,φ) = K(θ,φ)·e^{-jkr}/r`,  `K = (Kθ, Kφ)`, complex.
- ONE common origin for all 64 elements (array centre).
  No per-element origins.
- Grid: `θ_i = 0..180°` (`Nθ = 181`), `φ_k = 0..359°` (`Nφ = 360`; drop the
  duplicated 360° column).
- Data per frequency: `K[e, pol, i, k]`, `e = 0..63`, `pol ∈ {θ, φ}`.
- Fit and synthesis use identical tables, so any consistent
  normalisation works; the orthonormal choice below gives Parseval
  (needed for the power matrix).

## 0a. Re-centring: data-driven global origin r₀

Far fields referenced to a simulation origin O, array centre at r₀
(convention `e^{−jkr}/r`):
```
K_O(θ,φ) = K_c(θ,φ) · e^{+jk r₀·r̂},    r̂ = (sinθcosφ, sinθsinφ, cosθ)
```
- Exact pointwise phase factor in the far field. The SWE about O can
  represent it, but needs `L ≈ ⌈k(|r₀| + a)⌉ + 10` → Ncoef grows like
  `(k(|r₀|+a))²` (a = 2.8λ: shift 1λ → 1.5×, 5λ → 4.3×, 10λ → L > 90,
  exceeds the 1° grid). It also adds a direction-dependent delay
  `±|r₀|/c` to the frequency axis (step 9).
- A global shift is common to all elements and factors out of the
  superposition (`K_array = e^{jk r₀·r̂} Σ w_e K_c,e`) → remove it.
  Per-element offsets `d_e` do not factor out and stay in the
  coefficients (common origin).

### Objective: most compact mode spectrum

For a trial r₀, rephase `K' = K·e^{−jk r₀·r̂}`, fit, and minimise
```
J(r₀) = Σ_e ⟨ν⟩_e,   ⟨ν⟩_e = Σ_n n(n+1) S_e(n) / Σ_n S_e(n)
```
(mean-square angular frequency; `√⟨ν⟩/k` ≈ RMS radiating radius seen
from r₀). J is exactly quadratic in r₀ → closed-form minimiser.

### Mode energy as a field integral

- `Z_nm = ∇_s Y_nm/√ν` (TM, gradient type),
  `X_nm = −r̂ × ∇_s Y_nm/√ν` (TE, rotated gradient); with
  `Δ_s Y = −νY` and `curl_s F := r̂·∇×F`:
  ```
  div_s Z = −√ν Y,   div_s X = 0
  curl_s X = +√ν Y,  curl_s Z = 0
  ```
- Hence
  ```
  D := div_s K  = −Σ √ν · b_nm · Y_nm
  C := curl_s K = +Σ √ν · a_nm · Y_nm
  Σ ν(|a|² + |b|²) = ∫ ( |D|² + |C|² ) dΩ
  ```

### Re-centring is affine in r₀ → J quadratic

- `g = e^{−jk r₀·r̂}`, `∇_s g = −jk g P_t r₀`, `P_t = I − r̂r̂ᵀ`:
  ```
  div_s (gK) = g · ( D − jk r₀·K )
  curl_s(gK) = g · ( C + jk r₀·(r̂×K) )
  J(r₀) = ∫ |D − jk r₀·K|² + |C + jk r₀·(r̂×K)|² dΩ
  ```
- For real r₀ the quadratic terms add up to `|K|² P_t` (K and r̂×K
  span the tangent plane, any polarisation):
  ```
  J(r₀) = J(0) + gᵀ r₀ + ½ r₀ᵀ H r₀
  H = 2k² M,   M = ∫ |K|² P_t dΩ                     (3×3, real, sym.)
  g = −2k n,   n = ∫ Im( C̄ (r̂×K) − D̄ K ) dΩ         (3-vector)
  ```

### Closed-form solution

```
r₀ = M⁻¹ n / k
```
Summed over elements and frequencies with per-element normalisation
`w_ef = 1/∫|K_ef|² dΩ`:
```
r₀ = [ Σ_ef w_ef k_f² M_ef ]⁻¹ · [ Σ_ef w_ef k_f n_ef ]
```

### Physical meaning

- Locally single-polarised field `K = A e^{jψ}` (A real, tangential):
  `Im(C̄(r̂×K) − D̄K) = |K|² ∇_s ψ` →
  ```
  r₀ = (1/k) · M⁻¹ ∫ |K|² ∇_s ψ dΩ
  ```
  = power-weighted LS fit of the phase gradient to a pure translation,
  `∇_s ψ ≈ k P_t r₀` (point source at p → r₀ = p exactly).
- The D/C form gives the same without phase unwrapping and stays valid
  in nulls and for mixed / elliptical polarisation.
- For `K = K_c e^{jk p·r̂}`: `r₀ = p − M⁻¹ n_c / k` = true shift + the
  intrinsic centre of the centred pattern — the right answer for
  compression.

### Implementation

1. Fit once with `L_big` (large enough for the un-centred data) →
   `a_nm, b_nm`.
2. Scalar spectra `D_m = −Θ_n^{|m|}·(√ν b)`, `C_m = +Θ_n^{|m|}·(√ν a)`,
   inverse FFT in φ (reuses the Θ tables, no derivatives needed).
3. Cartesian vectors: `K = Kθ θ̂ + Kφ φ̂`, `r̂×K = Kθ φ̂ − Kφ θ̂`,
   `θ̂ = (cosθcosφ, cosθsinφ, −sinθ)`, `φ̂ = (−sinφ, cosφ, 0)`.
4. M and n by quadrature, `dΩ = w_i^{CC} · 2π/Nφ` (exact: integrands
   band-limited to ~2·L_big < 180).
5. Solve 3×3 → rephase with `e^{−jk r₀·r̂}` → fit with the final L →
   store r₀ in the metadata (the "origin" field of step 4).
- Cost: one fit + two scalar syntheses + grid sums (vs ~10 fits for a
  search over J).

### Practical

- Large shifts (`L_big > 90` needed): estimate at the lowest frequency
  first (r₀ is frequency-independent, mode count scales with k), then
  refine with a second pass on the re-centred data at the highest
  frequency.
- One r₀ for all frequencies (sum over f) → coefficients stay smooth
  in f.
- Per-element centres: the same formula per element (no Σ_e) gives
  `r₀,e ≈ d_e + r₀` → check layout / port mapping.
- Asymmetric layouts / platforms: Σ_e gives the centroid; if the
  worst element should set L, refine with a minimax criterion (max
  tail energy above L), starting from the centroid.
- Conditioning: M is ill-conditioned only if nearly all power is in
  one direction; summing over elements and frequencies fixes this in
  practice.
- Downstream: `|e^{jk r₀·r̂}| = 1` → magnitudes, nulls, P_rad, R
  unchanged. For absolute phase in the original frame re-apply the
  factor pointwise after synthesis (step 7) or on the steering rows
  (step 8) — not in the coefficient domain (full translation matrix
  there).

### Check (synthetic)

- 8×8 at λ/2, random element patterns up to n = 3, shift
  r₀ = (0.7, −1.3, 0.4)λ, L_big = 60, 1° grid:
  - closed form (1 fit): (0.704252, −1.296070, 0.402876)λ;
  - brute-force quadratic search over J (19 fits): identical to
    9e-14; the 0.005λ offset from the true shift is the intrinsic
    centre of the random element patterns.
- Worst-element tail energy above L:

  | L | 20 | 25 | 30 | 35 |
  |---|---|---|---|---|
  | no shift removed | −2 dB | −13 dB | −37 dB | −70 dB |
  | re-centred | −33 dB | −71 dB | −120 dB | −178 dB |

  → −60 dB needs L ≈ 34 without, ≈ 24 with re-centring.

## 1. Basis (tangential vector spherical harmonics)

- Scalar: `Y_nm = Θ_n^m(θ)·e^{jmφ}/√(2π)`
  - `Θ_n^m` = fully normalised assoc. Legendre in cosθ,
    `∫_0^π |Θ_n^m|² sinθ dθ = 1`;  `Θ_n^m := Θ_n^{|m|}` for `m < 0`.
- θ-tables for `n = 1..L`, `m = −n..n`, `ν = n(n+1)`:
  ```
  A_nm(θ) = m·Θ_n^m(θ)/sinθ / √ν
  B_nm(θ) = dΘ_n^m/dθ      / √ν
  ```
- Mode functions (θ-comp, φ-comp), times `e^{jmφ}/√(2π)`:
  ```
  TE  X_nm = ( j·A_nm , −B_nm  )
  TM  Z_nm = (   B_nm , j·A_nm )
  ```
- Model:
  ```
  K(θ,φ) = Σ_{n=1..L} Σ_{m=−n..n} [ a_nm X_nm + b_nm Z_nm ]
  ```
  #coefficients `Ncoef = 2·L(L+2)`
- Numerics: stable recursions for `Θ`, `Θ/sinθ`, `dΘ/dθ` (or a library
  such as pyshtools / SHTns / scipy). Poles: use analytic limits;
  `m·Θ/sinθ` and `dΘ/dθ` are non-zero there only for `|m| = 1`.
- Time convention / `j^n` factors only matter when converting to
  Hansen `Q_smn` or near fields.

## 2. Truncation L

- `a` = radius of minimum sphere about the origin enclosing all
  radiating structure.
- Start with `L ≈ ceil(k_max·a) + 10`. After fitting, inspect
  ```
  S(n) = Σ_m |a_nm|² + |b_nm|²
  ```
  and raise L until tail and residual < target (e.g. −60 dB if
  deep nulls matter). Requires `Nφ ≥ 2L+1` and `Nθ ≫ L`.
- Simplest: one L (highest frequency) for all frequencies.
- With exact θ quadrature (step 4) the coefficients are nested: fit
  once with a generous `L_fit` (≤ 90 on the 1° grid) and truncate
  later per frequency/application, see step 4a.

## 3. φ-direction: FFT (exact separation by m)

- `C[e,pol,i,m] = fft_φ(K[e,pol,i,:])[m mod Nφ] / Nφ`,  `|m| ≤ L`
  (`numpy.fft.fft` along φ; negative m at index `Nφ + m`)

## 4. θ-direction: weighted least squares per m (linear, closed form)

- For each m: `n = n0..L`, `n0 = max(1,|m|)`, `N_m = L − n0 + 1`
  - Unknowns `x_m = [a_{n,m} ; b_{n,m}]`  (2N_m)
  - Design matrix `G_m` (2Nθ × 2N_m), all entries `/√(2π)`:
    ```
    θ-rows: [  j·A_nm(θ_i) ,   B_nm(θ_i) ]
    φ-rows: [   −B_nm(θ_i) , j·A_nm(θ_i) ]
    ```
- Quadrature weights: Clenshaw–Curtis in `x = cosθ`.
  The grid `θ_j = jπ/N`, `j = 0..N` (N = 180) is exactly the
  Chebyshev–Lobatto grid in cosθ, so these weights integrate
  `∫ f sinθ dθ` exactly for f polynomial in cosθ of degree ≤ N:
  ```
  w_j = c_j/N · ( 1 − Σ_{k=1..N/2} b_k·cos(2kθ_j)/(4k²−1) )
  c_j = 1 for j ∈ {0, N}, else 2
  b_k = 1 for 2k = N,     else 2
  ```
  Pole weights are small but non-zero (pole rows need the analytic
  limits of A, B). `W = diag(w, w)` for θ- and φ-rows.
- Why not `w_i = sinθ_i·Δθ`: that is a trapezoid rule with O(Δθ²)
  endpoint error at the poles → Gram matrix only ≈ I (max deviation
  ~7e-4 at |m| = 1, ~1e-5 at m = 0, 2, ≤ 1e-12 for |m| ≥ 5;
  L = 28, 1° grid). Usable, but not exactly nested.
- Exact discrete orthogonality: at fixed m every vector-harmonic
  inner-product integrand (TE·TE, TM·TM, TE·TM) is a polynomial in
  cosθ of degree n + n′, so with Clenshaw–Curtis weights
  ```
  2π · G_m^H W G_m = I     exactly, for 2L ≤ N  (L ≤ 90 here)
  ```
- Hence the least-squares fit is a plain quadrature projection —
  no pinv needed. Precompute ONCE per (grid, L, m):
  ```
  P_m = 2π · G_m^H W                          (2N_m × 2Nθ)
  ```
  (equivalent to `pinv(W^½ G_m) · W^½`, which remains the fallback
  for non-exact weights.)
- Fit all elements × frequencies with one matmul per m:
  ```
  X_m = P_m @ C_m,   C_m shape (2Nθ, 64·Nf)
  ```
- Store per frequency f: `Q_f` (Ncoef × 64), complex64
  - plus metadata: L, origin, grids, freq list, normalisation,
    coefficient index map (n, m, TE/TM).

## 4a. Nested truncation (fit high, discard later)

- Question: fit with many modes, drop the high orders later — do the
  remaining coefficients shift?
- Answer: no, as long as the modes are orthogonal over the fitting
  region on the discrete grid:
  - different m: separated exactly by the φ-FFT (aliasing only for
    |m| > Nφ/2);
  - same m, different n / TE vs TM: orthogonal over the full sphere;
    exact on the grid with Clenshaw–Curtis weights (step 4).
- Then the coefficients do not depend on the L used in the fit:
  truncating `L_fit → L` gives the same coefficients as fitting with L
  directly, and the extra error is exactly the energy of the dropped
  modes:
  ```
  ||K − K̂_L||²_W = residual(L_fit) + Σ_{n=L+1..L_fit} S(n)
  ```
  → `S(n)` directly gives the truncation error for any choice of L.
- Numerical check (synthetic pattern with modes to n = 50, fit with
  L_fit = 40 then truncate to 28, vs. fit with 28 directly):
  - Clenshaw–Curtis: relative coefficient difference ~2e-15;
  - `sinθ·Δθ`: up to ~8e-7 (|m| = 1), i.e. coupling (≤ 1e-3) × size
    of the dropped tail — still far below a −60 dB target;
  - residual identical in both cases.
- Practical: fit once with a generous L_fit, choose L per frequency
  from S(n) afterwards.
- Caveats:
  - requires full-sphere data; a partial sphere or masked regions
    break orthogonality → coefficients couple and shift on truncation;
  - unweighted LS (`W = I`) also breaks it — the weights are what make
    the fit a projection;
  - content above n ≈ N/2 = 90 or |m| > Nφ/2 = 180 aliases into the
    kept modes (not a concern for a ≈ 2.8λ).

## 5. Validation

- Resynthesize every element (step 7):
  ```
  rel. RMS error             = ||K − K̂||_W / ||K||_W
  max error re. peak [dB]    = 20·log10( max|K − K̂| / max|K| )
  ```
- Parseval: `∫|K|² dΩ ≈ ||q||²` (exact on the grid for a
  band-limited pattern with Clenshaw–Curtis weights; the gap is the
  out-of-band residual)
- Optional: null depth of steered patterns vs. raw grid data.

## 6. Superposition (each weight change)

- `q = Q_f @ w`  (w: 64 complex weights)
- Many weight sets at once: `Q_f @ W`
- Between frequency samples: see step 9 (remove feed delays first;
  plain linear interpolation is not good enough for coarse sampling).
- Radiated power: `P_rad = ||q||²/(2η) = w^H R w/(2η)`,
  `R = Q_f^H Q_f` (64×64, precompute).

## 7. Synthesis (coefficients → any display grid)

- Build unweighted `G_m` on the display θ grid.
- For each `|m| ≤ L`:  `c_m(θ) = G_m @ x_m`  → Kθ, Kφ
- Put `c_m` at index `(m mod Nφ_disp)`, then
  ```
  K(θ, φ_k) = Nφ_disp · ifft_φ(c)          (numpy.fft.ifft)
  ```
  `Nφ_disp ≥ 2L+1`; larger `Nφ_disp` = finer φ for free (zero-pad).
  Append the φ = 360° column for plotting if needed.
- Cost ≈ `4·Nθ·L²` MACs + Nθ FFTs of length Nφ; tables a few MB
  → cache-resident, sub-ms with NumPy/Numba.

## 8. Point evaluation / null steering

- Direction `(θ0,φ0)`: basis rows `b` (2 × Ncoef) → steering
  `s = b @ Q_f` (2 × 64: Kθ and Kφ responses to unit weights).
- Pick component(s) to null (e.g. Ludwig-3 co/cross-pol).
- Stack constraint rows into `A` (Nc × 64); minimum-change nulls:
  ```
  w = ( I − A^H (A A^H)^{-1} A ) w0
  ```
- Real achievable null depth ≈ model error from step 5.

## 9. Frequency axis: sampling and interpolation

The fit is linear and identical at every frequency, and the basis is
orthonormal (Parseval), so all frequency modelling can be done on the
coefficients `Q[c, e, f]` instead of on the patterns — same L2 error.
Use one common `L_fit` for all f (nested, step 4a).

### 9.1 Delay spread → sampling requirement

- Impulse response of a coefficient: `h_{c,e}(t) = IFT_f{ q_{c,e}(f) }`.
  Delay spread T = time span holding all energy above the error target.
- Contributions (supports add, effects convolve):
  - feed delay `τ_e`: pure delay `e^{−j2πfτ_e}` (port reference plane
    behind feed line / cable) — removable, see 9.2;
  - geometry: element offset `d_e` from the common origin → arrivals
    within `±|d_e|/c`; bound for any element `2a/c`;
  - element ringing: amplitude time constant `τ_r = Q/(π f_c)`,
    decays to ε after `τ_r·ln(1/ε)`;
  - mutual coupling: multiple passes across the aperture
    (≈ aperture/c per pass);
  - global shift r₀ of the array from the origin: `±|r₀|/c`,
    direction-dependent (not removable by τ_e) → remove by
    re-centring (step 0a).
- Sampling every Δf makes the time response periodic with `1/Δf`:
  ```
  Δf < 1/T            hard limit (otherwise time-domain aliasing)
  Δf ≈ 1/(2…3·T)      practical (exponential tails)
  ```
  For a resonance: `Δf < π·Δf_3dB / ln(1/ε)` → at −60 dB ≥ 2.2
  samples per 3 dB bandwidth, 4–6 in practice.
- Example (rough assumptions): 14–16 GHz, 8×8 at λ/2 @ 16 GHz,
  a ≈ 52 mm.
  - 2a/c ≈ 0.35 ns; ringing Q ≈ 8 → 1.2 ns to −60 dB;
    coupling ≈ 0.75 ns → T ≈ 2.3 ns → Δf < 430 MHz.
  - 250 MHz sampling (9 points) is fine for a low-Q element covering
    the band; a high-Q feature (Q ≈ 50, 300 MHz wide) would need
    Δf ≲ 140 MHz → under-resolved.

### 9.2 Remove feed delays first

- Estimate per element from adjacent samples:
  ```
  arg( q_e(f)^H q_e(f+Δf) ) ≈ −2πΔf·τ_e      (average over f)
  ```
  multiply out `e^{+j2πfτ_e}`, store `τ_e`, re-apply after
  interpolation.
- Why it matters: interpolators assume content near t = 0. Linear
  interpolation of `e^{−j2πfτ}`, phase step `δ = 2πΔfτ`, midpoint
  error `1 − cos(δ/2)`:
  - τ = 0.5 ns, Δf = 250 MHz: δ = 45° → ≈ −22 dB;
  - geometric part, corner element (±0.15 ns): δ ≈ 14° → ≈ −43 dB
    — not removable by τ_e, already ruins −60 dB nulls with linear
    interpolation.

### 9.3 Interpolator

- After τ_e removal, choose by what remains:
  - cubic spline or global low-order polynomial (Chebyshev /
    barycentric): covers the geometric part easily (for 0.15 ns over
    2 GHz a degree-8 polynomial is accurate to ~1e-7);
  - rational fit with common poles for resonant behaviour (9.5).
- Interpolate `Q`, then compute `R(f) = Q(f)^H Q(f)` — do not
  interpolate R (quadratic in Q).

### 9.4 Validation without new simulations: leave-one-out

- Drop each interior sample, predict it from the remaining ones,
  compare (pattern-domain error = coefficient error by Parseval).
- Directly answers whether sampling + interpolator suffice; does not
  test extrapolation at the band edges.
- Optional diagnostic: power delay profile
  `PDP(t) = Σ_{c,e} |h_{c,e}(t)|²` (Kaiser window β ≈ 8–10, not
  Hann); a wrapped-around tail at the window end means Δf too coarse.

### 9.5 High-Q resonances: rational fit + adaptive points

- Nyquist reasoning assumes a generic band-limited model. A resonance
  is one pole pair: long delay spread, but few degrees of freedom.
- Vector fitting with poles shared by all ~110k responses:
  ```
  Q(f) ≈ Σ_p R_p / (j2πf − p_p) + D + E·f
  ```
  Shared poles are pinned down robustly; geometric phase goes into
  `D + E·f` / low-order terms (small here, B·2a/c ≈ 0.7).
- Needs a few samples within the resonance's 3 dB width, not dense
  sampling over the whole band.
- Adaptive sampling: coarse grid → fit → add a simulation point where
  models of different order / leave-one-out predictions disagree →
  repeat. New points land only near the resonance.
- Solver side: fast / adaptive frequency sweeps (check that far
  fields, not only S-parameters, are available), or a time-domain
  solver (one run per port gives all frequencies; cost scales with T,
  i.e. high Q → longer run, not more frequency points).

### 9.6 Frequency compression (only with dense sampling)

- With few samples (e.g. 9) there is nothing to gain in storage or
  speed. With dense samples (e.g. 201 from a sweep):
  - remove τ_e, optionally normalise per frequency (`1/||Q_f||_F`);
  - joint SVD of `M` ((Ncoef·64) × Nf) ≈ `U_r S_r V_r^H`; discarded
    `Σ σ_ρ²` = discarded pattern energy (Parseval, Eckart–Young);
  - fit the r curves `v_ρ(f)` with Chebyshev / splines;
  - `Q(f) = e^{−j2πfτ} · Σ_ρ v_ρ(f) U_ρ`; for fixed weights
    precompute `U_ρ @ w` → r-term sum per frequency.
- Expected size: `r ≈ B·T + few`, Chebyshev terms
  `≈ (π/2)·B·T + margin`.

## Typical sizes (8×8, λ/2 spacing, a ≈ 2.8λ)

- `L ≈ 28` → `Ncoef ≈ 1700` per element vs 2 × 65k grid samples
- Per frequency: ~0.9 MB vs 67 MB raw (both complex64)
