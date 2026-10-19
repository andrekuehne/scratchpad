# SWE Compression of Embedded Element Patterns — Process Summary

Assumption: full-sphere far-field data on a regular θ,φ grid.

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
- Between frequency samples: interpolate Q entrywise in f
  (linear or cubic; smooth because the origin is fixed).
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

## Typical sizes (8×8, λ/2 spacing, a ≈ 2.8λ)

- `L ≈ 28` → `Ncoef ≈ 1700` per element vs 2 × 65k grid samples
- Per frequency: ~0.9 MB vs 67 MB raw (both complex64)
