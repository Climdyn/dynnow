# dynnow

Diagnostics of ensemble forecasts, computed from Gram matrices in pixel space or in spectral space.

dynnow implements the multivariate, phase-space evaluation framework of [Bonte and Vannitsem (2026)](#history-and-references): an ensemble forecast is interpreted as the
linear subspace spanned by its members in phase space (the *ensemble subspace*), and this subspace is
characterised geometrically — its extension in each direction, the part of the forecast error it
captures, its convergence with the ensemble size, and its comparison with the subspaces of other
ensembles or of surrogate ensembles.

It works on any ensemble forecast defined on a flat regular grid (regional models, nowcasts of radar
precipitation) or on the sphere (global models), for one or several variables and lead times.
Computations can be carried out in pixel space or in spectral space.

## Notation and the idea in one paragraph

The ensemble members of a forecast are denoted x_i for i = 1, ..., N. The verifying analysis is y,
while the ensemble mean, the error, and the residual vectors are, respectively,

    x̄ = 1/N Σ_i x_i,        e = y − x̄,        v_i = x_i − x̄.

The time dependence (and any other index such as the variable) is omitted for the sake of clarity.
The analysis revolves around the residual vectors v_i (how the members spread around the mean) and
the error vector e (how the analysis deviates from the mean).

Instead of manipulating the fields themselves (huge arrays), dynnow first reduces everything to
**Gram matrices**: the small tables of inner products `G_ij = <v_i, v_j>` between residual vectors,
plus the inner products <v_i, e> between residual vectors and the error. All the diagnostics are then
linear algebra on these small matrices, which is cheap and exact.

The link between the Gram matrix, the residual vectors and the eigenvectors is the method of
snapshots (Sec. 3.1 of the paper). The ensemble covariance matrix is

    C = 1/(N−1) VVᵀ,        where V is the matrix whose columns are the v_i.

It has rank N−1 and is huge in the high-dimensional phase space of weather forecasting, but instead
of diagonalising it, one can diagonalize the Gram matrix

    G = 1/(N−1) VᵀV.

G is N×N, much smaller than C, and has the same eigenvalues λ_i as C. Its eigenvectors W give the
eigenvectors u_i of C (the Empirical Orthogonal Functions of the ensemble) through U = VWΣ⁻¹, where
Σ comes from the singular value decomposition V = UΣWᵀ; equivalently,

    u_i = Σ_j Q_ji v_j,        with Q_ji = W_ji / sqrt((N−1) λ_i).

Conversely, everything about the u_i follows from G and the coefficients Q without ever forming them
as fields: for instance `<e, u_i> = Σ_j Q_ji <v_j, e>`. The eigenvectors as actual fields
(`comp_eigenvec()`) are only reconstructed at the end, if wanted. The u_i form an orthonormal basis of
the **ensemble subspace**, the subspace spanned by the v_i, in which the ensemble has variance.

The used inner product between two fields f and g is, on the sphere, the integral over the sphere of
the product of f and g, <f, g> = ∫_S² f g dΩ (Eq. 11 of the paper); in the orthonormal basis of
spherical harmonics it is <a, b> = Σ_lm a_lm b*_lm (Eq. 25). The code can easily be adapted to other
scalar products.

The inner product can be computed in **pixel space** (a quadrature over the grid) or, when the
geometry allows it, in **spectral space** (sum over spherical harmonic or Fourier coefficients, by
Parseval). The spectral space comes with a bonus: the Gram matrices split by scale, from which the
power spectra follow directly.

## What is computed

All diagnostics are methods of the `Ensemble` class (see the example below); results are collected in
`ensemble.diagnostics`, an `xarray.Dataset` with a `lead_time` dimension. The section numbers refer
to the paper.

**Ensemble covariance matrix and eigen-decomposition (Sec. 3.1)**

- `eigenval`, `eigenvec_coefficients`: the eigenvalues λ_i of the covariance matrix (each is the
  variance of the ensemble in the direction of the associated eigenvector) and the coefficients Q_ji
  above, ordered by decreasing eigenvalue.

**Projection on ensemble subspace (Sec. 3.2)**

- `cosine_error_projection`: cos γ(e, p), where p = Σ_i (e·u_i) u_i is the projection of the error on
  the ensemble subspace. The ensemble represents the error
  completely when cos γ(e, p) = 1.
- `cosine_error_members`: the cosines between the error and each eigenvector, (e·u_i)/|e|; their
  squares sum to cos²γ(e, p).

**Projection on eigenvectors and relative variances (Sec. 3.3)**

- `d_spectrum`: (e·u_i)²/λ_i
- `histos_projection_eigenvec`: histograms of the cosines between each residual vector and each
  eigenvector.

**Energy score**

- `multivariate_energy_score`: the (fair by default) multivariate energy score of the ensemble,
  computed from the Gram matrices.

**Convergence of the covariance matrix with the ensemble size (Sec. 4.1)**

- `comp_ensemble_convergence()`: constructs smaller ensembles of 5, 10, 20, ... members randomly
  extracted from the full one, and computes their eigenvalue spectra and the principal angles between
  the first eigenvectors of the smaller ensembles and those of the full ensemble (pure linear algebra
  on sub-blocks of the Gram matrix, no new field computation) — do the covariance matrix and the
  ensemble subspace converge with the number of members?

**Power spectra (Sec. 3.5)**

- `PS_error`, `PS_member_vec`, `PS_eigenvec` (spectral space only): power spectra of the error, of
  the residual vectors and of the eigenvectors. The shape of these spectra structures the phase space
  and controls how much of the diagnostics is due to the power spectrum alone — hence the surrogates.

**On the fields themselves**

- `comp_eigenvec()`: the eigenvectors u_i as actual fields in pixel space.
- `comp_FSS(thresholds, scales)`: the fractions skill score of the ensemble (flat grid only).

**Surrogate ensembles (Sec. 3.5)**

- `generate_surrogates(kind)`: builds surrogate ensembles that share some statistics with the
  original but are otherwise random; comparing the diagnostics of both quantifies how much of the
  values of the metrics is really due to the skill of the model to produce v_i with relevant
  orientations, rather than to the shape of the power spectrum of the v_i. Available kinds:
  - `'spectral'`: the surrogates of the paper, x̄ + w_i where w_i is a statistically isotropic random
    field on the sphere with the same power spectrum as the corresponding v_i;
  - `'spectral_nowcasting'`: same, but the phases are randomised around those of a reference member
    (keeps the position of the structures, for nowcasting ensembles);
  - `'maaft'`: amplitude-adjusted Fourier transform surrogates (same power spectrum *and* same
    distribution of pixel values);
  - `'hist'`: each member's pixel values are shuffled in space (same distribution of values, no
    spatial structure at all).

**Comparison of two ensembles (Secs. 3.4 and 4.4)**

Module `dynnow.comparison`, used e.g. to compare two ensemble subspaces:

- `comp_cross_gram(e1, e2)`: the cross Gram matrix `<v1_i, v2_j>` between the residual vectors of the
  two ensembles (each centered on its own ensemble mean);
- `comp_principal_angles(e1, e2, cross_gram)`: the principal angles θ_i between the two ensemble
  subspaces — the smallest angles between all pairs of unit vectors of the subspaces (their cosines
  are the singular values of the matrix of inner products between orthonormal bases of the two
  subspaces). The first principal angle measures in which direction the two ensembles agree the most;
- `comp_cosine_member_member(e1, e2, cross_gram)`: cosines between the residual vectors of one
  ensemble and the residual vectors of the other.

## Geometry and inner products

The inner product `<f, g> = sum_x w(x) f(x) g(x)` is defined by a geometry object:

- `SphereGeometry(latitude, longitude, mask=None)`: fields on a (latitude, longitude) grid in degrees.
  On a global Driscoll-Healy grid, the weights are the exact DH quadrature weights and the inner
  product is the integral over the sphere, ∫_S² f g dΩ; on a regional domain it falls back to
  cos(latitude) weights.
- `PlaneGeometry(y, x)`: fields on a regular Cartesian grid, uniform weights.

A boolean `mask` (True inside the domain) zeroes the weights outside and renormalises them on the
domain; NaNs outside the domain are then allowed. Pixel space always works with a mask; spectral
space requires the whole domain (no mask, DH grid on the sphere, periodic plane). Computing the inner
products in spectral space is what the paper does (Sec. 2), because spherical harmonics are
orthonormal for the inner product of Eq. (11) and this avoids contributions from the oversampled
points near the poles of the regular grid.

`space='spectral'` computes the same inner products from the spectral coefficients and additionally
provides the Gram matrices split by scale (`gram_*_by_scale`), i.e. the contribution of each degree l
(sphere) or radial wavenumber k (plane) to each inner product. The `scale` coordinate gives the
corresponding wavelength in km. Pixel and spectral spaces agree up to the scales not representable by
the spectral basis.

## Lazy evaluation with dask

Like the previous version, dynnow is fully lazy: `comp_*` methods build a dask graph and nothing is
computed until `.compute()` is called. The intended workflow is to call all the `comp_*` methods you
need and then `ensemble.compute()` once, so that the (expensive) Gram matrices are computed exactly
once and everything downstream is cheap linear algebra on the already-materialised matrices.

## Installation

Requires Python >= 3.10, with `xarray`, `scipy`, `numba`, `pysteps` and `pyshtools`
(the latter only for spherical geometries).

## Example

```python
from dynnow import SphereGeometry, Ensemble
from dynnow import comparison

# forecast: DataArray (lead_time, member, latitude, longitude)
# observation: DataArray (lead_time, latitude, longitude) — the verifying analysis y
geometry = SphereGeometry(forecast.latitude, forecast.longitude)     # optionally mask=...
original = Ensemble(forecast, geometry, observation=observation, space='spectral')

original.comp_gram()                     # Gram matrices (the only expensive step)

original.comp_eigenval_eigenvec()                # eigenvalues λ_i and eigenvector coefficients Q
original.comp_cosine_error_projection()          # cos γ(e, p): is the error in the ensemble subspace?
original.comp_d_spectrum()                       # (e·u_i)²/λ_i
original.comp_multivariate_energy_score()        # fair energy score
for field in ('error', 'eigenvec', 'member_vec'):
    original.comp_power_spectrum(field)
convergence = original.comp_ensemble_convergence()   # DataTree, one child per sub-ensemble size

original.compute()                       # materialise them; everything below is cheap linear algebra
original.diagnostics                     # xr.Dataset with all the results

# surrogate ensembles with the same power spectrum, then comparison of the subspaces
spectral = original.generate_surrogates('spectral')
spectral.comp_gram()
spectral.comp_eigenval_eigenvec()
cross_gram = comparison.comp_cross_gram(original, spectral).cross_gram
principal_angles = comparison.comp_principal_angles(original, spectral, cross_gram)
```

## Project layout

| module | content |
|---|---|
| `geometry.py` | `SphereGeometry`, `PlaneGeometry`: inner-product weights (and mask), spectral transforms, Gram kernels (pixel and split by scale) |
| `linalg.py` | linear algebra on Gram matrices: eigenvalues, eigenvector coefficients, principal angles, re-centering |
| `ensemble.py` | `Ensemble`: one ensemble (+ verifying analysis), `comp_gram()` then all the diagnostics, results in `ensemble.diagnostics` |
| `comparison.py` | two ensembles: cross Gram matrix, principal angles, cosine member-member |
| `surrogates.py` | surrogate ensembles |

<a id="history-and-references"></a>
## History and references

This dynnow version (v3) is a rewrite of dynnow v2, restructured around Gram
matrices computed once up front rather than diagnostics scattered on the fields.

If you use dynnow, please cite:

> M. Bonte and S. Vannitsem, *Multivariate phase-space interpretation of ensemble forecasts: a
> comparison of physical and AI models*, preprint [arXiv:2610.16422](https://arxiv.org/abs/2610.16422)
> (2026).
