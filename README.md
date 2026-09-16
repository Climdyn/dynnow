This repository contains python code to analyze dynamical properties of ensembles of nowcasts from radar precipitation data, or more generally of any ensemble forecast on a flat or spherical grid.

The repository is organized in the following way:
 - src/dynnow/ensemble.py contains the abstract 'Ensemble' base class and its two concrete subclasses:
   - 'FourierEnsemble', for forecasts on a flat, regular Cartesian grid (dims `y`, `x`), whose spectral expansion is a 2D FFT.
   - 'SphericalHarmonicsEnsemble', for forecasts on the globe (dims `latitude`, `longitude`, on a Driscoll-Healy grid), whose spectral expansion is a spherical harmonics transform (via `pyshtools`).

   Both subclasses share the same public API (methods to compute power spectra, ensemble eigenvalues/eigenvectors, cosine scores, FSS, and to generate surrogate ensembles); only the geometry-dependent details (spatial/spectral dims, spectral transform, power spectrum) differ. An example of how to use each of them is in doc/example.ipynb
 - src/dynnow/surrogates.py contains the functions to generate the surrogate ensembles
 - src/dynnow/analysis.py contains some functions to automatically generate the surrogates and compute the scores of the ensembles

All computations are lazily done first and then handled by dask when '.compute()' is called.
