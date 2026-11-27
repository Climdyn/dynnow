'''
Geometry of the domain on which the fields are defined (sphere or plane).

A geometry defines the inner product between two fields f and g as a weighted mean over the domain:
    <f, g> = sum_x w(x) f(x) g(x),    with sum_x w(x) = 1,
so that <f, f> is the mean of f**2 over the domain. Pixels outside the mask have w(x) = 0.

When the domain allows it (whole sphere on a Driscoll-Healy grid or periodic plane, and no mask), the
same inner product can also be computed from the spectral coefficients and split by scale (Parseval):
    <f, g> = sum_s <f, g>_s
where s is the degree l on the sphere or the radial wavenumber k on the plane.

The functions *_one_lead_time take numpy arrays of shape (..., vec, *spatial_shape) and return
the Gram matrices G[..., i, j] = <f_i, g_j> of shape (..., vec, vec1).
'''

import numpy as np
import xarray as xr
import pyshtools as pysh
from pyshtools.backends.shtools import DHaj


class Geometry:

    def __reduce__(self):
        '''
        tells pickle (used by dask to send each task to the workers) to save only the constructor arguments
        (the 1D coordinates and the mask) and to call the constructor again when unpickling, instead of
        copying all the attributes (the weights are an 8 MB array for a 0.25 degree grid)
        '''
        return (self.__class__, self._init_args)

    def set_weights(self, weights, mask):
        '''weights: DataArray with self.spatial_dims (unnormalised), mask: None or boolean DataArray with self.spatial_dims'''
        if mask is not None:
            weights = weights * mask
        weights = weights.transpose(*self.spatial_dims)
        self.weights = weights / weights.sum()
        self.spatial_shape = weights.shape
        self.has_mask = mask is not None

    def comp_pixel_gram_one_lead_time(self, vectors1, vectors2=None):
        '''
        vectors1: (..., vec, *spatial_shape), vectors2: (..., vec1, *spatial_shape) or None (then vectors2 = vectors1)
        returns G[..., i, j] = <vectors1_i, vectors2_j>, shape (..., vec, vec1)
        only the pixels with a non-zero weight are used (so that NaNs outside the domain are allowed)
        '''
        inside = self.weights.values.ravel() > 0
        w = self.weights.values.ravel()[inside]

        lead_shape = vectors1.shape[:-3]
        x1 = vectors1.reshape(-1, vectors1.shape[-3], np.prod(self.spatial_shape))[..., inside].astype(np.float64)
        if vectors2 is None:
            x2 = x1
        else:
            x2 = vectors2.reshape(-1, vectors2.shape[-3], np.prod(self.spatial_shape))[..., inside].astype(np.float64)

        gram = np.empty((x1.shape[0], x1.shape[1], x2.shape[1]))
        for t in range(x1.shape[0]):
            gram[t] = (x1[t] * w) @ x2[t].T
        return gram.reshape(*lead_shape, x1.shape[1], x2.shape[1])

    def check_spectral(self):
        assert self.has_spectral, f'no spectral space for this geometry (Driscoll-Healy grid: {getattr(self, "is_DH", None)}, mask: {self.has_mask})'


class SphereGeometry(Geometry):
    '''
    Fields on a (latitude, longitude) grid in degrees.

    Weights: Driscoll-Healy quadrature weights if the grid is a Driscoll-Healy grid covering the whole
    sphere (latitude = 90, 90 - dlat, ..., -90 + dlat and longitude = 0, dlon, ..., 360 - dlon, with
    nlon = nlat or 2 * nlat), cos(latitude) otherwise (regional domain or other grid).
    With the DH weights and no mask, <f, g> = 1/(4 pi) * integral of f g over the sphere, which is also the
    sum of the products of the 4pi-normalised spherical harmonic coefficients (the spectral space).
    '''

    spatial_dims = ['latitude', 'longitude']
    PS_dim = 'l'

    def __init__(self, latitude, longitude, mask=None):
        '''
        latitude, longitude: 1D coordinates (DataArray or numpy array) in degrees
        mask: None or boolean DataArray (latitude, longitude), True inside the domain
        '''
        latitude = np.asarray(latitude)
        longitude = np.asarray(longitude)
        self._init_args = (latitude, longitude, mask)  # for pickling, see Geometry.__reduce__
        nlat, nlon = len(latitude), len(longitude)

        self.is_DH = (nlon in (nlat, 2 * nlat) and nlat % 2 == 0
                      and np.allclose(latitude, 90 - 180 / nlat * np.arange(nlat))
                      and np.allclose(longitude, 360 / nlon * np.arange(nlon)))

        if self.is_DH:
            latitude_weights = DHaj(nlat)
        else:
            latitude_weights = np.cos(np.deg2rad(latitude))
        weights = xr.DataArray(latitude_weights[:, None] * np.ones(nlon), dims=('latitude', 'longitude'),
                               coords={'latitude': latitude, 'longitude': longitude})
        self.set_weights(weights, mask)

        self.has_spectral = self.is_DH and not self.has_mask
        if self.has_spectral:
            self.sampling = nlon // nlat
            self.lmax = nlat // 2 - 1
            self.PS_coord = np.arange(self.lmax + 1)
            R = 6.371e3  # earth radius in km
            with np.errstate(divide='ignore'):
                self.scales = 2 * np.pi * R / np.sqrt(self.PS_coord * (self.PS_coord + 1))

    def to_spectral(self, field):
        '''field: (nlat, nlon) -> 4pi-normalised real spherical harmonic coefficients (2, lmax + 1, lmax + 1)'''
        return pysh.expand.SHExpandDH(np.asarray(field, dtype=np.float64), sampling=self.sampling)

    def to_pixel(self, coeffs):
        '''coeffs: (2, lmax + 1, lmax + 1) -> field (nlat, nlon)'''
        return pysh.expand.MakeGridDH(np.asarray(coeffs, dtype=np.float64), sampling=self.sampling)

    def comp_scale_gram_one_lead_time(self, vectors1, vectors2=None):
        '''
        G[..., l, i, j] = sum_{cos/sin, m} c_i(l, m) c_j(l, m) where c are the spherical harmonic coefficients,
        shape (..., l, vec, vec1). Summing over l gives the pixel Gram matrix.
        '''
        self.check_spectral()
        L = self.lmax + 1

        def to_coeffs(vectors):
            # (T, n, nlat, nlon) -> (T, L, n, 2 * L) so that a matrix product sums over (cos/sin, m)
            flat = vectors.reshape(-1, *self.spatial_shape)
            coeffs = np.stack([self.to_spectral(f) for f in flat])                        # (T * n, 2, L, L)
            coeffs = coeffs.reshape(-1, vectors.shape[-3], 2, L, L)                        # (T, n, 2, L, L)
            return coeffs.transpose(0, 3, 1, 2, 4).reshape(coeffs.shape[0], L, vectors.shape[-3], 2 * L)

        lead_shape = vectors1.shape[:-3]
        c1 = to_coeffs(vectors1)
        c2 = c1 if vectors2 is None else to_coeffs(vectors2)
        gram = c1 @ np.swapaxes(c2, -1, -2)                                                # (T, L, n1, n2)
        return gram.reshape(*lead_shape, L, c1.shape[2], c2.shape[2])
    
    def downscale_one_field(self, field, factor):
        cilm = self.to_spectral(field)
        idx = cilm.shape[1] / factor
        assert idx == int(idx)
        idx = int(idx)
        out = self.to_pixel(cilm[:, :idx, :idx])
        return out
    
    def downscale(self, field, factor):

        output_sizes = {'lat': len(field.latitude) / factor, 'lon': len(field.longitude) / factor}
        for k in output_sizes.keys():
            assert output_sizes[k] == int(output_sizes[k])
            output_sizes[k] = int(output_sizes[k])
        
        out = xr.apply_ufunc(self.downscale_one_field,
                             field,
                             factor,
                             input_core_dims = [['latitude', 'longitude'], []],
                             output_core_dims = [['lat', 'lon']],
                             dask = 'parallelized',
                             dask_gufunc_kwargs = {'output_sizes': output_sizes},
                             output_dtypes = [field.dtype],
                             vectorize = True
                            )
        
        new_lat = field.latitude[::factor].data
        new_lon = field.longitude[::factor].data
        return out.assign_coords({'lat': new_lat, 'lon': new_lon}).rename({'lat': 'latitude', 'lon': 'longitude'})


class PlaneGeometry(Geometry):
    '''
    Fields on a regular (y, x) grid, with uniform weights.
    Without mask, the domain is assumed periodic and the spectral space is given by np.fft.fft2: by Parseval,
    <f, g> = mean of f g = 1/N**2 * sum_k conj(F(k)) G(k) with N the number of pixels.
    The scales are bins of the radial wavenumber k = round(|frequency| * max(ny, nx)), and
    scale = max(ny, nx) / k in pixels.
    '''

    spatial_dims = ['y', 'x']
    PS_dim = 'k'

    def __init__(self, y, x, mask=None):
        '''
        y, x: 1D coordinates
        mask: None or boolean DataArray (y, x), True inside the domain
        '''
        y = np.asarray(y)
        x = np.asarray(x)
        self._init_args = (y, x, mask)  # for pickling, see Geometry.__reduce__
        weights =xr.DataArray(np.ones((len(y), len(x))), dims=('y', 'x'), coords={'y': y, 'x': x})
        self.set_weights(weights, mask)

        self.has_spectral = not self.has_mask
        ny, nx = self.spatial_shape
        n = max(ny, nx)
        frequency = np.sqrt(np.fft.fftfreq(ny)[:, None]**2 + np.fft.fftfreq(nx)[None, :]**2)
        self.wavenumber = np.round(frequency * n).astype(int)                             # (ny, nx), bin of each Fourier coefficient
        self.PS_coord = np.arange(self.wavenumber.max() + 1)
        with np.errstate(divide='ignore'):
            self.scales = n / self.PS_coord

    def to_spectral(self, field):
        return np.fft.fft2(field)

    def to_pixel(self, coeffs):
        # take the real part because some imaginary part can remain
        return np.real(np.fft.ifft2(coeffs))

    def comp_scale_gram_one_lead_time(self, vectors1, vectors2=None):
        '''
        G[..., k, i, j] = 1/N**2 * Re sum_{|k'| in bin k} conj(F_i(k')) F_j(k'), shape (..., k, vec, vec1).
        Summing over k gives the pixel Gram matrix.
        '''
        self.check_spectral()
        N = np.prod(self.spatial_shape)
        bins = self.wavenumber.ravel()

        lead_shape = vectors1.shape[:-3]
        F1 = np.fft.fft2(vectors1.reshape(-1, vectors1.shape[-3], *self.spatial_shape)).reshape(-1, vectors1.shape[-3], N)
        if vectors2 is None:
            F2 = F1
        else:
            F2 = np.fft.fft2(vectors2.reshape(-1, vectors2.shape[-3], *self.spatial_shape)).reshape(-1, vectors2.shape[-3], N)

        gram = np.empty((F1.shape[0], len(self.PS_coord), F1.shape[1], F2.shape[1]))
        for k in self.PS_coord:
            in_bin = bins == k
            gram[:, k] = np.real(np.conj(F1[..., in_bin]) @ np.swapaxes(F2[..., in_bin], -1, -2)) / N**2
        return gram.reshape(*lead_shape, len(self.PS_coord), F1.shape[1], F2.shape[1])
