import xarray as xr

def comp_ensemble_convergence(ensemble, smaller_sizes = [5, 10, 20, 50], agreement_order = 4):
    '''
    Computes diagnostics to assess the convergence of the ensemble in phase space
    Several smaller ensembles are defined (of sizes smaller_sizes) and the eigenvalues of the covariance matrices of these ensembles are computed
    Are also computed the principal angles between the eigenvectors of the smaller ensembles and those of the full ensemble
    '''
    if not 'eigenval' not in ensemble or 'spectral_eigenvec' not in self:
        ensemble.comp_eigenval_eigenvec()

    ensemble_convergence = xr.DataTree(xr.Dataset(data_vars = {'lead_time': ensemble.lead_time}))
    
    for n_member in smaller_sizes:
        smaller_ensemble = type(ensemble)(data_vars = {'forecast': ensemble.forecast.isel(member = np.random.choice(len(ensemble.member), size = n_member, replace = False))})
        smaller_ensemble.comp_eigenval_eigenvec()
        principal_angles = comp_principal_angles(ensemble.isel(order = range(agreement_order)),
                                                 smaller_ensemble.isel(order = range(agreement_order)))\
            .rename({'order': 'agreement_order'})
        ensemble_convergence = ensemble_convergence.assign({f'{n_member}members': xr.DataTree(xr.Dataset(data_vars = {'eigenval': smaller_ensemble.eigenval.rename({'order': 'smaller_order'}),
                                                                                                                      'principal_angles': principal_angles}))})
        
    return ensemble_convergence