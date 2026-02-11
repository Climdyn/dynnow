import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np



def ax_histo_projection_eigenvec(ax, histo, title = None, **kwargs):

    ax.bar(histo.bins_center, histo, width = histo.bin_width, **kwargs)
    ax.set_xlabel(r'$c_{ij}$')
    ax.set_ylabel('Density')
    ax.set_xlim(-1., 1.)

    if title is not None:
        ax.set_title(title)

def ax_eigenval_order(ax, eigenval, title = None, **kwargs):
    '''spectrum of eigenvalues for one lead time'''
    
    for ct in range(len(eigenval.central_time)):
        e = eigenval.isel(central_time = ct)
        ax.plot(e/e.max(dim = 'order'), **kwargs)

    ax.set_yscale('log')
    ax.set_ylabel(r"Normalized eigenvalues (unitless)")
    ax.set_xlabel('Order of the eigenvalue')

def ax_eigenval_lead_time(ax, eigenval, title = None, **kwargs):
    lead_times = eigenval.lead_time.data/np.timedelta64(1, 'm')
    ax.plot(lead_times, eigenval, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r"Eigenvalues $\lambda_i$'s ($\text{dBR}^2$)")
    ax.set_xlabel('Lead time (minutes)')
    if title is not None:
        ax.set_title(title)

def ax_spectral_field(ax, spectral_ensemble_mean, cmap = cm.viridis, title = None, ylabel = None, **kwargs):

    lead_times = spectral_ensemble_mean.lead_time.data/np.timedelta64(1, 'm')
    norm = mcolors.Normalize(vmin = np.min(lead_times), vmax = np.max(lead_times))

    for lt, lead_t in enumerate(lead_times):
        ax.plot(spectral_ensemble_mean.scale, spectral_ensemble_mean.isel(lead_time = lt), c = cmap(norm(lead_t)), **kwargs)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('scale (km)')
    ax.xaxis.set_inverted(True)

    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return norm

def axs_spectral_vec(axs, eigenval, spectral_eigenvec, t_plots, norm, cmap = cm.viridis):
    
    for t, t_plot in enumerate(t_plots):
        for o in range(len(spectral_eigenvec.order)):
            axs[t].plot(spectral_eigenvec.scale, spectral_eigenvec.isel(lead_time = t_plot, order = o), c = cmap(norm(eigenval.isel(lead_time = t_plot, order = o))))
    
    for t, ax in enumerate(axs):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.xaxis.set_inverted(True)

def ax_cosine_error_projection(ax, cosine_error_projection, label, linestyle = 'solid', title = None):
    lead_times = cosine_error_projection.lead_time.data/np.timedelta64(1, 'm')
    for i, ct in enumerate(cosine_error_projection.central_time):
        ax.plot(lead_times, cosine_error_projection.sel(central_time = ct), alpha = 0.5, c = f'C{i}', linestyle = linestyle)
    ax.plot(lead_times, cosine_error_projection.mean(dim = 'central_time').data, linewidth = 5, c = 'black', linestyle = linestyle)
    ax.plot([], [], c = 'black', label = label, linestyle = linestyle)

    if title is not None:
        ax.set_title(title)

def axs_FSS(axs, FSS, cmap = cm.viridis, linestyle = 'solid', label = None):
    '''FSS has shape (time, thresholds, scales)'''

    lead_times = FSS.lead_time.data/np.timedelta64(1, 'm')
    norm = mcolors.Normalize(vmin = np.min(lead_times), vmax = np.max(lead_times))

    for thr, thresh in enumerate(FSS.threshold_FSS):
        for lt, lead_t in enumerate(lead_times):
            axs[thr].plot(FSS.scale_FSS, FSS.isel(lead_time = lt).sel(threshold_FSS = thresh), c = cmap(norm(lead_t)), linestyle = linestyle)
            axs[thr].set_xscale('log')
            axs[thr].set_xlabel('scale (km)')
            axs[thr].set_title(f'threshold = {np.round(thresh.data, 2)} dBR = {np.round(10**(thresh.data/10), 2)} mm/h')
    axs[0].set_ylabel('FSS')

    if label is not None:
        axs[0].plot([], [], linestyle = linestyle, label = label, c = 'black')

    return norm

    