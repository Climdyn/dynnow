import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def ax_spectral_error_variance(scales, spectral_error, spectral_variance, ax, norm, times, cmap = cm.viridis):
    for t, time in enumerate(times):
        ax.plot(scales, spectral_variance[t], linestyle = 'dashed', c = cmap(norm(time)))
        ax.plot(scales, spectral_error[t], linestyle = 'dotted', c = cmap(norm(time)))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('scale (km)')
    ax.xaxis.set_inverted(True)
    
    # create proxy lines for the legend
    ax.plot([], [], label = 'spectral variance', linestyle = 'dashed', c = 'black')
    ax.plot([], [], label = 'spectral error', linestyle = 'dotted', c = 'black')
    ax.legend()

def plot_spectral_error_variance(scales, spectral_error, spectral_variance, n = 8, double = False):

    if double:
        fig, axs = plt.subplots(1, 2, figsize = (16, 8))
    else:
        fig, ax = plt.subplots(1, 1, figsize = (10, 8))
        axs = [ax]
        
    cmap = cm.viridis
    
    times = (np.arange(len(spectral_error)) + 1) * 5
    norm = mcolors.Normalize(vmin = np.min(times), vmax = np.max(times))
    
    ax_spectral_error_variance(scales, spectral_error, spectral_variance, ax, norm, times)

    # colorbar showing time
    sm = cm.ScalarMappable(cmap = cmap, norm = norm)
    cbar = plt.colorbar(sm, ax = axs[0])
    cbar.set_label('time (minutes)')

    if double:
        for scale in range(len(scales) // n):
            axs[1].plot(times, spectral_variance[:, scale * n], label = f'{np.round(scales[scale * n], 1)}')
        axs[1].set_xlabel('time (minutes)')
        axs[1].set_ylabel(f'spectral variance')
        axs[1].set_yscale('log')
        axs[1].legend(title = 'scales', loc = 'lower center', ncols = 5)

    return fig, axs

def ax_spectral_mean(scales, spectral_mean, ax, times, norm, linestyle = 'solid', alpha = 1, cmap = cm.viridis):
    
    for t, time in enumerate(times):
        ax.plot(scales, spectral_mean[t], c = cmap(norm(time)), linestyle = linestyle, alpha = alpha)
    
    ax.set_yscale('log')
    ax.set_ylabel('spectrum of mean')
    ax.set_xscale('log')
    ax.set_xlabel('scale (km)')
    ax.xaxis.set_inverted(True)

def plot_eigenvalues(val):
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    cmap = cm.viridis
    times = (np.arange(len(val)) + 1) * 5
    norm = mcolors.Normalize(vmin = np.min(times), vmax = np.max(times))
    for t, time in enumerate(times):
        axs[0].plot(np.arange(len(val[t])) + 1, val[t], c = cmap(norm(time)))
        axs[1].plot(np.arange(len(val[t])) + 1, val[t], c = cmap(norm(time)))
    axs[1].set_yscale('log')
    axs[0].set_xlabel('order of the eigenvalue')
    axs[1].set_xlabel('order of the eigenvalue')

    # colorbar showing time
    sm = cm.ScalarMappable(cmap = cmap, norm = norm)
    cbar = plt.colorbar(sm, ax = axs)
    cbar.set_label('time (minutes)')

def axs_spectral_vec(scales, eigenval, spectral_eigenvec, axs, t_plots, norm, cmap = cm.viridis, linestyle = 'solid', alpha = 1):
    
    for t, t_plot in enumerate(t_plots):
        for i in range(spectral_eigenvec.shape[1]):
            axs[t].plot(scales, spectral_eigenvec[t_plot, i], c = cmap(norm(eigenval[t_plot, i])), linestyle = linestyle, alpha = alpha)
    
    for t, ax in enumerate(axs):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.xaxis.set_inverted(True)
    
def plot_spectral_vec(scales, eigenval, spectral_eigenvec, t_plots = np.array([0, 5, 10, 15])):
    
    times_plots = (t_plots + 1) * 5

    fig, axs = plt.subplots(1, len(t_plots), figsize = (len(t_plots) * 5, 5), sharey = True)
    
    cmap = cm.viridis

    norm = mcolors.LogNorm(vmin = np.min(eigenval[t_plots]), vmax = np.max(eigenval[t_plots]))
    
    axs_spectral_vec(scales, eigenval, spectral_eigenvec, axs, t_plots, norm, cmap = cm.viridis)
    
    # colorbar showing eigenvalue
    sm = cm.ScalarMappable(cmap = cmap, norm = norm)
    cbar = plt.colorbar(sm, ax = axs)
    cbar.set_label('eigenvalue')

def axs_first_spectral_eigenvec(scales, first_spectral_eigenvec, axs, t_plots, c = 'C0'):
    for t, t_plot in enumerate(t_plots):
        for ct in range(first_spectral_eigenvec.shape[0]):
            axs[t].plot(scales, first_spectral_eigenvec[ct, t_plot], c = c)
    
        axs[t].set_yscale('log')
        axs[t].set_xscale('log')
        axs[t].xaxis.set_inverted(True)
        axs[t].set_xlabel('scale (km)')
        axs[t].set_title(f'lead time = {(t_plot + 1) * 5} minutes')

    axs[0].set_ylabel('Spectral content of largest uncertainty mode')

def axs_normalized_eigenval(scales, eigenval, axs, t_plots, c = 'C0'):
    for t, t_plot in enumerate(t_plots):
        for ct in range(eigenval.shape[0]):
            axs[t].plot(eigenval[ct, t_plot]/eigenval[ct, t_plot].max(), c = c)
    
        axs[t].set_yscale('log')
        axs[t].set_xlabel('order')
        axs[t].set_title(f'lead time = {(t_plot + 1) * 5} minutes')
    
    axs[0].set_ylabel('Normalized size of uncertainty')

def plot_correlation_modes(corr):
    fig, ax = plt.subplots()

    modes = np.arange(corr.shape[1]) + 1
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin = np.min(modes), vmax = np.max(modes))
    
    for m, mode in enumerate(modes):
        ax.plot(np.abs(corr[:, m]), c = cmap(norm(mode)))
    ax.set_xlabel('time')
    ax.set_ylabel('correlation of (true - mean) with ensemble modes')
    
    # colorbar showing mode
    sm = cm.ScalarMappable(cmap = cmap, norm = norm)
    cbar = plt.colorbar(sm, ax = ax)
    cbar.set_label('mode')

def ax_center_of_mass(ax, central_time, pre_lead_times, lead_times, in_com, true_com, pred_com):
    
    flts = np.concatenate((pre_lead_times, lead_times), axis = 0)/np.timedelta64(1, 'm')
    lts = lead_times/np.timedelta64(1, 'm') 
    full_true_com = np.concatenate((in_com, true_com), axis = 0)

    mean = np.mean(pred_com, axis = 1)
    std = np.std(pred_com, axis = 1)
    zero = in_com[-1]
    
    # plot y_com
    ax.plot(flts, full_true_com[:, 0] - zero[0], c = 'C0')
    ax.plot(lts, mean[:, 0] - zero[0], c = 'C0', linewidth = 0.5)
    ax.fill_between(lts, mean[:, 0] - zero[0] + std[:, 0], mean[:, 0] - zero[0] - std[:, 0], color = 'C0', alpha = 0.2)

    # plot x_com
    ax.plot(flts, full_true_com[:, 1] - zero[1], c = 'C4')
    ax.plot(lts, mean[:, 1] - zero[1], c = 'C4', linewidth = 0.5)
    ax.fill_between(lts, mean[:, 1] - zero[1] + std[:, 1], mean[:, 1] - zero[1] - std[:, 1], color = 'C4', alpha = 0.2)

    ax.set_title(f'{central_time.astype('datetime64[m]')}')
    ax.set_xlabel('Lead time (minutes)')

def plot_center_of_mass(central_times, pre_lead_times, lead_times, in_com, true_com, pred_com):

    fig, axs = plt.subplots(2, 5, figsize = (17, 7), sharex = True)

    for t, ax in enumerate(axs.flatten()):
        ax_center_of_mass(ax, central_times[t], pre_lead_times, lead_times, in_com[t], true_com[t], pred_com[t])

    p1 = ax.plot([], [], c = 'C0', label = 'actual $y_{COM}$')
    p2 = ax.plot([], [], c = 'C4', label = 'actual $x_{COM}$')
    p3 = ax.plot([], [], color = 'C0', linewidth = 0.5)
    p4 = ax.fill(np.nan, np.nan, 'C0', alpha = 0.2)
    p5 = ax.plot([], [], color = 'C4', linewidth = 0.5)
    p6 = ax.fill(np.nan, np.nan, 'C4', alpha = 0.2)

    fig.legend([p1[0],
                p2[0],
                (p3[0], p4[0], ),
                (p5[0], p6[0], )
               ],
               [p1[0].get_label(),
                p2[0].get_label(),
                'ensemble $y_{COM}$',
                'ensemble $x_{COM}$'
               ],
               loc = 'center right'
              )
    
    plt.subplots_adjust(hspace = 0.3)

def plot_event_observation_and_predictions(observation, predictions, lts = [0, 4, 9, 14, 19], members = np.arange(3), cmap = cm.viridis):
    vmin = -15
    vmax = max(np.max(predictions.data), np.max(observation.data))
    subplot_size = 2.5

    # lead time increases from left to right, observation is on top row, and other rows are for members
    fig, axs = plt.subplots(len(members) + 1, len(lts), figsize = (len(lts) * subplot_size + 3, subplot_size * (len(members) + 1)), sharex = True, sharey = True)

    # plot observation and predictions
    for i, lt in enumerate(lts):
        sc = axs[0, i].imshow(observation.isel(lead_time = lt), vmin = vmin, vmax = vmax, cmap = cmap)
        for j, m in enumerate(members):
            sc = axs[j + 1, i].imshow(predictions.isel(lead_time = lt, member = m), vmin = vmin, vmax = vmax, cmap = cmap)

    # label rows
    axs[0, 0].set_ylabel('observation')
    for j, m in enumerate(members):
        axs[j + 1, 0].set_ylabel(f'member {j + 1}')
    # label columns
    for i, lt in enumerate(lts):
        axs[0, i].set_title(f'{(lt + 1) * 5} minutes')

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.colorbar(sc, ax = axs, shrink = 0.5)

def axs_FSS(axs, FSS, times, thresholds, scales, norm, cmap, linestyle = 'solid'):
    '''FSS has shape (time, thresholds, scales)'''

    for thr, thresh in enumerate(thresholds):
        for t, ti in enumerate(times):
            axs[thr].plot(scales, FSS[t, thr], c = cmap(norm(ti)), linestyle = linestyle)
            axs[thr].set_xscale('log')
            axs[thr].set_xlabel('scale (km)')
            axs[thr].set_title(f'threshold = {np.round(thresholds[thr], 2)} dBR = {np.round(10**(thresholds[thr]/10), 2)} mm/h')
    axs[0].set_ylabel('FSS')
    
