import numpy as np
import mne
import os
import pandas as pd
import glob
import gc
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

# --- mpl setup ---

mpl.rcParams['image.cmap'] = 'viridis'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2(np.linspace(0, 1, 8)))
# mpl.rcParams['font.family'] = 'Helvetica Neue'
mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.dpi'] = 100

# --- Setup dataframe ---

path = "/work3/s224188/FG_Data/FG_overview_df_v2.pkl"
overview = pd.read_pickle(path)

folder_path = "/work3/s224188/FG_Data/PreprocessedEEGData/"

# Use glob to find all FIF files in the folder
epoch_files = glob.glob(os.path.join(folder_path, '*-epo.fif'))
epochs_list = []

for f in epoch_files: # Load each FIF file into a list of epoch objects
    
    basename = os.path.basename(f) # Extract participant ID from filename
    
    Exp_id = basename.split('_')[0] # Get Exp id
    
    gender = overview.loc[overview['Exp_id'] == Exp_id, 'Gender'].iloc[0]
    
    epochs = mne.read_epochs(f, preload=False, verbose=False)
    epochs_list.append({ # Construct list of corresponding exp id, gender and epochs
        'Exp_id': Exp_id,
        'Gender': gender,
        'Epochs': epochs
    })
epochs_df = pd.DataFrame(epochs_list)
sfreq = int(epochs_df['Epochs'][0].info['sfreq'])
n_epochs, n_channels, n_timepoints = epochs_df['Epochs'][0].get_data().shape

# --- Functions ---

def get_psds_for_channel(participants_data, channel, settings, window, band = [1,40], overlap = 0.75, mean = True):
    
    overlap = int(window * sfreq * overlap) # scale overlap to window size
    psd_dict = {}
    for setting in settings:
        all_psds = []
        for epochs in participants_data:
            # Pick channel 
            filtered_epochs = epochs.copy().load_data().pick([channel])
            filtered_epochs = filtered_epochs.filter(l_freq=band[0], h_freq=band[1], fir_design='firwin', n_jobs=10, verbose = False)
            filtered_epochs = filtered_epochs[setting]
            # Compute PSD
            psds = filtered_epochs.compute_psd(method='welch', window = 'hamming', average = None, 
                                               fmin=band[0], fmax=band[1], n_fft=int(sfreq*window*2/band[0]), 
                                               n_overlap=int(overlap*2/band[0]), n_jobs=10, verbose = False);
            if mean:
                mean_psds = np.array(psds.get_data().mean(axis=0).mean(axis = 1)).flatten()
                # Standardize to baseline 
                baseline = mean_psds[int(len(mean_psds)/20):int(len(mean_psds)/10)].mean() 
                all_psds.append(mean_psds/baseline) # append the mean over epochs and frequencies to all psds
            else:
                psds_list = np.array(psds.get_data().mean(axis=0))
                baseline_snippet = psds_list[:, int(len(psds_list)/20):int(len(psds_list)/10) + 1]  # add 1 because Python slicing is exclusive
                baseline_mean = np.mean(baseline_snippet, axis=1, keepdims=True)
                log_ratio = 10 * np.log10(psds_list / baseline_mean)
                all_psds.append(np.reshape(log_ratio, (len(log_ratio[0]),len(log_ratio[0][0]))))
            # Unload data from iteration to save memory space
            del filtered_epochs
            gc.collect()
            
        label = f'{setting}, {band} frequencies'
        psd_dict[label] = np.reshape(all_psds, (len(all_psds), len(all_psds[0]))) if mean else np.array(all_psds).mean(axis=0)
        
    return psd_dict

def plot_ERD(psds_dict, ax): 
    """ Takes a dict where each value should be of size (n_segments, n_channels), \\
    and returns frequency over time plots in a (n_items, 2) grid"""
    times = np.linspace(-0.5, 5.5, len(next(iter(psds_dict.values()))[0])) # Create time points equal to powers
    
    for data_label, powers in psds_dict.items():
        mean_line = np.mean(powers, axis = 0)
        std_line = np.std(powers, axis = 0)/np.sqrt(len(powers))
        ax.plot(times, mean_line, label = data_label)
        ax.fill_between(times, mean_line - std_line, mean_line + std_line, alpha=0.3)
    
    ax.axhline(y=1, color='black', linestyle='--', label='baseline', alpha = 0.3) # plot baseline
    ax.set_ylim(0.5,1.3)
    ax.set_xlim(-0.01,4)

def plot_spectrogram_on_ax(psd_list, freqs = np.arange(1,41), ax=0):
    n_times = psd_list.shape[1]
    time_vector = np.linspace(-0.5, 5.5, n_times)
    start_idx = np.argmin(np.abs(time_vector - 0))
    end_idx = np.argmin(np.abs(time_vector - 4))
    psd_list = psd_list[:, start_idx:end_idx+1]
    time_vector = time_vector[start_idx:end_idx+1]
    
    im = ax.imshow(psd_list, aspect='auto', origin='lower',
                   extent=[time_vector[0], time_vector[-1], freqs[0], freqs[-1]], vmin=0, vmax=1)
    return im

# --- Channel grid for plots ---

ch_grid_dict = {                                       'Fp1':(2.7,0), 'Fpz':(4,0), 'Fp2':(5.3,0),
                                          'AF7':(1.3,1), 'AF3':(2.5,1), 'AFz':(4,1), 'AF4':(5.5,1), 'AF8':(6.7,1),
                'F7':(0,2),  'F5':(1,2),  'F3':(2,2),  'F1':(3,2),  'Fz':(4,2),  'F2':(5,2),  'F4':(6,2),  'F6':(7,2),  'F8':(8,2),
                'FT7':(0,3), 'FC5':(1,3), 'FC3':(2,3), 'FC1':(3,3), 'FCz':(4,3), 'FC2':(5,3), 'FC4':(6,3), 'FC6':(7,3), 'FT8':(8,3),
                'T7':(0,4),  'C5':(1,4),  'C3':(2,4),  'C1':(3,4),  'Cz':(4,4),  'C2':(5,4),  'C4':(6,4),  'C6':(7,4),  'T8':(8,4),
                'TP7':(0,5), 'CP5':(1,5), 'CP3':(2,5), 'CP1':(3,5), 'CPz':(4,5), 'CP2':(5,5), 'CP4':(6,5), 'CP6':(7,5), 'TP8':(8,5),
                'P7':(0,6),  'P5':(1,6),  'P3':(2,6),  'P1':(3,6),  'Pz':(4,6),  'P2':(5,6),  'P4':(6,6),  'P6':(7,6),  'P8':(8,6),
                'P9':(0,7),                                                                                             'P10':(8,7),
                                          'PO7':(1.3,7), 'PO3':(2.5,7), 'POz':(4,7), 'PO4':(5.5,7), 'PO8':(6.7,7),
                                                       'O1':(2.8,8),  'Oz':(4,8),  'O2':(5.2,8),
                                                                    'Iz':(4,9)
                
                }

# --- Plotting ---

# Parameters
mne.set_log_level('warning')
window = 1
settings = ['T3Pn', 'T3Pn']
overlap = 0.9
epochs = epochs_df['Epochs']
# band = [12,35]

for band in [[4,8], [8,12], [12,35], [1,40]]:
    # ERD Grid Plot
    handles, labels = None, None

    # Define grid size based on your ch_grid_dict
    n_cols = max(pos[0] for pos in ch_grid_dict.values()) + 1
    n_rows = max(pos[1] for pos in ch_grid_dict.values()) + 1

    fig = plt.figure(figsize=(11, 8))

    for ch, (x, y) in tqdm(ch_grid_dict.items(), desc=f'Creating ERD grid, with frequency band: {band}, and settings: {settings}'):
    
        psds = get_psds_for_channel(participants_data=epochs, overlap=overlap, channel=ch, band=band, window=window, settings=settings);
        # Normalize positions to [0, 1] figure coordinates
        top_margin = 0.08 
        ax_width = 1 / n_cols
        ax_height = (1 - top_margin) / n_rows

        left = x * ax_width
        bottom = 1 - top_margin - (y + 1) * ax_height
    
        ax = fig.add_axes([left, bottom, ax_width, ax_height])
        plot_ERD(psds, ax)
        ax.text(0.5, 0.95, ch, ha='center', va='top', transform=ax.transAxes, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
        if handles is None and labels is None:  # Capture legend once
            handles, labels = ax.get_legend_handles_labels()

    # General legend
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)

    plt.suptitle('ERD plots for all channels', fontsize=16)
    fig.savefig(f'erd_grid_plot_group_f_nf_{band[0]}_{band[1]}.png', dpi=300)
    plt.close(fig)

# Plot Spectrogram
mpl.rcParams['image.cmap'] = 'viridis'
fig = plt.figure(figsize=(11, 8))
axes = []  # to collect axes
ims = []   # to collect plots 
band = [1,40]
global_max, global_min = -np.inf, np.inf
for ch, (x, y) in tqdm(ch_grid_dict.items(), desc=f'Creating spectrogram grid, with frequency band: {band}, and settings: {settings}'):
    
    psds = get_psds_for_channel(participants_data=epochs, overlap=overlap, channel=ch, band=band, window=window, settings=settings, mean=False)
    # Get difference between psds
    diff_psds = list(psds.values())[0] - list(psds.values())[1]
    # Normalize positions to [0, 1] figure coordinates
    top_margin = 0.08 
    ax_width = 1 / n_cols
    ax_height = (1 - top_margin) / n_rows

    left = x * ax_width
    bottom = 1 - top_margin - (y + 1) * ax_height
    
    ax = fig.add_axes([left, bottom, ax_width, ax_height])
    im = plot_spectrogram_on_ax(diff_psds, ax = ax)
    ims.append(im)    # Collect image for rescaling later
    axes.append(ax)   # Collect axes

    ax.text(0.5, 0.95, ch, ha='center', va='top', transform=ax.transAxes, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Update global min and max
    local_min = np.min(diff_psds)
    local_max = np.max(diff_psds)

    if local_min < global_min:
        global_min = local_min
    if local_max > global_max:
        global_max = local_max

    del diff_psds  # Free memory


# Fix color limits for all images
for im in ims:
    im.set_clim(global_min, global_max)
    
# Shared colorbar on the right
cbar = fig.colorbar(ims[0], ax=axes, location='right', shrink=0.9)
cbar.set_label('Power Difference (DB)')

plt.suptitle("Spectrogram plots for all channels", fontsize=16)
fig.savefig('Spectrogram_grid_plot_solo_group_nf.png', dpi=300)
plt.close(fig)