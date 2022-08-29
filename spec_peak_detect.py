import os
import sys
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from functools import partial
from IPython import embed
from thunderfish.dataloader import open_data
from thunderfish.powerspectrum import spectrogram, next_power_of_two, decibel

def load_data(folder):
    filename = os.path.join(folder, 'traces-grid1.raw')
    data = open_data(filename, -1, 60.0, 10.0)
    samplerate = data.samplerate
    channels = data.channels

    return filename, data, samplerate, channels

def compute_spectrogram(data, samplerate, freq_res, overlap_frac, channels, start_idx, end_idx):
    core_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(core_count - 1)
    func = partial(spectrogram, ratetime=samplerate, freq_resolution=freq_res, overlap_frac=overlap_frac)

    a = pool.map(func, [data[start_idx: end_idx+1, channel] for channel in np.arange(channels)])

    spectra = [a[channel][0] for channel in range(len(a))]
    spec_freqs = a[0][1]
    spec_times = a[0][2]
    pool.terminate()

    comb_spectra = np.sum(spectra, axis=0)
    db_comb_spectra = decibel(comb_spectra)
    tmp_times = spec_times + (start_idx / samplerate)

    return comb_spectra, db_comb_spectra, tmp_times, spec_freqs

def EOD_detection(db_spectra, spec_freq, spec_time, db_th0, db_th1, min_freq = 250, max_freq = 1200):
    groups = []
    fundamentals = []
    f_mask = np.arange(len(spec_freq))[(spec_freq <= 3000)]

    db_spectra = db_spectra[f_mask[0]:f_mask[-1]+1, np.arange(np.shape(db_spectra)[1])]
    spec_freq = spec_freq[f_mask]

    #for i in np.arange(np.shape(db_spectra)[1]):
    for i in np.array(np.linspace(0, np.shape(db_spectra)[1]-1, 100), dtype=int):

        embed()
        quit()
        # get all entries above th (bool-mask)
        psd_mask = np.array(db_spectra[:, i] >= db_th0, dtype=bool)

        # include 1 entry before and after valid (to valid bool-mask)
        help_array = np.diff(np.array(psd_mask, dtype=int))
        psd_mask[np.arange(len(help_array))[help_array == 1]] = 1
        psd_mask[np.arange(len(help_array))[help_array == -1] + 1] = 1

        # index of valid bool-mask
        psd_mask_idx = np.arange(len(db_spectra))[psd_mask]

        # 1st-derivation of valid psd snippets (zeros crossings are peaks in PSDs)
        # psd_derivation = np.diff(db_spectra[psd_mask, i])
        psd_derivation = np.full(len(spec_freq), np.nan)
        # psd_derivation[psd_mask_idx] = np.diff(np.arange(len(db_spectra)))[psd_mask_idx]
        psd_derivation[psd_mask_idx[1:]] = np.diff(db_spectra[psd_mask, i])

        peaks = np.arange(len(psd_derivation)-1)[(psd_derivation[:-1] > 0) & (psd_derivation[1:] < 0)]
        trough = np.arange(len(psd_derivation)-1)[(psd_derivation[:-1] <= 0) & (psd_derivation[1:] > 0)]
        # peaks = psd_mask_idx[help_peaks]
        # trough = psd_mask_idx[help_trough]

        f_peaks = peaks[(spec_freq[peaks] >= min_freq) & (spec_freq[peaks] <= max_freq)]
        f_trough = trough[(spec_freq[trough] >= min_freq) & (spec_freq[trough] <= max_freq)]
        # ToDo: For each fundamental peak detect 1 & 2 harmonic !!!
        h1_peaks = np.array(list(set(peaks) - set(f_peaks)), dtype=int)
        h1_trough = np.array(list(set(trough) - set(f_trough)), dtype=int)


        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(30/2.54, 20/2.54))
        ax[0].plot(spec_freq, db_spectra[:, i])
        ax[0].plot([spec_freq[0], spec_freq[-1]], [db_th0, db_th1], 'k-')

        # dada = np.full(len(spec_freq), np.nan)
        # dada[psd_mask_idx[1:]] = psd_derivation

        # ax[1].plot(spec_freq[psd_mask][1:], psd_derivation)
        ax[1].plot(spec_freq[1:], np.diff(db_spectra[:, i]), color='grey')
        ax[1].plot(spec_freq, psd_derivation)
        ax[1].plot([spec_freq[0], spec_freq[-1]], [0, 0], 'k--', lw=1, alpha=0.8)

        ax[0].plot(spec_freq[f_peaks], db_spectra[f_peaks, i], 'o', color='forestgreen')
        ax[0].plot(spec_freq[f_trough], db_spectra[f_trough, i], 'o', markeredgecolor='forestgreen', color='none')
        ax[0].plot(spec_freq[h1_peaks], db_spectra[h1_peaks, i], 'o', color='orange')
        try:
            ax[0].plot(spec_freq[h1_trough], db_spectra[h1_trough, i], 'o', markeredgecolor='orange', color='none')
        except:
            embed()
            quit()
        ax[1].plot(spec_freq[peaks], np.zeros(len(peaks)), 'ko')
        ax[1].plot(spec_freq[trough], np.zeros(len(trough)), 'o', markeredgecolor='k', color='none')

        plt.show()


        fundamentals.append(spec_freq[f_peaks])

        # th_cross_up = np.arange(len(db_spectra[:, i])-1)[(db_spectra[:-1, i] <= db_th0) & (db_spectra[1:, i] > db_th0)]
        # th_cross_down = np.arange(1, len(db_spectra[:, i]))[(db_spectra[:-1, i] >= db_th0) & (db_spectra[1:, i] < db_th0)]
        #
        # fig, ax = plt.subplots()
        # ax.plot(spec_freq, db_spectra[:, i])
        # ax.plot(spec_freq[th_cross_up], db_spectra[th_cross_up, i], 'o', color='green')
        # ax.plot(spec_freq[th_cross_down], db_spectra[th_cross_down, i], 'o', color='orange')
        # plt.plot([spec_freq[0], spec_freq[-1]], [db_th0, db_th1], 'k-')
        # plt.show()

def main(folder):
    freq_res = 1
    overlap_frac = 0.8

    # load data
    filename, data, samplerate, channels = load_data(folder)

    start_idx = int(120 * 60 * samplerate)
    analysis_window_sec = 5 * 60
    end_idx = int(start_idx + analysis_window_sec * samplerate)

    # spectrogram
    comb_spectra, db_comb_spectra, tmp_times, spec_freqs = compute_spectrogram(data, samplerate, freq_res, overlap_frac,
                                                                               channels, start_idx, end_idx)

    # detection
    f0, f1 = 400, 1200
    db_th0 = -90

    detection_mask = np.full(db_comb_spectra.shape, np.nan)
    detection_mask[db_comb_spectra >= db_th0] = 1

    db_th2 = -90
    detection_mask2 = np.full(db_comb_spectra.shape, np.nan)
    detection_mask2[db_comb_spectra >= db_th2] = 1

    ###############################

    f_mask = np.arange(len(spec_freqs))[(spec_freqs >= f0) & (spec_freqs <= f1)]
    f_mask2 = np.arange(len(spec_freqs))[(spec_freqs >= f0*2) & (spec_freqs <= f1*2)]
    t_mask = np.arange(len(tmp_times))

    fig = plt.figure(figsize=(20/2.54, 14/2.54))
    gs = gridspec.GridSpec(2, 1, left=0.1, bottom = 0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 0])

    h = ax.imshow(db_comb_spectra[f_mask[0]:f_mask[-1]+1, t_mask[0]:t_mask[-1]+1][::-1],
                  extent=[tmp_times[0], tmp_times[-1], spec_freqs[f_mask[0]], spec_freqs[f_mask[-1]+1]],
                  aspect='auto', vmin=-100, vmax=-50, alpha=0.7, cmap='jet', interpolation='gaussian')

    h1 = ax2.imshow(db_comb_spectra[f_mask2[0]:f_mask2[-1]+1, t_mask[0]:t_mask[-1]+1][::-1],
                    extent=[tmp_times[0], tmp_times[-1], spec_freqs[f_mask2[0]], spec_freqs[f_mask2[-1]+1]],
                    aspect='auto', vmin=-100, vmax=-50, alpha=0.7, cmap='jet', interpolation='gaussian')

    m = ax.imshow(detection_mask[f_mask[0]:f_mask[-1]+1, t_mask[0]:t_mask[-1]+1][::-1],
                  extent=[tmp_times[0], tmp_times[-1], spec_freqs[f_mask[0]], spec_freqs[f_mask[-1]+1]],
                  aspect='auto', vmin=0, vmax=1, alpha=1, cmap='binary')

    m1 = ax2.imshow(detection_mask2[f_mask[0]:f_mask[-1]+1, t_mask[0]:t_mask[-1]+1][::-1],
                  extent=[tmp_times[0], tmp_times[-1], spec_freqs[f_mask2[0]], spec_freqs[f_mask2[-1]+1]],
                  aspect='auto', vmin=0, vmax=1, alpha=1, cmap='binary')

    ax.set_xlabel('time', fontsize=12)
    plt.show()

    EOD_detection(db_comb_spectra, spec_freqs, tmp_times, db_th0, db_th2, min_freq = 250, max_freq = 1200)

    embed()
    quit()
    pass

if __name__ == '__main__':
    main(sys.argv[1])