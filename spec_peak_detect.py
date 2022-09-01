import os
import sys
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from functools import partial
from IPython import embed
from tqdm import tqdm
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

    # ToDo: return full spectrum
    comb_spectra = np.sum(spectra, axis=0)
    db_comb_spectra = decibel(comb_spectra)
    tmp_times = spec_times + (start_idx / samplerate)

    return comb_spectra, db_comb_spectra, tmp_times, spec_freqs


def clean_peaks(peaks, trough, spec_freq, power, peak_trough_max_df = 10, peak_power_th=5):

    peaks = peaks[np.argsort(power[peaks])]
    power_diff = np.repeat(power[peaks][:, np.newaxis], len(trough), axis=1) - np.repeat(power[trough][np.newaxis, :], len(peaks), axis=0)
    freq_diff = np.abs(np.repeat(spec_freq[peaks][:, np.newaxis], len(trough), axis=1) - np.repeat(spec_freq[trough][np.newaxis, :], len(peaks), axis=0))

    mask = (freq_diff < peak_trough_max_df) & (power_diff < peak_power_th)
    embed()

    i0s, i1s = mask.nonzero()
    del_peaks = []
    for i0, i1 in zip(i0s, i1s):
        if mask[i0, i1] == True:
            del_peaks.append(i0)
            mask[:, i1] = False
            mask[i0, :] = False

    real_peaks_idx = np.array(list(set(np.arange(len(peaks))) - set(del_peaks)), dtype=int)
    return peaks[real_peaks_idx], trough


def harmonic_groups(peaks, spec_freq, power, min_freq, max_freq, plot=True):
    peak_freq = spec_freq[peaks]
    peak_power = power[peaks]

    mask = np.ones(len(peaks), dtype=bool)

    if plot:
        fig, ax = plt.subplots(figsize=(30 / 2.54, 20 / 2.54))
        ax.plot(spec_freq, power)
        ax.plot(spec_freq[peaks], power[peaks], 'o', color='forestgreen')

        plt.show()

    pass


def eod_detection(db_spectra, spec_freq, spec_time, db_th0, db_th1, min_freq = 250, max_freq = 1200, min_group_size=2, plot=False):
    groups = []
    fundamentals = []
    f_mask = np.arange(len(spec_freq))[(spec_freq <= 3000)]

    db_spectra = db_spectra[f_mask[0]:f_mask[-1]+1, np.arange(np.shape(db_spectra)[1])]
    spec_freq = spec_freq[f_mask]

    for i in tqdm(np.arange(np.shape(db_spectra)[1]), desc='EOD extract'):
        #  threshold mask for power
        psd_mask = np.array(db_spectra[:, i] >= db_th0, dtype=bool)

        # include 1 entry before and after valid (to valid bool-mask)
        help_array = np.diff(np.array(psd_mask, dtype=int))
        psd_mask[np.arange(len(help_array))[help_array == 1]] = 1
        psd_mask[np.arange(len(help_array))[help_array == -1] + 1] = 1

        # 1st-derivation of valid psd (zeros crossings are peaks in PSDs)
        psd_derivation = np.diff(db_spectra[:, i])
        psd_derivation[~psd_mask[1:]] = np.nan

        peaks = np.arange(len(psd_derivation)-1)[(psd_derivation[:-1] > 0) & (psd_derivation[1:] < 0)] + 1
        trough = np.arange(len(psd_derivation)-1)[(psd_derivation[:-1] <= 0) & (psd_derivation[1:] > 0)] + 1

        all_peaks = peaks
        peaks, trough = clean_peaks(peaks, trough, spec_freq, db_spectra[:, i])

        # ToDo: Harmonic groups
        harmonic_groups(peaks, spec_freq, db_spectra[:, i], min_freq, max_freq)

        #############################
        f_peaks = peaks[(spec_freq[peaks] >= min_freq) & (spec_freq[peaks] <= max_freq)]
        f_trough = trough[(spec_freq[trough] >= min_freq) & (spec_freq[trough] <= max_freq)]
        h1_peaks = np.array(list(set(peaks) - set(f_peaks)), dtype=int)
        h1_trough = np.array(list(set(trough) - set(f_trough)), dtype=int)

        if plot:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(30/2.54, 20/2.54))
            ax[0].plot(spec_freq, db_spectra[:, i])
            ax[0].plot([spec_freq[0], spec_freq[-1]], [db_th0, db_th1], 'k-')

            # ax[1].plot(spec_freq[psd_mask][1:], psd_derivation)
            ax[1].plot(spec_freq[1:], np.diff(db_spectra[:, i]), color='grey')
            ax[1].plot(spec_freq[1:], psd_derivation)
            ax[1].plot([spec_freq[0], spec_freq[-1]], [0, 0], 'k--', lw=1, alpha=0.8)


            ax[0].plot(spec_freq[all_peaks], db_spectra[all_peaks, i], 'o', color='grey', alpha=0.5)
            ax[0].plot(spec_freq[f_peaks], db_spectra[f_peaks, i], 'o', color='forestgreen')
            ax[0].plot(spec_freq[f_trough], db_spectra[f_trough, i], 'o', markeredgecolor='forestgreen', color='none')
            ax[0].plot(spec_freq[h1_peaks], db_spectra[h1_peaks, i], 'o', color='orange')
            ax[0].plot(spec_freq[h1_trough], db_spectra[h1_trough, i], 'o', markeredgecolor='orange', color='none')
            ax[1].plot(spec_freq[peaks], np.zeros(len(peaks)), 'ko')
            ax[1].plot(spec_freq[trough], np.zeros(len(trough)), 'o', markeredgecolor='k', color='none')

            plt.show()


        fundamentals.append(spec_freq[f_peaks])

    return fundamentals


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
    min_freq, max_freq = 250, 1200
    db_th0 = -90

    detection_mask = np.full(db_comb_spectra.shape, np.nan)
    detection_mask[db_comb_spectra >= db_th0] = 1

    db_th2 = -90
    detection_mask2 = np.full(db_comb_spectra.shape, np.nan)
    detection_mask2[db_comb_spectra >= db_th2] = 1

    fundamentals = eod_detection(db_comb_spectra, spec_freqs, tmp_times, db_th0, db_th2, min_freq = min_freq, max_freq = max_freq)

    embed()
    quit()

    ###############################

    f_mask = np.arange(len(spec_freqs))[(spec_freqs >= min_freq) & (spec_freqs <= max_freq)]
    f_mask2 = np.arange(len(spec_freqs))[(spec_freqs >= min_freq*2) & (spec_freqs <= max_freq*2)]
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

    for i in range(len(fundamentals)):
        ax.plot(np.ones(len(fundamentals[i]))*tmp_times[i], fundamentals[i], '.', color='firebrick', markersize=8)

    ax.set_xlabel('time', fontsize=12)
    plt.show()

    pass

if __name__ == '__main__':
    main(sys.argv[1])