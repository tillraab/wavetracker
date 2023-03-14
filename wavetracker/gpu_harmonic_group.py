import math
import sys
import time
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np
import cupy as cp
from numba import cuda, jit, float64, int64
from .config import Configuration

import random
# try:
#     from numba import cuda, jit
#     imported_cuda=True
# except ImportError:
#     imported_cuda=False
#     class cuda():
#         @classmethod
#         def jit(cls, *args, **kwargs):
#             def decorator_jit(func):
#                 return func
#             return decorator_jit
#

# min_group_size_glob = 2

from thunderfish.powerspectrum import decibel
from thunderfish.eventdetection import detect_peaks
from thunderfish.eventdetection import detect_peaks_fixed as dpf

def define_globals():
    cfg = Configuration()

    global max_f
    max_f = cfg.harmonic_groups['max_freq']

    global min_f
    min_f = cfg.harmonic_groups['min_freq']

    global min_group_size
    min_group_size = cfg.harmonic_groups['min_group_size']

    global max_divisor
    max_divisor = cfg.harmonic_groups['max_divisor']

    global mains_freq
    mains_freq = cfg.harmonic_groups['mains_freq']

    global mains_freq_tol
    mains_freq_tol = cfg.harmonic_groups['mains_freq_tol']

    global low_threshold
    low_threshold = 10.

    global high_threshold
    high_threshold = 15.

    global min_good_peak_power
    min_good_peak_power = cfg.harmonic_groups['min_good_peak_power']

define_globals()

####################################################################
# 1)
@cuda.jit('void(f4[:,:], f4[:,:])')
def jit_decibel(power, db_power):
    """Transform power to decibel relative to ref_power.

    \\[ decibel = 10 \\cdot \\log_{10}(power/ref\\_power) \\]
    Power values smaller than `min_power` are set to `-np.inf`.

    Parameters
    ----------
    power: float or array
        Power values, for example from a power spectrum or spectrogram.
    ref_power: float or None or 'peak'
        Reference power for computing decibel.
        If set to `None` or 'peak', the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `-np.inf`.

    Returns
    -------
    decibel_psd: array
        Power values in decibel relative to `ref_power`.
    """
    ref_power = 1.0
    min_power = 1e-20

    i, j = cuda.grid(2)
    if (i > power.shape[0]) or (j > power.shape[1]):
        return
    if power[i, j] <= min_power:
        db_power[i, j] = -math.inf
    else:
        db_power[i, j] = 10.0 * math.log10(power[i, j] / ref_power)

####################################################################
# 2)
@cuda.jit('void(f4[:], f4[:], f4[:], f8[:], f4)', device=True)
def detect_peaks_fixed(data, peaks, trough, spec_freq, th):
    # initialize:
    direction = 0

    min_inx = 0
    trough_count = 0
    last_min_idx = 0

    max_inx = 0
    peak_count = 0
    last_max_idx = 0

    min_value = data[0]
    max_value = min_value

    threshold = th

    p, t = 0, 0

    # loop through the data:
    # for index, value in enumerate(data):
    for i in range(len(data)):
        # rising?
        if direction > 0:
            if data[i] > max_value:
                # update maximum element:
                max_inx = i
                max_value = data[i]
            # otherwise, if the new value is falling below
            # the maximum value minus the threshold:
            # the maximum is a peak!
            if data[i] <= max_value - low_threshold:
                peaks[max_inx] = 1
                p = 1
                last_max_idx = max_inx
                peak_count += 1
                # change direction:
                direction = -1
                # store minimum element:
                min_inx = i
                min_value = data[i]

        # falling?
        if direction < 0:
            if data[i] < min_value:
                # update minimum element:
                min_inx = i
                min_value = data[i]
            # otherwise, if the new value is rising above
            # the minimum value plus the threshold:
            # the minimum is a trough!
            if data[i] >= min_value + low_threshold:
                trough[min_inx] = 1
                t = 1
                last_min_idx = min_inx
                trough_count += 1
                # change direction:
                direction = +1
                # store maximum element:
                max_inx = i
                max_value = data[i]

        # don't know direction yet:
        if direction == 0:
            if data[i] <= max_value - low_threshold:
                direction = -1  # falling
            if data[i] >= min_value + low_threshold:
                direction = 1  # rising

            if data[i] > max_value:
                # update maximum element:
                max_inx = i
                max_value = data[i]
            if data[i] < min_value:
                # update minimum element:
                min_inx = i
                min_value = data[i]

        # check if this is a good peak
        if p != 0 and t != 0:
            p, t = 0, 0
            # ddB > high_th
            if not data[last_max_idx] - data[last_min_idx] > high_threshold:
                continue
            # in freq boundaries
            if spec_freq[last_max_idx] < min_f or spec_freq[last_max_idx] > max_f*min_group_size:
                continue
            # not a main freq 1/2
            if spec_freq[last_max_idx] % mains_freq < mains_freq_tol:
                continue
            # not a main freq 2/2
            if abs(spec_freq[last_max_idx] % mains_freq - 50) < mains_freq_tol:
                continue
            # ToDo: Parameter?: min_good_peak_power
            # if data[last_max_idx] < min_good_peak_power:
            #     continue
            peaks[last_max_idx] = 2
            trough[last_min_idx] = 2

    if peak_count > trough_count:
        peaks[last_max_idx] = 0
    elif peak_count < trough_count:
        trough[last_min_idx] = 0
    else:
        pass

@cuda.jit('void(f4[:,:], f4[:,:], f4[:,:], f8[:], f4)')
def peak_detekt_coordinater(spec, peaks, troughs, spec_freq, low_th):
    i = cuda.grid(1)
    if i < spec.shape[0]:
        detect_peaks_fixed(spec[i], peaks[i], troughs[i], spec_freq, low_th)
        # detect_peaks_fixed(spec[i], peaks[i], low_th)
    # tester(spec[i], peaks[i], troughs[i])
    cuda.syncthreads()

########################################################################################

@cuda.jit("f4[:], i8[:]", device=True)
def get_value(log_spec, out):

    # min_group_size = 3
    peak_sum = 0
    n = 0

    for i in range(len(out)-1):
        if out[i] != 0:
            n += 1
            peak_sum += 10**log_spec[out[i]] / 10.
    peak_sum = math.log10(peak_sum * min_group_size / n)

    diff_sum = 0
    n = 0
    v0 = 0.
    for i in range(len(out)-1):
        if out[i] == 0:
            continue
        if v0 == 0:
            v0 = log_spec[out[i]]
            continue
        diff_sum += log_spec[out[i]] - v0
        v0 = log_spec[out[i]]
        n += 1
    diff_mean = diff_sum / n

    # print(diff_sum / n1)

    std_vals = 0
    n = 0
    v0 = 0.
    for i in range(len(out)-1):
        if out[i] == 0:
            continue
        if v0 == 0:
            v0 = log_spec[out[i]]
            continue
        # std_vals += ((log_spec[out[i]] - v0) - diff_mean)**2
        v0 = log_spec[out[i]]
        n += 1
    diff_std = (std_vals / (n-1))**0.5
    # print(peak_sum)

    # print(peak_sum)
    out[-1] = peak_sum / diff_std

@cuda.jit('void(f8, f4[:], f8[:], f4[:], i8[:])', device=True)
def get_group(freq, log_spec, spec_freqs, peaks, out):
    fzero = freq
    fzero_h = 1
    fe = 1e6
    ioi = 0
    max_freq_tol = 1.
    mains_freqs = 50.
    mains_freqs_tol = 1.

    # for devisor in range(1, max_devisor+1):
    for h in range(1, len(out)):
        for i in range(len(peaks)):
            if peaks[i] != 0:
                new_fe = abs(spec_freqs[i]/h - fzero/fzero_h)
                if new_fe < fe and new_fe < max_freq_tol:
                    ioi = i
                    fe = new_fe
                if new_fe > fe:
                    if ioi != 0:
                        fzero = spec_freqs[ioi]
                        fzero_h = h
                        out[h-1] = ioi
                        # if h <= 3:
                        #     if log_spec[ioi] > peak_power:
                        #         peak_power = log_spec[ioi]
                        #         out[-1] = peak_power
        ioi = 0
        fe = 1e6

    # min_group_size = 3
    peak_sum = 0
    n = 0
    nn = 0
    for i in range(min_group_size):
        if out[i] != 0:
            nn += 1
            if spec_freqs[out[i]] % mains_freqs < mains_freqs_tol or abs(spec_freqs[out[i]] % mains_freqs - 50) < mains_freqs_tol:
                continue
            else:
                n += 1
                peak_sum += log_spec[out[i]]
    # peak_mean = 10. * math.log10(peak_sum / n)
    peak_mean = peak_sum / n
    value = -1e6
    # out[-1] = -1e6
    if nn >= min_group_size - 1:
        # out[-1] = peak_mean
        value = peak_mean
    return value
@cuda.jit('void(f8[:,:], f4[:,:], f8[:], f4[:, :], i8[:,:,:], f8[:, :])')
def hg_coordinater(g_check_freqs, g_log_spec, spec_freq, peaks, out, value):
    i, j = cuda.grid(2)
    # if i < 1 and j < 1:
    if i < g_check_freqs.shape[0] and j < g_check_freqs.shape[1]:
        if g_check_freqs[i, j] != 0:
            value[i, j] = get_group(g_check_freqs[i, j], g_log_spec[i], spec_freq, peaks[i], out[i, j, :])
            # cuda.syncthreads()
            # get_value(g_log_spec[i], out[i, j, :])

###############################################################################

def harmonic_groups_test_cpu(psd, verbose=0,
                             low_threshold=10, high_threshold=0.0, thresh_bins=100,
                             low_thresh_factor=6.0, high_thresh_factor=10.0):

    if verbose > 0:
        print('')
        if verbose > 1:
            print(70*'#')
        print('##### harmonic_groups', 48*'#')

    # decibel power spectrum:
    log_psd = decibel(psd)
    max_idx = np.argmax(~np.isfinite(log_psd))
    if max_idx > 0:
        log_psd = log_psd[:max_idx]

    # thresholds:
    # detect peaks in decibel power spectrum:
    peaks, troughs = detect_peaks(log_psd, low_threshold)
    return peaks, troughs

def main():
    #############################################################################################
    # cuda function
    spec = np.load("./wavetracker/spec2.npy")
    spec_freq = np.load("./wavetracker/spec_freqs2.npy")
    spec_times = np.load("./wavetracker/spec_time2.npy")
    # embed()
    # quit()
    low_threshold = 10.
    high_threshold = 15.

    # g_low_threshold = cuda.to_device(low_threshold)

    t0 = time.time()
    device = cuda.get_current_device()

    spec = spec.transpose()
    g_spec = cuda.to_device(spec)
    g_log_spec = cuda.device_array_like(g_spec)
    g_spec_freq = cuda.to_device(spec_freq)

    ### logaritmic spec ###
    blockdim = (32, 32)
    griddim = (g_spec.shape[0] // blockdim[0] + 1, g_spec.shape[1] // blockdim[1] + 1)
    # griddim = (16, 129)
    jit_decibel[griddim, blockdim](g_spec, g_log_spec)
    log_spec = g_log_spec.copy_to_host()

    # embed()
    # quit()
    ### peak detection ###
    g_peaks = cuda.device_array_like(g_spec)
    # g_peaks = cuda.device_array(g_spec.shape, dtype=bool)
    g_troughs = cuda.device_array_like(g_spec)
    # g_troughs = cuda.device_array(g_spec.shape, dtype=bool)
    tpb = 1024
    bpg = g_spec.shape[0]
    # tpb = 1024
    # bpg = 1000
    peak_detekt_coordinater[bpg, tpb](g_log_spec, g_peaks, g_troughs, g_spec_freq, low_threshold)
    # peak_detekt_coordinater[griddim, blockdim](g_peaks, g_log_spec, low_threshold)

    peaks = g_peaks.copy_to_host()
    troughs = g_troughs.copy_to_host()
    print(f'time took on gpu: {time.time() - t0:.4f}s')


    # t0 = time.time()
    # all_freqs = cuda.device_array(shape=(g_peaks.shape[0], int(max(np.sum(g_peaks, axis=1))), 5))
    # all_freqs = np.full(shape=(peaks.shape[0], int(np.max(np.sum(peaks, axis=1))), 6), fill_value=math.nan)
    # # [t][freq_idx][freq, peak, count???, trough, good]
    #
    # # embed()
    # # quit()
    # for i in range(peaks.shape[0]):
    #     all_freqs[i, :, 4] = 0
    #     all_freqs[i, :len(spec_freq[peaks[i] == 1]), 0] = spec_freq[peaks[i] == 1]
    #     all_freqs[i, :len(spec_freq[peaks[i] == 1]), 1] = log_spec[i][peaks[i] == 1]
    #     all_freqs[i, :len(spec_freq[peaks[i] == 1]), 3] = log_spec[i][troughs[i] == 1]
    #     all_freqs[i, :len(spec_freq[peaks[i] == 1]), 4] = log_spec[i][peaks[i] == 1] - log_spec[i][troughs[i] == 1] > high_threshold
    # print(f'helper_t: {time.time() - t0:.4f}s')
    # all_freqs[]

    # all_freqs[time, peak no, [freq, peak, count???, trough, good]]




    # all_freqs = np.load('all_freqs.npy')

    # out = cuda.device_array_like(g_log_spec)
    # g_good_peaks = cuda.device_array_like(peaks)
    # max_devisor = 3
    # min_group_size = 2
    # devisor_groups = cuda.device_array(shape=(max_devisor, min_group_size), dtype=float64)
    # embed()
    # quit()
    t0 = time.time()
    # check_freqs = np.zeros(shape=(peaks.shape[0], 3*int(np.max(np.sum(peaks != 0, axis=1)))))
    check_freqs = np.zeros(shape=(peaks.shape[0], 3*int(np.max(np.sum(peaks == 2, axis=1)))))
    for i in range(len(peaks)):
        # fs = spec_freq[peaks[i] != 0]
        fs = spec_freq[peaks[i] == 2]
        fs = fs[(fs < max_f) & (fs > min_f)]
        # flist = []
        for d in range(max_divisor):
            check_freqs[i, d*len(fs):(d+1)*len(fs)] = fs/(d+1)
            # flist.extend(fs/(d+1))
        # unique_flist = np.unique(flist)
        # check_freqs[i, :len(unique_flist)]  = unique_flist
    g_check_freqs = cuda.to_device(check_freqs)
    # g_value = cuda.device_array_like(g_check_freqs)
    out = cuda.device_array(shape=(check_freqs.shape[0], check_freqs.shape[1], 5), dtype=int)
    value = cuda.device_array(shape=(check_freqs.shape[0], check_freqs.shape[1]), dtype=float)
    # embed()
    # quit()
    tpb = (32, 32)
    bpg = (check_freqs.shape[0] // tpb[0] + 1, check_freqs.shape[1] // tpb[1] + 1)

    hg_coordinater[bpg, tpb](g_check_freqs, g_log_spec, g_spec_freq, g_peaks, out, value)
    out_cpu = out.copy_to_host()
    value_cpu = value.copy_to_host()
    print(f'hg: {time.time() - t0:.4f}s')


    harmonic_helper = np.cumsum(out_cpu>0, axis= 2)



    fund_list = []
    for t in tqdm(range(out_cpu.shape[0])):
        # fig, ax = plt.subplots(figsize=(30/2.54, 18/2.54))
        # ax.plot(spec_freq[spec_freq < max_f * min_group_size], log_spec[t][spec_freq < max_f * min_group_size])
        # mask = (peaks[t] == 1) & (spec_freq < max_f * min_group_size)
        # ax.plot(spec_freq[mask], log_spec[t][mask], 'o', color='grey')
        # mask2 = (peaks[t] == 2) & (spec_freq < max_f * min_group_size)
        # ax.plot(spec_freq[mask2], log_spec[t][mask2], 'o', color='red')
        # plt.show()


        fund_list.append([])

        good_peaks = np.arange(len(peaks[t]))[peaks[t] == 2]
        entities = len(peaks[t][peaks[t] == 2])
        power_array = value_cpu[t, :entities*3]
        order = np.argsort(power_array)[::-1]
        # ToDo: Tolerance for missing harmonics ?! NO!!!
        order = order[harmonic_helper[t, order, min_group_size-1] == min_group_size]

        peak_idxs = np.arange(len(peaks[t]))[peaks[t] == 2]

        sorting_mask = np.argsort(log_spec[t][peak_idxs])[::-1]
        best_peak_idxs = peak_idxs[sorting_mask]

        assigned = np.zeros_like(peaks[t])


        for search_peak_idx in best_peak_idxs:
            for i in order:
                if search_peak_idx in out_cpu[t, i, :max_divisor]:
                    non_zero_h = np.arange(out_cpu.shape[2])[out_cpu[t, i] != 0] + 1
                    non_zero_idx = out_cpu[t, i, :5][out_cpu[t, i, :5] != 0]
                    if non_zero_h[0] != 1:
                        continue
                    # ToDo: papameter: max_double_use
                    if 2 in assigned[non_zero_idx]: # double use
                        continue

                    # ToDo: parameter: max_double_uses_in_group
                    if np.sum(assigned[non_zero_idx]) == 0:
                        assigned[non_zero_idx] += 1
                        print(out_cpu[t, i])
                        fund_list[-1].append(spec_freq[int(non_zero_idx[0] / non_zero_h[0])])
                        break
        # embed()
        # quit()
        # plt.show()

    embed()
    quit()
    fig, ax = plt.subplots(figsize=(30/2.54, 18/2.54))
    # ax.pcolormesh(t, f, spec, vmax=-50, vmin=-100)
    f1 = np.where(spec_freq > max_f*min_group_size)[0][0]
    ax.pcolormesh(spec_times, spec_freq[:f1], log_spec[:, :f1].transpose(), vmax=-50, vmin=-120, alpha=0.7, cmap='jet')
    for i in range(len(fund_list)):
        ax.plot(np.ones(len(fund_list[i]))*spec_times[i], fund_list[i], '.', color='k', zorder = 2)
        f = spec_freq[peaks[i] != 0]
        f = f[f < max_f*min_group_size]
        ax.plot(np.ones(len(f))*spec_times[i], f+0.1, '.', color='grey', zorder = 1)
    # for i in range(len(peaks)):
    #     f = spec_freq[peaks[i] != 0]
    #     f = f[f < max_f]
    #     ax.plot(np.ones(len(f))*spec_times[i], f, 'o', color='grey', zorder = 1)

    ax.set_ylim(min_f - 100, max_f*min_group_size + 100)
    embed()
    quit()


if __name__ == '__main__':
    main()
