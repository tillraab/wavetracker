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

# def define_globals():
#     cfg = Configuration()
#
#     global max_f
#     max_f = cfg.harmonic_groups['max_freq']
#
#     global min_f
#     min_f = cfg.harmonic_groups['min_freq']
#
#     global min_group_size
#     min_group_size = cfg.harmonic_groups['min_group_size']
#
#     global max_divisor
#     max_divisor = cfg.harmonic_groups['max_divisor']
#
#     global mains_freq
#     mains_freq = cfg.harmonic_groups['mains_freq']
#
#     global mains_freq_tol
#     mains_freq_tol = cfg.harmonic_groups['mains_freq_tol']
#
#     global low_threshold
#     low_threshold = 10.
#
#     global high_threshold
#     high_threshold = 15.
#
#     global min_good_peak_power
#     min_good_peak_power = cfg.harmonic_groups['min_good_peak_power']
#
#     global max_freq_tol
#     max_freq_tol = cfg.harmonic_groups['max_freq_tol']
# #
# #
# define_globals()

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
@cuda.jit('void(f4[:], f4[:], f4[:], f8[:], f8, f8, f8, f8, f8, f8, f8)', device=True)
def detect_peaks_fixed(data, peaks, trough, spec_freq, low_threshold, high_threshold, min_freq, max_freq,
                            mains_freq, mains_freq_tol, min_good_peak_power):
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
            # if spec_freq[last_max_idx] < min_f or spec_freq[last_max_idx] > max_f*min_group_size:
            if spec_freq[last_max_idx] < min_freq or spec_freq[last_max_idx] > max_freq:
                continue
            # not a main freq 1/2
            if spec_freq[last_max_idx] % mains_freq < mains_freq_tol:
                continue
            # not a main freq 2/2
            if abs(spec_freq[last_max_idx] % mains_freq - mains_freq) < mains_freq_tol:
                continue
            # ToDo: Parameter?: min_good_peak_power
            if data[last_max_idx] < min_good_peak_power:
                continue
            peaks[last_max_idx] = 2
            trough[last_min_idx] = 2

    if peak_count > trough_count:
        peaks[last_max_idx] = 0
    elif peak_count < trough_count:
        trough[last_min_idx] = 0
    else:
        pass

@cuda.jit('void(f4[:,:], f4[:,:], f4[:,:], f8[:], f8, f8, f8, f8, f8, f8, f8)')
def peak_detect_coordinater(spec, peaks, troughs, spec_freq, low_threshold, high_threshold, min_freq, max_freq,
                            mains_freq, mains_freq_tol, min_good_peak_power):
    i = cuda.grid(1)
    if i < spec.shape[0]:
        detect_peaks_fixed(spec[i], peaks[i], troughs[i], spec_freq, low_threshold, high_threshold, min_freq, max_freq,
                            mains_freq, mains_freq_tol, min_good_peak_power)
        # detect_peaks_fixed(spec[i], peaks[i], low_th)
    # tester(spec[i], peaks[i], troughs[i])
    cuda.syncthreads()

########################################################################################

@cuda.jit('f8(f8, f4[:], f8[:], f4[:], i8[:], i8, f8, f8, f8)', device=True)
def get_group(freq, log_spec, spec_freqs, peaks, out, min_group_size, max_freq_tol, mains_freq, mains_freq_tol):
    fzero = freq
    fzero_h = 1
    # fe = 1e6
    # ioi = 0
    # max_freq_tol = 1.
    # ToDo: globals or parameter !!!
    # mains_freq = 50.
    # mains_freq_tol = 1.

    # for devisor in range(1, max_devisor+1):
    for h in range(1, len(out)):
        ioi = 0
        fe = 1e6
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

    # min_group_size = 3
    peak_sum = 0
    n = 0
    nn = 0
    for i in range(min_group_size):
        if out[i] != 0:
            nn += 1
            if spec_freqs[out[i]] % mains_freq < mains_freq_tol or abs(spec_freqs[out[i]] % mains_freq - 50) < mains_freq_tol:
                continue
            else:
                n += 1
                peak_sum += log_spec[out[i]]

    if n != 0:
        peak_mean = peak_sum / n
    else:
        peak_mean = -1e6

    value = peak_mean if nn >= min_group_size - 1 else -1e6
    return value

@cuda.jit('void(f8[:,:], f4[:,:], f8[:], f4[:, :], i8[:,:,:], f8[:, :], i8, f8, f8, f8)')
def get_harmonic_groups_coordinator(g_check_freqs, g_log_spec, spec_freq, peaks, out, value,
                                    min_group_size, max_freq_tol, mains_freq, mains_freq_tol):
    i, j = cuda.grid(2)
    # if i < 1 and j < 1:
    if i < g_check_freqs.shape[0] and j < g_check_freqs.shape[1]:
        if g_check_freqs[i, j] != 0:
            value[i, j] = get_group(g_check_freqs[i, j], g_log_spec[i], spec_freq, peaks[i], out[i, j, :],
                                    min_group_size, max_freq_tol, mains_freq, mains_freq_tol)
            cuda.syncthreads()

###############################################################################

def get_fundamentals(assigned_hg, spec_freq):
    f_list = []
    for t in range(assigned_hg.shape[0]):
        f_list.append([])
        for hg in np.unique(assigned_hg[t]):
            if hg == 0:
                continue
            f_list[-1].append(spec_freq[assigned_hg[t] == hg][0])
    return f_list

def harmonic_group_pipeline(spec, spec_freq, cfg, verbose = 0):
    ### logaritmic spec ###
    if verbose >= 1: t0 = time.time()
    spec = spec.transpose()
    g_spec = cuda.to_device(spec)
    g_log_spec = cuda.device_array_like(g_spec)
    g_spec_freq = cuda.to_device(spec_freq)

    blockdim = (32, 32)
    griddim = (g_spec.shape[0] // blockdim[0] + 1, g_spec.shape[1] // blockdim[1] + 1)
    jit_decibel[griddim, blockdim](g_spec, g_log_spec)
    log_spec = g_log_spec.copy_to_host()
    if verbose >= 1: print(f'power log transform: {time.time() - t0:.4f}s')

    ### peak detection ###
    if verbose >= 1: t0 = time.time()
    g_peaks = cuda.device_array_like(g_spec)
    g_troughs = cuda.device_array_like(g_spec)
    tpb = 1024
    bpg = g_spec.shape[0]

    # Start an asynchronous data transfer from the CPU to the GPU
    stream = cuda.stream()
    cuda.memcpy_htod_async(g_log_spec, log_spec, stream)

    peak_detect_coordinater[bpg, tpb, stream](g_log_spec, g_peaks, g_troughs, g_spec_freq,
                                      float64(cfg.harmonic_groups['low_threshold']),
                                      float64(cfg.harmonic_groups['high_threshold']),
                                      float64(cfg.harmonic_groups['min_freq']),
                                      float64(cfg.harmonic_groups['max_freq']),
                                      float64(cfg.harmonic_groups['mains_freq']),
                                      float64(cfg.harmonic_groups['mains_freq_tol']),
                                      float64(cfg.harmonic_groups['min_good_peak_power']))
    peaks = g_peaks.copy_to_host()
    # troughs = g_troughs.copy_to_host()
    if verbose >= 1: print(f'peak_detect: {time.time() - t0:.4f}s')
    meminfo = cuda.current_context().get_memory_info()

    print("Currently allocated:", meminfo[0]/1024/1024, "MB")
    print("Total memory:", meminfo[1]/1024/1024, "MB")
    # print("Free memory:", meminfo[0]/1024/1024, "MB")

    ### harmonic groups ###
    if verbose >= 1: t0 = time.time()
    check_freqs = np.zeros(shape=(peaks.shape[0], cfg.harmonic_groups['max_divisor']*int(np.max(np.sum(peaks == 2, axis=1)))))
    for i in range(len(peaks)):
        fs = spec_freq[peaks[i] == 2]
        fs = fs[(fs < cfg.harmonic_groups['max_freq']) & (fs > cfg.harmonic_groups['min_freq'])]
        for d in range(cfg.harmonic_groups['max_divisor']):
            check_freqs[i, d*len(fs):(d+1)*len(fs)] = fs/(d+1)
    if verbose >= 1: print(f'get check freqs: {time.time() - t0:.4f}s')

    if verbose >= 1: t0 = time.time()
    # max_group_size = int(max_f * min_group_size // min_f)
    max_group_size = int(cfg.harmonic_groups['max_freq'] * cfg.harmonic_groups['min_group_size'] // cfg.harmonic_groups['min_freq'])

    g_check_freqs = cuda.to_device(check_freqs)
    out = cuda.device_array(shape=(check_freqs.shape[0], check_freqs.shape[1], max_group_size), dtype=int)
    value = cuda.device_array(shape=(check_freqs.shape[0], check_freqs.shape[1]), dtype=float)
    out_cpu = np.zeros(shape=(check_freqs.shape[0], check_freqs.shape[1], max_group_size), dtype=int)
    value_cpu = np.zeros(shape=(check_freqs.shape[0], check_freqs.shape[1]), dtype=float)

    #tpb = (32, 32)
    tpb = (32, 32)
    bpg = (check_freqs.shape[0] // tpb[0] + 1, check_freqs.shape[1] // tpb[1] + 1)
    get_harmonic_groups_coordinator[bpg, tpb](g_check_freqs, g_log_spec, g_spec_freq, g_peaks, out, value,
                                              int64(cfg.harmonic_groups['min_group_size']),
                                              float64(cfg.harmonic_groups['max_freq_tol']),
                                              float64(cfg.harmonic_groups['mains_freq']),
                                              float64(cfg.harmonic_groups['mains_freq_tol']))

    out.copy_to_host(out_cpu)
    value.copy_to_host(value_cpu)
    # embed()
    # cuda.memcpy_dtoh(out_cpu, out)
    # value_cpu = value.copy_to_host()
    if verbose >= 1: print(f'get harmonic groups: {time.time() - t0:.4f}s')

    ### assign harmonic groups ###
    harmonic_helper = np.cumsum(out_cpu>0, axis= 2)
    assigned_hg = np.zeros_like(peaks)
    # fund_list = []
    for t in range(out_cpu.shape[0]):
        next_hg = 1
        assigned = np.zeros_like(peaks[t])

        peak_idxs = np.arange(len(peaks[t]))[peaks[t] == 2]
        sorting_mask = np.argsort(log_spec[t][peak_idxs])[::-1]
        best_peak_idxs = peak_idxs[sorting_mask]

        search_freq_count = len(check_freqs[t][check_freqs[t] != 0])
        power_array = value_cpu[t, :search_freq_count]
        order = np.argsort(power_array)[::-1]
        # ToDo: Tolerance for missing harmonics ?! NO!!!
        order = order[harmonic_helper[t, order, cfg.harmonic_groups['min_group_size']-1] == cfg.harmonic_groups['min_group_size']]


        for search_peak_idx in best_peak_idxs:
            for i in order:
                if search_peak_idx in out_cpu[t, i, :cfg.harmonic_groups['min_group_size']]:
                    non_zero_h = np.arange(out_cpu.shape[2])[out_cpu[t, i] != 0] + 1
                    non_zero_idx = out_cpu[t, i][out_cpu[t, i] != 0]
                    if non_zero_h[0] != 1:
                        continue
                    # ToDo: papameter: max_double_use
                    if 2 in assigned[non_zero_idx]: # double use
                        continue

                    # ToDo: parameter: max_double_uses_in_group
                    if np.sum(assigned[non_zero_idx]) == 0:
                        assigned[non_zero_idx] += 1
                        assigned_hg[t, non_zero_idx] = next_hg
                        next_hg += 1
                        break
    return assigned_hg, peaks, log_spec



def main():
    #############################################################################################
    ### load example data ###
    spec = np.load("./wavetracker/spec2.npy")
    spec_freq = np.load("./wavetracker/spec_freqs2.npy")
    spec_times = np.load("./wavetracker/spec_time2.npy")

    cfg = Configuration()

    assigned_hg, peaks, log_spec = harmonic_group_pipeline(spec, spec_freq, cfg, verbose = 0)

    # for t in range(out_cpu.shape[0]):
    for t in range(10):
        fig, ax = plt.subplots(figsize = (30 / 2.54, 18 / 2.54))
        ax.plot(spec_freq[spec_freq < cfg.harmonic_groups['max_freq'] * cfg.harmonic_groups['min_group_size']],
                log_spec[t][spec_freq < cfg.harmonic_groups['max_freq'] * cfg.harmonic_groups['min_group_size']])
        for hg in np.unique(assigned_hg[t]):
            if hg == 0:
                continue
            else:
                f0 = spec_freq[assigned_hg[t] == hg][0]
                if f0 < cfg.harmonic_groups['min_freq']:
                    ax.plot(spec_freq[assigned_hg[t] == hg], log_spec[t][assigned_hg[t] == hg], 'o', markersize=8, mfc='none', mew=2,
                            label=f'{spec_freq[assigned_hg[t] == hg][0]:.2f}Hz')
                else:
                    ax.plot(spec_freq[assigned_hg[t] == hg], log_spec[t][assigned_hg[t] == hg], 'o', markersize=10,
                            label=f'{spec_freq[assigned_hg[t] == hg][0]:.2f}Hz')
        ax.legend(loc = 1)

        # plt.show()

    fundamentals = get_fundamentals(assigned_hg, spec_freq)
    # f_list = []
    # for t in range(assigned_hg.shape[0]):
    #     f_list.append([])
    #     for hg in np.unique(assigned_hg[t]):
    #         if hg == 0:
    #             continue
    #         f_list[-1].append(spec_freq[assigned_hg[t] == hg][0])


    fig, ax = plt.subplots(figsize=(30/2.54, 18/2.54))
    f1 = np.where(spec_freq > cfg.harmonic_groups['max_freq'] * cfg.harmonic_groups['min_group_size'])[0][0]
    ax.pcolormesh(spec_times, spec_freq[:f1], log_spec[:, :f1].transpose(), vmax=-50, vmin=-120, alpha=0.7, cmap='jet')
    for i in range(assigned_hg.shape[0]):
        ax.plot(np.ones(len(fundamentals[i]))*spec_times[i], fundamentals[i], '.', color='k', zorder = 2)
    ax.set_ylim(cfg.harmonic_groups['min_freq'] - 100, cfg.harmonic_groups['max_freq'] + 100)

    plt.show()

    embed()
    quit()


if __name__ == '__main__':
    main()
