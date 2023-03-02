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

def define_globals():
    global min_group_size
    min_group_size = 2

    global max_divisor
    max_divisor = 2

    global breaker
    breaker = False

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
@cuda.jit('void(f4[:], f4[:], f4[:], f4)', device=True)
def detect_peaks_fixed(data, peaks, trough, th):

    # peaks = []
    # troughs = []

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
            if data[i] <= max_value - threshold:
                peaks[max_inx] = 1
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
            if data[i] >= min_value + threshold:
                trough[min_inx] = 1
                last_min_idx = min_inx
                trough_count += 1
                # change direction:
                direction = +1
                # store maximum element:
                max_inx = i
                max_value = data[i]

        # don't know direction yet:
        if direction == 0:
            if data[i] <= max_value - threshold:
                direction = -1  # falling
            if data[i] >= min_value + threshold:
                direction = 1  # rising

            if data[i] > max_value:
                # update maximum element:
                max_inx = i
                max_value = data[i]
            if data[i] < min_value:
                # update minimum element:
                min_inx = i
                min_value = data[i]

    if peak_count > trough_count:
        peaks[last_max_idx] = 0
    elif peak_count < trough_count:
        trough[last_min_idx] = 0
    else:
        pass

@cuda.jit('void(f4[:,:], f4[:,:], f4[:,:], f4)')
def peak_detekt_coordinater(peaks, troughs, spec, low_th):
    i = cuda.grid(1)
    if i < spec.shape[0]:
        detect_peaks_fixed(spec[i], peaks[i], troughs[i], low_th)
        # detect_peaks_fixed(spec[i], peaks[i], low_th)
    # tester(spec[i], peaks[i], troughs[i])
    cuda.syncthreads()

####################################################################
#3)
@cuda.jit('f8(f8[:,:], f8)', device=True)
def test(peak_candidates, good_count):
    r = 1
    return r

@cuda.jit(device=True)
def group_candidate(peak_candidates, freq, divisor, freq_tol, max_freq_tol, fzero, fzero_harmonics, new_group, new_penalties):
    # ToDo: issue of device functions: I can alter stuff, but not access it !!!
    # ToDo: I can only send hard code out !!!



    # 1. find harmonics in good_freqs and adjust fzero accordingly:
    group_size = min_group_size if divisor <= min_group_size else divisor

    fzero = freq
    fzero_harmonics = 1

    good_count = 0
    for i in range(len(peak_candidates)):
        good_count += peak_candidates[i, 4]
    if good_count > 0:
        prev_freq = divisor * freq
        breaker = 0
        for h in range(divisor + 1, 2 * min_group_size + 1):
            min_df = 1e6
            idx = 0
            for j in range(len(peak_candidates)):
                if peak_candidates[j, 4] == 0: # TODO: CHECK !!!
                    continue
                if abs(peak_candidates[j, 0]/h - fzero) < min_df:
                    min_df = abs(peak_candidates[j, 0]/h - fzero)
                    idx = j
                else:
                    pass
            ff = peak_candidates[idx, 0]
            if abs(ff/h - fzero) > freq_tol:
                continue
            df = ff - prev_freq
            dh = round(df / fzero)
            fe = abs(df / dh - fzero)

            if fe > 2.0*freq_tol:
                if h > min_group_size:
                    breaker = 1

            if breaker == 0:
                prev_freq = ff
                fzero_harmonics = h
                fzero = ff / fzero_harmonics

    # 2. check fzero:
    # freq might not be in our group anymore, because fzero was adjusted:
    if abs(freq - fzero) < freq_tol:
        freqs = cuda.local.array(shape=(min_group_size,), dtype=float64)
        # next_freq_idx = 0
        peak_candidates[0, 5] = 0
        prev_h = 0
        prev_fe = 0.0

        for h in range(1, min_group_size + 1):
            penalty = 0
            i = 0
            min_df = 1e6
            for j in range(len(peak_candidates)):
                if abs(peak_candidates[j, 0]/h - fzero) < min_df:
                    min_df = abs(peak_candidates[j, 0]/h - fzero)
                    i = j
            f = peak_candidates[i, 0]
            fac = 1.0 if h >= divisor else 2.0
            fe = abs(f/h - fzero)
            if fe > fac*max_freq_tol:
                continue
            if fe > fac*freq_tol:
                penalty = (fe - (fac*freq_tol)) / (fac*max_freq_tol - fac*freq_tol)

            if peak_candidates[0, 5] > 0:
                pf = freqs[peak_candidates[0, 5] - 1]
                df = f - pf
                if df < 0.5 * fzero:
                    if peak_candidates[0, 5] > 0:
                        pf = freqs[peak_candidates[0, 5]-2]
                        df = f - pf
                    else:
                        pf = 0.0
                        df = h*fzero
                dh = math.floor(df/fzero + 0.5)
                fe = math.fabs(df/dh - fzero)
                if fe > 2*dh*fac*max_freq_tol:
                    continue
                if fe > 2*dh*fac*freq_tol:
                    penalty = (fe - (dh*fac*freq_tol)) / (2*dh*fac*max_freq_tol - dh*fac*freq_tol)
            else:
                fe = 0.0

            if h > prev_h or fe < prev_fe:
                if prev_h <= 0 and h - prev_h <= 1:
                    if h == prev_h and peak_candidates[0, 5] > 0:
                        freqs[peak_candidates[0, 5] - 1] = -1
                        peak_candidates[0, 5] -= 1
                    freqs[peak_candidates[0, 5]] = f
                    peak_candidates[0, 5] = peak_candidates[0, 5] + 1

                    new_group[peak_candidates[0, 5]] = i
                    new_penalties[peak_candidates[0, 5]] = penalty
                    # prev_h = h
                    prev_fe = fe

        # 4. check new group:

        # almost all harmonics in min_group_size required:


@cuda.jit('f8[:,:], f8, f8', device=True)
def build_harmonic_groups(peak_candidates, freq_tol, max_freq_tol):
    fmaxidx = 0
    for i in range(peak_candidates.shape[0]):
        if peak_candidates[i, 4] == 1:
            peak_candidates[i, 5] = 1
            if peak_candidates[i, 1] > peak_candidates[fmaxidx, 1]:
                fmaxidx = i
    fmax = peak_candidates[fmaxidx, 0]
            # print(peak_candidates[i, 0])

    # container for harmonic groups:
    # print(min_group_size)
    d1 = min_group_size if min_group_size >= max_divisor else max_divisor

    best_group = cuda.local.array(shape=(d1,), dtype=float64)
    best_value = -1e6
    best_divisor = 0
    best_fzero = 0.0
    best_fzero_harmonics = 0

    fzero = 0.
    new_group = cuda.local.array(shape=(d1,), dtype=float64)
    new_penalties = cuda.local.array(shape=(d1,), dtype=float64)
    for i in range(len(new_group)):
        new_group[i] = -1
        new_group[i] = 1
    fzero_harmonics = 0

    for divisor in range(1, max_divisor + 1):
        freq = fmax / divisor
        group_candidate(peak_candidates, freq, divisor, freq_tol, max_freq_tol, fzero, fzero_harmonics, new_group, new_penalties)
        # peak_candidates[0, 5] = fzero_harmonics


@cuda.jit("f8[:,:], f8, f8, f8, f8", device=True)
def harmonic_groups(peak_candidates, mains_freq, mains_freq_tol, freq_tol, max_freq_tol):
    # peak_candidates[time, peak no, [freq, peak, count???, trough, good, helper_mask]]
    good_count = 0
    max_idx = peak_candidates.shape[0]
    # print(mains_freq, mains_freq_tol)
    for i in range(len(peak_candidates)):
        if peak_candidates[i, 4] == 1:
            good_count += 1
        if math.isnan(peak_candidates[i, 0]):
            max_idx = i
            break

    peak_candidates[:, 2] = 0.0

    if mains_freq > 0.0:
        for i in range(max_idx):
            if peak_candidates[i, 4] == 0:
                continue
            if abs(peak_candidates[i, 0] - round(peak_candidates[i, 0]/mains_freq)*mains_freq) < mains_freq_tol:
                peak_candidates[i, 4] = 0
                good_count -= 1

    # print(sum(peak_candidates[:, 4]))
    # print(good_count)

    first = True
    while good_count > 0:
        # print(good_count)
        # ToDo: [check freq] implementations
        build_harmonic_groups(peak_candidates, freq_tol, max_freq_tol)
        good_count = 0

@cuda.jit("void(f8[:,:,:], f8, f8, f8, f8)")
def harmonic_group_coordinater(peak_candidates, mains_freq, mains_freq_tol, freq_tol, max_freq_tol):
    i = cuda.grid(1)
    # if i < peak_candidates.shape[0]:
    if i < 1:
        harmonic_groups(peak_candidates[i], mains_freq, mains_freq_tol, freq_tol, max_freq_tol)

    cuda.syncthreads()

####################################################################

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
    spec = np.load("./spec.npy")
    spec_freq = np.load("./spec_freqs.npy")
    spec_times = np.load("./spec_time.npy")
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
    peak_detekt_coordinater[bpg, tpb](g_peaks, g_troughs, g_log_spec, low_threshold)
    # peak_detekt_coordinater[griddim, blockdim](g_peaks, g_log_spec, low_threshold)

    peaks = g_peaks.copy_to_host()
    troughs = g_troughs.copy_to_host()
    print(f'time took on gpu: {time.time() - t0:.4f}s')

    t0 = time.time()
    # all_freqs = cuda.device_array(shape=(g_peaks.shape[0], int(max(np.sum(g_peaks, axis=1))), 5))
    all_freqs = np.full(shape=(peaks.shape[0], int(np.max(np.sum(peaks, axis=1))), 6), fill_value=math.nan)
    # [t][freq_idx][freq, peak, count???, trough, good]

    for i in range(peaks.shape[0]):
        all_freqs[i, :, 4] = 0
        all_freqs[i, :len(spec_freq[peaks[i] == 1]), 0] = spec_freq[peaks[i] == 1]
        all_freqs[i, :len(spec_freq[peaks[i] == 1]), 1] = log_spec[i][peaks[i] == 1]
        all_freqs[i, :len(spec_freq[peaks[i] == 1]), 3] = log_spec[i][troughs[i] == 1]
        all_freqs[i, :len(spec_freq[peaks[i] == 1]), 4] = log_spec[i][peaks[i] == 1] - log_spec[i][troughs[i] == 1] > high_threshold
    print(f'helper_t: {time.time() - t0:.4f}s')
    # all_freqs[]

    # all_freqs[time, peak no, [freq, peak, count???, trough, good]]
    #############################################################################################

    # all_freqs = np.load('all_freqs.npy')
    tpb = 1024
    bpg = all_freqs.shape[0]

    # embed()
    # quit()

    ### from .cfg ###
    main_freqs = 50.
    main_freqs_tol = 1.
    freq_tol_fac = 1.
    max_freq_tol = 1.
    global min_group_size
    min_group_size = int(2)
    # min_group_size = int(2)
    # max_divisor = int(2)
    #################
    delta_f = spec_freq[1] - spec_freq[0]
    freq_tol = delta_f*freq_tol_fac
    if max_freq_tol < 1.1*freq_tol:
        max_freq_tol = 1.1*freq_tol

    g_all_freqs = cuda.to_device(all_freqs)
    harmonic_group_coordinater[bpg, tpb](g_all_freqs, main_freqs, main_freqs_tol, freq_tol, max_freq_tol)


    # harmonic_groups(all_freqs[0], 50., 1.)




    ### --- CPU COMPARISON --- ###
    # t0 = time.time()
    # core_count = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(core_count - 1)
    # a = pool.map(harmonic_groups_test_cpu, spec)
    # print(f'time took on cpu: {time.time() - t0:.4f}s')

    # troughs = g_peaks.copy_to_host()
    # print(f'time took on gpu: {time.time() - t0:.4f}s')
    #
    # t0 = time.time()
    # x, y = np.where(peaks == 1)
    # print(f'test: {time.time() - t0:.4f}s')
    #
    # # good_peaks = np.copy(peaks)
    # # good_peaks[log_spec[np.array(peaks, dtype=bool)] - log_spec[np.array(peaks, dtype=bool)]]
    #
    # good_g_peaks = cuda.device_array(shape=g_peaks.shape, dtype=bool)
    # embed()
    # quit()
    ########################################################
    embed()
    quit()

    fig, ax = plt.subplots(2, 1, figsize=(30/2.54, 20/2.54), sharex='all', sharey='all')
    for i in range(peaks.shape[0]):
        ax[0].plot(np.ones(len(spec_freq[peaks[i] == 1])) * i, spec_freq[peaks[i] == 1], 'o', color='red')

    # fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54))
    for i in range(len(a)):
        ax[1].plot(np.ones(len(a[i][0]))*i, spec_freq[a[i][0]], 'o', color='red')

    plt.show()
    embed()
    quit()

    pass

if __name__ == '__main__':
    main()
