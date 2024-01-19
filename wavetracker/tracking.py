import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from plottools.colors import *
from plottools.tag import tag
colors_params(colors_muted, colors_tableau)
import os
import sys
import time
from IPython import embed
from tqdm import tqdm
from PyQt5.QtCore import *

from thunderfish.powerspectrum import decibel

def freq_tracking_v6(fund_v, idx_v, sign_v, times, freq_tolerance=2.5, max_dt=10., min_freq=200, max_freq=1200,
                     verbose=0, **kwargs):
    """
    Sorting algorithm tracking signal of wavetype electric fish detected in consecutive powespectra of single or
    multielectrode recordings using frequency difference in EOD frequency and frequnency-power amplitude difference
    across recording electodes as tracking parameters

    Signal tracking and identity assiginment is accomplished in four steps:
    1) Extracting possible amplitude difference distributions.
    2) Esitmate relative error between possible datapoint connections. Relative amplitude error is deducted from an
    amplitude error distribution, the relative frequnecy error from a blotzman function, both ranging from 0 to 1 for
    the purpose of compatibility.
    3) Calculate signal errors for all possible signals pairs in datawindow spanning 3*max_dt (30 sec in default mode).
    4) From connectins between signal pairs based on the signal error order starting from the smallest. Skip connections
    that would result in conflicing traces, i.e. a signal trace cannot have two freuqnecies at the same time.
    5) Since only the center third part of an analysis window (with a duration of max_dt) regards all possible signal
    partners (signals in the starting and end parts of the analysis window have potential partners outside the analysis
    window) for its fromed connections, singnal traces of only the central part (max_dt) of the analyisis window (3*max_dt)
    are valid and can be connected to signal traces from previous analysis windows. The order of these trace connections
    is again based on the smalles signal errors between signals of traces. Repeat these steps until the end of the
    recording.

    Parameters
    ----------
    fund_v : ndarray
        Contains the fundamental EOD frequencies of fish signal detected in a recording.
    idx_v : ndarray
        Time index a signal is detected at.
    sign_v : ndarray
        Contains the power of the respective signals fundamental EOD frequency accross recording electrodes.
        The first dimension corresponds to the signal, the second to the different electrodes accordingly.
    times : ndarray
        Accual time in second which is reffered to with parameter idx_v
    freq_tolerance: float
        Maximum tollerated frequency difference between a potential signal pair to be considered in Hz (default = 2.5).
    max_dt : float
        Maximum tollerated time difference between a potential signal pair to be considered in seconds (default = 10).
    min_freq : float
        Mininum fundamental EOD frequnecy of signals to be considered in this analyis (default = 200).
    max_freq : float
        Maximum fundamental EOD frequnecy of signals to be considered in this analyis (default = 1200).
    verbose : float
        Verbosity level regulating shell/logging feedback during analysis. Suggested for debugging in development.
    kwargs : dict
        Excess parameters from the configuration dictionary passed to the function.

    Returns
    -------
    ident_v: array
        Assigned identities of the signals passed to the algorithm.
    """

    def get_amplitude_error_dist(fund_v, idx_v, norm_sign_v, start_idx, idx_comp_range, min_freq, max_freq):
        """
        Collect a amplitude error distribution for a given data snippet (defined by start_idx and idx_compare_range).
        This distribution is later used to compute relative amplitude errors, by assessing how many amplitude errors
        in the distribution are smaller than a compare error.

        Parameters
        ----------
            fund_v : ndarray
                Contains the fundamental EOD frequencies of fish signal detected in a recording.
            idx_v : ndarray
                Time index a signal is detected at.
            norm_sign_v : ndarray
                Contains the normalized power of the respective signals fundamental EOD frequency accross recording
                electrodes, i.e. powers for each singal range from 0 to 1. The first dimension corresponds to the
                signal, the second to the different electrodes accordingly.
            start_idx : float
                First time index that should be regarded in the analysis.
            idx_comp_range : int
                Maximum time index difference between two potential signals to be considdered a pair.
            min_freq : float
                Minimum EOD frequency of signals to regarded in this analyis.
            max_freq : float
                Maximum EOD frequency of signals to regarded in this analyis.

        Returns
        -------
            a_error_distribution : ndarray
                Amplitude errors detected for all signal pairs in the corresponding data snippet.

        """
        a_error_distribution = []

        i0s = []
        i1s = []

        for i in tqdm(range(start_idx, int(start_idx + idx_comp_range * 3)), desc='error dist'):

            i0_v = np.arange(len(idx_v))[
                (idx_v == i) & (fund_v >= min_freq) & (fund_v <= max_freq)]  # indices of fundamtenals to assign
            i1_v = np.arange(len(idx_v))[
                (idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= min_freq) & (
                            fund_v <= max_freq)]  # indices of possible targets

            if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
                continue

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < min_freq or fund_v[i0_v[enu0]] > max_freq:
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < min_freq or fund_v[i1_v[enu1]] > max_freq:
                        continue
                    a_error_distribution.append(np.sqrt(np.sum(
                        [(norm_sign_v[i0_v[enu0]][k] - norm_sign_v[i1_v[enu1]][k]) ** 2 for k in
                         range(len(norm_sign_v[i0_v[enu0]]))])))
                    i0s.append(i0_v[enu0])
                    i1s.append(i1_v[enu1])

        a_error_distribution = np.array(a_error_distribution)
        return a_error_distribution

    def create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, min_freq, max_freq, update=False):
        """
        Generates or updates an errorc-cube containing the signal errors between potential origin and target signals in
        a predefined data snippet.

        Parameters
        ----------
            i0_m : list
                Indices of origin signals (i0_MATRIX) for consecutive time steps. This list is filled with i0_v, i.e.
                indices representing a signals freatures in the processed data vecotrs (fund_v, idx_v, ident_v etc.).
            i1_m : list
                Indices of target signals (i1_MATRIX) for corresponding origin signals. For each list in i0_m a list in i1_m
                is generated that contain the data indices (correspinding to fund_v etc.) representing potential target
                signals for the corresponding origin signals (defined by index_compare_range).
            error_cube : ndarray
                For each combination of origin and target signal (represented in i0_m and i1_m) contains the signal error.
                The first dimension represents time of the origin signal, the second the index of the origin signal (i0_m),
                and the third the target signal (i1_m).
            cube_app_idx : int
                Next time index that is currently not reprensented in error_cube/i0_m.
            min_freq : float
                Mininum fundamental EOD frequnecy of signals to be considered in this analyis.
            max_freq : float
                Maximum fundamental EOD frequnecy of signals to be considered in this analyis…
            update : bool
                If True: Pop the first layer of the error_cube, i0_m, and i1_m and append a next layer correspinding to
                the time index give with cube_append_idx

        Returns
        -------
            error_cube : ndarray
                For each combination of origin and target signal (represented in i0_m and i1_m) contains the signal error.
                The first dimension represents time of the origin signal, the second the index of the origin signal (i0_m),
                and the third the target signal (i1_m).
            i0_m : list
                Indices of origin signals (i0_MATRIX) for consecutive time steps. This list is filled with i0_v, i.e.
                indices representing a signals freatures in the processed data vecotrs (fund_v, idx_v, ident_v etc.).
            i1_m : list
                Indices of target signals (i1_MATRIX) for corresponding origin signals. For each list in i0_m a list in i1_m
                is generated that contain the data indices (correspinding to fund_v etc.) representing potential target
                signals for the corresponding origin signals (defined by index_compare_range).
            cube_app_idx : int
                Next time index that is currently not reprensented in error_cube/i0_m.

        """
        # ToDo: include all signals in 3*idx_comp_range into i0_m (instead of 2*idx_comp_range). But target signals
        #  (idx_v[i1_m]) shall still not exceed error_cube dimensions (as currently still is)

        if update:
            i0_m.pop(0)
            i1_m.pop(0)
            error_cube.pop(0)
            Citt = [cube_app_idx]

        else:
            error_cube = []  # [fundamental_list_idx, freqs_to_assign, target_freqs]
            i0_m = []
            i1_m = []
            Citt = np.arange(start_idx, int(start_idx + idx_comp_range * 2))

        for i in Citt:
            i0_v = np.arange(len(idx_v))[
                (idx_v == i) & (fund_v >= min_freq) & (fund_v <= max_freq)]  # indices of fundamtenals to assign
            i1_v = np.arange(len(idx_v))[
                (idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= min_freq) & (
                            fund_v <= max_freq)]  # indices of possible targets

            i0_m.append(i0_v)
            i1_m.append(i1_v)

            if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
                error_cube.append(np.array([[]]))
                continue

            error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < min_freq or fund_v[i0_v[enu0]] > max_freq:  # ToDo:should be obsolete
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < min_freq or fund_v[i1_v[enu1]] > max_freq:  # ToDo:should be obsolete
                        continue

                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue
                    a_error = np.sqrt(
                        np.sum([(norm_sign_v[i0_v[enu0]][j] - norm_sign_v[i1_v[enu1]][j]) ** 2 for j in range(channels)]))
                    f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                    t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                    rel_amplitude_error, rel_frequency_error = estimate_error(a_error, f_error, a_error_distribution)
                    error_matrix[enu0, enu1] = rel_amplitude_error + rel_frequency_error
            error_cube.append(error_matrix)

        if update:
            cube_app_idx += 1
        else:
            cube_app_idx = len(error_cube)

        return error_cube, i0_m, i1_m, cube_app_idx

    def get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, start_idx, idx_comp_range):
        """


        Parameters
        ----------
        i0_m : list
            Indices of origin signals (i0_MATRIX) for consecutive time steps. This list is filled with i0_v, i.e.
            indices representing a signals freatures in the processed data vecotrs (fund_v, idx_v, ident_v etc.).
        i1_m : list
            Indices of target signals (i1_MATRIX) for corresponding origin signals. For each list in i0_m a list in i1_m
            is generated that contain the data indices (correspinding to fund_v etc.) representing potential target
            signals for the corresponding origin signals (defined by index_compare_range).
        error_cube: ndarray
            For each combination of origin and target signal (represented in i0_m and i1_m) contains the signal error.
            The first dimension represents time of the origin signal, the second the index of the origin signal (i0_m),
            and the third the target signal (i1_m).
        fund_v : ndarray
            Contains the fundamental EOD frequencies of fish signal detected in a recording.
        idx_v : ndarray
            Time index a signal is detected at.
        start_idx: int
            First index of the current analysis window.
        idx_comp_range: int
            Maximum time index difference between two potential signals to be considdered a pair.

        Returns
        -------
        tmp_ident_v_ret: array
            Identity vector generated for the current analysis window.
        errors_to_v: array
            Contains for each assigned temporal identity the error value based on which this connection was established.

        """

        next_tmp_identity = 0

        max_shape = np.max([np.shape(layer) for layer in error_cube[1:]], axis=0)
        cp_error_cube = np.full((len(error_cube) - 1, max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube[1:]):
            cp_error_cube[enu, :np.shape(error_cube[enu + 1])[0], :np.shape(error_cube[enu + 1])[1]] = layer

        min_i0 = np.min(np.hstack(i0_m))
        max_i1 = np.max(np.hstack(i1_m))
        tmp_ident_v = np.full(max_i1 - min_i0 + 1, np.nan)
        errors_to_v = np.full(max_i1 - min_i0 + 1, np.nan)
        tmp_idx_v = idx_v[min_i0:max_i1 + 1]
        tmp_fund_v = fund_v[min_i0:max_i1 + 1]

        i0_m = np.array(i0_m, dtype=object) - min_i0
        i1_m = np.array(i1_m, dtype=object) - min_i0

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube, axis=None), np.shape(cp_error_cube))

        made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections[~np.isnan(cp_error_cube)] = 1

        layers = layers + 1

        i_non_nan = len(cp_error_cube[layers - 1, idx0s, idx1s][~np.isnan(cp_error_cube[layers - 1, idx0s, idx1s])])


        for enu, layer, idx0, idx1 in zip(np.arange(i_non_nan), layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):

            if np.isnan(cp_error_cube[layer - 1, idx0, idx1]):
                break

            # _____ some control functions _____ ###

            if tmp_idx_v[i1_m[layer][idx1]] - start_idx > idx_comp_range * 3:
                continue

            if np.isnan(tmp_ident_v[i0_m[layer][idx0]]):
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    tmp_ident_v[i0_m[layer][idx0]] = next_tmp_identity
                    tmp_ident_v[i1_m[layer][idx1]] = next_tmp_identity
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1
                    next_tmp_identity += 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

                else:
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]  # idxs of target
                    if tmp_idx_v[i0_m[layer][idx0]] in tmp_idx_v[mask]:  # if goal already in target continue
                        continue

                    same_id_idx = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
                    f_after = tmp_fund_v[same_id_idx[same_id_idx > i0_m[layer][idx0]]]
                    f_before = tmp_fund_v[same_id_idx[same_id_idx < i0_m[layer][idx0]]]
                    compare_freqs = []
                    if len(f_after) > 0:
                        compare_freqs.append(f_after[0])
                    if len(f_before) > 0:
                        compare_freqs.append(f_before[-1])

                    if len(compare_freqs) == 0:
                        continue
                    else:
                        if np.all(np.abs(np.array(compare_freqs) - tmp_fund_v[i0_m[layer][idx0]]) > 0.5):
                            continue

                    tmp_ident_v[i0_m[layer][idx0]] = tmp_ident_v[i1_m[layer][idx1]]

                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v


            else:
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    if tmp_idx_v[i1_m[layer][idx1]] in tmp_idx_v[mask]:
                        continue


                    tmp_ident_v[i1_m[layer][idx1]] = tmp_ident_v[i0_m[layer][idx0]]
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

                else:
                    if tmp_ident_v[i0_m[layer][idx0]] == tmp_ident_v[i1_m[layer][idx1]]:
                        if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                            errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                        continue

                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    idxs_i0 = tmp_idx_v[mask]
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
                    idxs_i1 = tmp_idx_v[mask]

                    if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                        continue

                    tmp_ident_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]] = tmp_ident_v[i1_m[layer][idx1]]

                    if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                        errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                        not_made_connections[layer - 1, idx0, idx1] = 0
                        made_connections[layer - 1, idx0, idx1] = 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v



        tmp_ident_v_ret = np.full(len(fund_v), np.nan)
        tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

        return tmp_ident_v_ret, errors_to_v


    def assign_tmp_ids(ident_v, tmp_ident_v, idx_v, fund_v, error_cube, idx_comp_range, next_identity, i0_m, i1_m,
                       min_freq, max_freq):
        """
        Appends valid parts of temporal identity vecor to indentity traces extracted in previous tracking steps.

        Parameters
        ----------
            ident_v : ndarray
                Vector which inferes identity of signals which are established with this algorithm. This vector is updated
                using tmp_ident_v in this step.
            tmp_ident_v: ndarray
                Identity vector generated for the current analysis window. The central part of this array shall be appended
                to already established identity traces in this analysis step.
            idx_v : ndarray
                Time index a signal is detected at.
            fund_v : ndarray
                Contains the fundamental EOD frequencies of fish signal detected in the respectrive recording.
            error_cube : ndarray
                For each combination of origin and target signal (represented in i0_m and i1_m) contains the signal error.
                The first dimension represents time of the origin signal, the second the index of the origin signal (i0_m),
                and the third the target signal (i1_m).
            idx_comp_range : int
                Maximum time index difference between two potential signals to be considdered a pair.
            next_identity : int
                Next identity to be assigned that is not present in ident_v so far.
            i0_m : list
                Indices of origin signals (i0_MATRIX) for consecutive time steps. This list is filled with i0_v, i.e.
                indices representing a signals freatures in the processed data vecotrs (fund_v, idx_v, ident_v etc.).
            i1_m : list
                Indices of target signals (i1_MATRIX) for corresponding origin signals. For each list in i0_m a list in i1_m
                is generated that contain the data indices (correspinding to fund_v etc.) representing potential target
                signals for the corresponding origin signals (defined by index_compare_range).
            min_freq : float
                Mininum fundamental EOD frequnecy of signals to be considered in this analyis (default = 200).
            max_freq : float
                Maximum fundamental EOD frequnecy of signals to be considered in this analyis (default = 1200).

        Returns
        -------
            ident_v : ndarray
                Vector which inferes identity of signals which are established with this algorithm. This vector is updated
                using tmp_ident_v in this step.
            next_identity : int
                Next identity to be assigned that is not present in ident_v so far.
        """
        max_shape = np.max([np.shape(layer) for layer in error_cube], axis=0)
        cp_error_cube = np.full((len(error_cube), max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube):
            cp_error_cube[enu, :np.shape(error_cube[enu])[0], :np.shape(error_cube[enu])[1]] = layer

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube[:idx_comp_range], axis=None),
                                                np.shape(cp_error_cube[:idx_comp_range]))

        i_non_nan = len(cp_error_cube[layers, idx0s, idx1s][~np.isnan(cp_error_cube[layers, idx0s, idx1s])])
        min_i0 = np.min(np.hstack(i0_m))
        max_i1 = np.max(np.hstack(i1_m))

        p_ident_v = ident_v[min_i0:max_i1 + 1] # this is a pointer: changes in p_ident_v changes also ident_v
        p_tmp_ident_v = tmp_ident_v[min_i0:max_i1 + 1]
        p_idx_v = idx_v[min_i0:max_i1 + 1]
        p_fund_v = fund_v[min_i0:max_i1 + 1]

        p_i0_m = np.array(i0_m, dtype=object) - min_i0
        p_i1_m = np.array(i1_m, dtype=object) - min_i0

        already_assigned = []
        for layer, idx0, idx1 in zip(layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):
            idents_to_assigne = p_ident_v[~np.isnan(p_tmp_ident_v) & (p_idx_v > start_idx + idx_comp_range) &
                                          (p_idx_v <= start_idx + idx_comp_range * 2)]

            if len(idents_to_assigne[np.isnan(idents_to_assigne)]) == 0:
                break

            if np.isnan(cp_error_cube[layer, idx0, idx1]):
                break

            if ~np.isnan(p_ident_v[p_i1_m[layer][idx1]]):
                continue

            if np.isnan(p_tmp_ident_v[p_i1_m[layer][idx1]]):
                continue

            if p_i1_m[layer][idx1] < idx_comp_range:
                if p_i1_m[layer][idx1] >= idx_comp_range * 2.:
                    print('impossible')
                    embed()
                    quit()
                continue

            if p_fund_v[p_i0_m[layer][idx0]] > max_freq or p_fund_v[p_i0_m[layer][idx0]] < min_freq:
                continue
            if p_fund_v[p_i1_m[layer][idx1]] > max_freq or p_fund_v[p_i1_m[layer][idx1]] < min_freq:
                continue

            if np.isnan(p_ident_v[p_i0_m[layer][idx0]]):
                continue

            idxs_i0 = p_idx_v[(p_ident_v == p_ident_v[p_i0_m[layer][idx0]]) & (p_idx_v > start_idx + idx_comp_range) &
                              (p_idx_v <= start_idx + idx_comp_range * 2)]
            idxs_i1 = p_idx_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) & (np.isnan(p_ident_v)) &
                              (p_idx_v > start_idx + idx_comp_range) & (p_idx_v <= start_idx + idx_comp_range * 2)]

            if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                continue

            if p_i1_m[layer][idx1] in already_assigned:
                continue

            already_assigned.append(p_i1_m[layer][idx1])

            p_ident_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) &
                      (np.isnan(p_ident_v)) & (p_idx_v > start_idx + idx_comp_range) &
                      (p_idx_v <= start_idx + idx_comp_range * 2)] = p_ident_v[p_i0_m[layer][idx0]]

        for ident in np.unique(p_tmp_ident_v[~np.isnan(p_tmp_ident_v)]):
            if len(p_ident_v[p_tmp_ident_v == ident][~np.isnan(p_ident_v[p_tmp_ident_v == ident])]) == 0:
                p_ident_v[(p_tmp_ident_v == ident) & (p_idx_v > start_idx + idx_comp_range) & (
                        p_idx_v <= start_idx + idx_comp_range * 2)] = next_identity
                next_identity += 1

        return ident_v, next_identity


    channels = np.shape(sign_v)[1]

    detection_time_diff = times[1] - times[0]
    dps = 1. / detection_time_diff
    idx_comp_range = int(np.floor(dps * max_dt))

    ident_v = np.full(len(fund_v), np.nan)

    norm_sign_v = (sign_v - np.min(sign_v, axis=1).reshape(len(sign_v), 1)) / (np.max(sign_v, axis=1).reshape(len(sign_v), 1) - np.min(sign_v, axis=1).reshape(len(sign_v), 1))


    # start_idx = 0 if not ioi_fti else idx_v[ioi_fti]  # Index Of Interest for temporal identities
    abs_start_idx = np.min(idx_v)
    start_idx = np.min(idx_v)  # Index Of Interest for temporal identities

    a_error_distribution = get_amplitude_error_dist(fund_v, idx_v, norm_sign_v, abs_start_idx, idx_comp_range, min_freq, max_freq)

    error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m=None, i1_m=None, error_cube=None, min_freq=min_freq,
                                                             max_freq=max_freq, cube_app_idx=None)

    next_identity = 0

    # for i in tqdm(np.arange(np.max(idx_v)+1), desc='tracking'):
    t0 = time.time()
    min_idx, max_idx = np.min(idx_v), np.max(idx_v)
    last_start_idx = min_idx
    for start_idx in tqdm(np.arange(np.min(idx_v), np.max(idx_v)+1), desc='tracking'):
        if verbose == 3:
            if time.time() - t0 > 5:
                print(f'{" ":^25}  Progress {(start_idx-min_idx) / (max_idx-min_idx):3.1%} '
                      f'({start_idx-min_idx:.0f}/{max_idx-min_idx:.0f})'
                      f'-> {start_idx-last_start_idx} itt/5s', end="\r")
                t0=time.time()
                last_start_idx = start_idx

        if len(np.hstack(i0_m)) == 0 or len(np.hstack(i1_m)) == 0:
            error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, min_freq, max_freq, update=True)
            continue

        if (abs_start_idx - start_idx) % idx_comp_range == 0:  # next total sorting step
            tmp_ident_v, errors_to_v = get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, start_idx, idx_comp_range)

            if abs_start_idx - start_idx == 0: # initial assignment of tmp_identities
                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    ident_v[(tmp_ident_v == ident) & (idx_v <= start_idx + idx_comp_range)] = next_identity
                    next_identity += 1

            # assing tmp identities ##################################
            ident_v, next_identity = assign_tmp_ids(ident_v, tmp_ident_v, idx_v, fund_v, error_cube, idx_comp_range,
                                                    next_identity, i0_m, i1_m, min_freq, max_freq)

        error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, min_freq, max_freq,
                                                                 update=True)
        # start_idx += 1

    return ident_v


def estimate_error(a_error, f_error, a_error_distribution):
    """
    Translate absolute signal errors to relative signal errors.

    Parameters
    ----------
        a_error : float
            Amplitude error between two electric fish signals corresponding to the euclidean distance between the power
            of the fundamental EOD frequencies across recording electrodes.
        f_error : float
            Frequency error as absolute difference in EOD frequency between signals.
        a_error_distribution : ndarray
            Distribution of ampluitude errors generated for a defined datasnippet at the beginning of the analysis used
            for relative classification of amplitude errors.

    Returns
    -------
        rel_amplitude_error : float
            Relative ampliude error that range from 0 to 1 depending on how many of amplitude errors in
            "a_error_distribution" are smaller than the given error.
        rel_frequency_error : float
            Relative frequency error that ranges from 0 to 1 depending on a bolzman function which approaches 1 after
            a certain threshold.

    """
    a_weight = 2. / 3
    f_weight = 1. / 3
    if len(a_error_distribution) > 0:
        rel_amplitude_error = a_weight * len(a_error_distribution[a_error_distribution < a_error]) / len(a_error_distribution)
    else:
        rel_amplitude_error = 1
    rel_frequency_error = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.35, dx=.08)

    return rel_amplitude_error, rel_frequency_error


def boltzmann(t, alpha=0.25, beta=0.0, x0=4, dx=0.85):
    """
    Calulates a boltzmann function. Used to translate absolute frequency errors to values between 0 and 1.

    Parameters
    ----------
        t: ndarray
            Time vector.
        alpha: float
            Max value of the boltzmann function.
        beta: float
            Min value of the boltzmann function.
        x0: float
            Time of turning point of the boltzmann function.
        dx: float
            Slope of the boltzman function.

    Returns
    -------
        boltz : ndarray
            Boltzmann function corresponding to input parameters.
    """

    boltz = (alpha - beta) / (1. + np.exp(- (t - x0) / dx)) + beta
    return boltz

def load_example_data(folder):
    """
    Load wavetracker structure example data for tracking.

    Parameters
    ----------
        folder : str
            Path to the data files

    Returns
    -------
        fund_v : ndarray
            Contains the fundamental EOD frequencies of fish signal detected in the respectrive recording.
        sign_v : ndarray
            Contains the power of the respective signals fundamental EOD frequency accross recording electrodes.
            The first dimension corresponds to the signal, the second to the different electrodes accordingly.
        idx_v : ndarray
            Time index a signal is detected at.
        times : ndarray
            Accual time in second which is reffered to with parameter idx_v
        start_time : float
            Start time of the analyzed data in seconds (usually obsolet since whole recordings are analyzed).
        end_time : float
            Endtime of the analyzed data (usually obsolet since whole recordings are analyzed).
    """

    if os.path.exists(os.path.join(folder, 'fund_v.npy')):
        fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
        sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
        idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
        times = np.load(os.path.join(folder, 'times.npy'))
        start_time, end_time = np.load(os.path.join(folder, 'meta.npy'))

    else:
        fund_v, sign_v, idx_v, times, start_time, end_time  = [], [], [], [], [], []
        print('WARNING !!! files not found !')

    return fund_v, sign_v, idx_v, times, start_time, end_time

def main():
    if len(sys.argv) <= 1:
        print('require data suitable for tracking using the wavetracker apporoch. See fn "load_example_data".')
    folder = sys.argv[1]

    # ---------------- Example data for tracking paper ---------------- #
    fund_v, sign_v, idx_v, times, start_time, end_time, validated_ident_v = load_example_data(folder)
    # ----------------------------------------------------------------- #

    ident_v = freq_tracking_v6(fund_v, idx_v, sign_v, times)


if __name__ == '__main__':
    main()
