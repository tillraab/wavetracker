import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thunderfish.powerspectrum import decibel
from IPython import embed
from tqdm import tqdm
import sys

def gauss(t, shift, sigma, size, norm = False):
    if not hasattr(shift, '__len__'):
        g = np.exp(-((t - shift) / sigma) ** 2 / 2) * size
        if norm:
            g /= np.sum(g)
        return g
    else:
        t = np.array([t, ] * len(shift))
        res = np.exp(-((t.transpose() - shift).transpose() / sigma) ** 2 / 2) * size
        s = np.sum(res, axis=1)
        res = res / s.reshape(len(s), 1)
        return res


def get_valid_ids(times, idx_v, ident_v, fund_v, valid_v, old_valid_ids, i0, stride, f_th, kde_th):
    # fig = plt.figure(figsize=(30/2.54, 18/2.54))
    # gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, width_ratios=[3, 1], hspace=0)
    # ax = []
    # ax.append(fig.add_subplot(gs[0, 0]))
    # ax.append(fig.add_subplot(gs[0, 1], sharey = ax[0]))

    window_t_mask = (times[idx_v] >= i0) & (times[idx_v] < i0 + stride)

    ff = fund_v[(~np.isnan(ident_v)) & (window_t_mask)]
    # tt = times[idx_v[(~np.isnan(ident_v)) & (window_t_mask)]]

    convolve_f = np.arange(400, 1200, 0.1)
    g = gauss(convolve_f, ff, sigma=f_th, size=1, norm=True)
    kde = np.sum(g, axis=0)

    if not kde_th:
        kde_th = np.max(g) * len(times[(times >= times[0]) & (times < times[0] + stride)]) * 0.05

    # ax[1].plot(kde, convolve_f)
    # ax[1].plot([th, th], [convolve_f[0], convolve_f[-1]], '--', color='k', lw=2)

    valid_f = convolve_f[kde > kde_th]

    valid_ids = []
    for id in tqdm(np.unique(ident_v[~np.isnan(ident_v)])):
        f = fund_v[(ident_v == id) & (window_t_mask)]
        if len(f) <= 1:
            continue
        t = times[idx_v[(ident_v == id) & (window_t_mask)]]
        valid = False
        if np.min(np.abs(valid_f - f[:, np.newaxis])) <= (convolve_f[1] - convolve_f[0]) / 2:
            valid = True
        elif id in old_valid_ids:
            valid = True
        else:
            pass

        if valid:
            c = np.random.rand(3)
            a = 1
            valid_ids.append([id, np.median(f), times[idx_v[(ident_v == id)]][0]])
            # ax[0].text(t[0], f[0], f'{id:.0f}', ha='center', va='bottom')
            valid_v[ident_v == id] = 1
        else:
            c = 'grey'
            a = 0.2

    #     ax[0].plot(t, f, color=c, alpha = a, marker='.')
    #
    # ax[0].set_ylim(np.min(ff)-10, np.max(ff)+10)
    # plt.show()
    return kde_th, np.array(valid_ids)


def connect_by_similarity(idx_v, ident_v, valid_ids, f_th):
    # valid_ids = np.array(valid_ids)
    d_med_f = np.abs(valid_ids[:, 1] - valid_ids[:, 1][:, np.newaxis])

    idx0s, idx1s = np.unravel_index(np.argsort(d_med_f, axis=None), np.shape(d_med_f))
    similar_mask = idx0s == idx1s
    idx0s = idx0s[~similar_mask]
    idx1s = idx1s[~similar_mask]

    for enu, (idx0, idx1) in enumerate(zip(idx0s, idx1s)):
        if np.abs(valid_ids[idx0, 1] - valid_ids[idx1, 1]) > f_th:
            break
        # print(f'{enu:.0f}: ids = {valid_ids[idx0, 0]:.0f} {valid_ids[idx1, 0]:.0f}; df = {np.abs(valid_ids[idx0, 1] - valid_ids[idx1, 1]):.1f}Hz')
        id0 = valid_ids[idx0, 0]
        id1 = valid_ids[idx1, 0]
        taken_idxs0 = idx_v[(ident_v == id0)]
        taken_idxs1 = idx_v[(ident_v == id1)]
        double_idx = np.in1d(taken_idxs0, taken_idxs1)
        combine = True

        # if len(taken_idxs0)*0.01 > len(taken_idxs1):
        #     combine = True
        # elif len(taken_idxs1)*0.01 > len(taken_idxs0):
        #     combine = True
        # elif np.sum(double_idx) < len(taken_idxs1)*0.01 and np.sum(double_idx) < len(taken_idxs0)*0.01:
        #     combine = True
        # else:
        #     pass

        if double_idx.any():
            combine = False

        if not combine:
            continue

        if valid_ids[idx0, 1] < valid_ids[idx1, 1]:
            # valid_ids[idx1, 0] = valid_ids[idx0, 0]
            valid_ids[:, 0][valid_ids[:, 0] == id1] = id0
            ident_v[ident_v == id1] = id0
        else:
            # valid_ids[idx0, 0] = valid_ids[idx1, 0]
            valid_ids[:, 0][valid_ids[:, 0] == id0] = id1
            ident_v[ident_v == id0] = id1

    previous_valid_ids = np.unique(valid_ids[:, 0])

    return previous_valid_ids, ident_v


def connect_with_overlap(fund_v, ident_v, valid_v, idx_v, times):
    time_tol = 5*60
    # time_tol = 0
    freq_tol = 1.

    valid_ids = []
    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        f = fund_v[(ident_v == id)]
        t = times[idx_v[(ident_v == id)]]
        valid_ids.append([id, len(f)])
    valid_ids = np.array(valid_ids)
    valid_ids[:, 1] = np.argsort(valid_ids[:, 1])

    connections_candidates = []

    for ii, jj in itertools.combinations(range(len(valid_ids)), r=2):
        id0 = valid_ids[ii, 0]
        id1 = valid_ids[jj, 0]

        t0 = times[idx_v[ident_v == id0]]
        t1 = times[idx_v[ident_v == id1]]

        t0_span = [t0[0] - time_tol, t0[-1] + time_tol]
        t1_span = [t1[0] - time_tol, t1[-1] + time_tol]

        t_overlap = False
        if t0_span[0] <= t1_span[0]:
            if t0_span[1] > t1_span[0]:
                t_overlap = True

        if not t_overlap and t1_span[0] <= t0_span[0]:
            if t1_span[1] > t0_span[0]:
                t_overlap = True

        if not t_overlap:
            continue

        ##############################################
        overlap_t = np.array(t0_span + t1_span)
        overlap_t = overlap_t[np.argsort(overlap_t)][1:3]

        f0 = fund_v[(ident_v == id0) & (times[idx_v] > overlap_t[0]) & (times[idx_v] < overlap_t[1])]
        f1 = fund_v[(ident_v == id1) & (times[idx_v] > overlap_t[0]) & (times[idx_v] < overlap_t[1])]

        if len(f0) <= 1 or len(f1) <= 1:
            continue

        f0_span = [np.min(f0) - freq_tol, np.max(f0) + freq_tol]
        f1_span = [np.min(f1) - freq_tol, np.max(f1) + freq_tol]

        f_overlap = False
        if f0_span[0] <= f1_span[0]:
            if f0_span[1] > f1_span[0]:
                f_overlap = True

        if f1_span[0] <= f0_span[0]:
            if f1_span[1] > f0_span[0]:
                f_overlap = True

        if not f_overlap:
            continue

        taken_idxs0 = idx_v[(ident_v == id0)]
        taken_idxs1 = idx_v[(ident_v == id1)]
        double_idx = np.in1d(taken_idxs0, taken_idxs1)
        combine = False
        # if len(taken_idxs0)*0.05 > len(taken_idxs1):
        #     combine = True
        # elif len(taken_idxs1)*0.05 > len(taken_idxs0):
        #     combine = True
        if np.sum(double_idx) < len(taken_idxs1)*0.01 or np.sum(double_idx) < len(taken_idxs0)*0.01:
            combine = True
        else:
            continue

        # fig, ax = plt.subplots()
        #
        # ax.plot(times[idx_v[ident_v == id1]], fund_v[ident_v == id1])
        # ax.plot(times[idx_v[ident_v == id0]], fund_v[ident_v == id0])
        #
        # f_range = [np.min(fund_v[ident_v == id0]), np.max(fund_v[ident_v == id0]),
        #            np.min(fund_v[ident_v == id1]), np.max(fund_v[ident_v == id1])]
        #
        # ax.plot(overlap_t, f0_span, lw=3, color='k')
        # ax.plot(overlap_t, f1_span, lw=3, color='k')
        # ax.plot(overlap_t[::-1], f0_span, lw=3, color='k')
        # ax.plot(overlap_t[::-1], f1_span, lw=3, color='k')
        # ax.set_ylim(np.min(f_range)-10, np.max(f_range) + 10)
        # plt.show()


        connections_candidates.append([id0, id1, valid_ids[ii, 1] + valid_ids[jj, 1]])

    connections_candidates = np.array(connections_candidates)

    for pair_no in np.argsort(connections_candidates[:, 2])[::-1]:
        id0 = connections_candidates[pair_no, 0]
        id1 = connections_candidates[pair_no, 1]

        taken_idxs0 = idx_v[(ident_v == id0)]
        taken_idxs1 = idx_v[(ident_v == id1)]

        double_idx = np.in1d(taken_idxs0, taken_idxs1)

        if np.sum(double_idx) < len(taken_idxs1)*0.01 or np.sum(double_idx) < len(taken_idxs0)*0.01:
            if len(taken_idxs0) >= len(taken_idxs1):
                ident_v[ident_v == id1] = id0
                connections_candidates[:, 0][connections_candidates[:, 0] == id1] = id0
                connections_candidates[:, 1][connections_candidates[:, 1] == id1] = id0
            else:
                ident_v[ident_v == id0] = id1
                connections_candidates[:, 0][connections_candidates[:, 0] == id0] = id1
                connections_candidates[:, 1][connections_candidates[:, 1] == id0] = id1

    # embed()
    # quit()

        # if len(taken_idxs0) >= len(taken_idxs1):
        #     ident_v[ident_v == id1] = id0
        #     valid_ids[:, 0][valid_ids[:, 0] == id1] = id0
        # else:
        #     ident_v[ident_v == id0] = id1
        #     valid_ids[:, 0][valid_ids[:, 0] == id0] = id1
        #
    return ident_v



def main(folder = None):
    if not folder:
        if os.path.exists("/home/raab/data/cleanup_test/2023-03-02-09_54"):
            folder = "/home/raab/data/cleanup_test/2023-03-02-09_54"
        elif os.path.exists("/home/raab/data/2023-03-02-09_54"):
            folder = "/home/raab/data/2023-03-02-09_54"
        else:
            print('no file found.')
            exit()
    fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
    idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
    ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
    loaded_ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))

    # fine_time = np.load(os.path.join(folder, 'fine_times.npy'))
    # fine_freq = np.load(os.path.join(folder, 'fine_freqs.npy'))
    # fine_spec_shape = np.load(os.path.join(folder, 'fine_spec_shape.npy'))
    # fine_spec = np.memmap(os.path.join(folder, 'fine_spec.npy'), dtype='float', mode='r',
    #                            shape=(fine_spec_shape[0], fine_spec_shape[1]), order='F')
    # for i0 in np.arange(3*60*60, times[-1], 10*60):

    valid_v = np.zeros_like(ident_v)
    stride = 10 * 60
    overlap = 0.1
    kde_f_res = 0.1
    f_th = 5
    kde_th = None
    previous_valid_ids = np.array([])

 #   for i0 in tqdm(np.arange(3*60*60, times[-1], int(stride*(1-overlap)))):
    for i0 in tqdm(np.arange(0, times[-1], int(stride*(1-overlap)))):

        kde_th, valid_ids = get_valid_ids(times, idx_v, ident_v, fund_v, valid_v, previous_valid_ids, i0, stride, f_th, kde_th)

        previous_valid_ids, ident_v = connect_by_similarity(idx_v, ident_v, valid_ids, f_th)


    fig = plt.figure(figsize=(30/2.54, 18/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))

    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        f = fund_v[ident_v == id]
        i = idx_v[ident_v == id]
        ax[0].text(times[i[0]], f[0], f'{id:.0f}', ha='center', va='bottom')
        ax[0].plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], marker='.')
    plt.show()

    embed()
    quit()
    ##############################################################
    valid_power = np.max(sign_v[valid_v == 1], axis=1)
    if np.min(valid_power) >= 0:
        valid_power = decibel(valid_power)
    density_v = np.zeros_like(valid_v)

    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        i = idx_v[ident_v == id]
        id_densities = 1 / np.diff(i)

        id_densities = np.concatenate((id_densities, np.array([np.median(id_densities)])))

        # density = len(i) / (i[-1] - i[0] + 1)
        # density_v[ident_v == id] = density
        density_v[ident_v == id] = id_densities

    fig, ax = plt.subplots()
    ax.plot(valid_power, density_v[valid_v == 1], '.')
    plt.show()

    ##############################################################
    mean_p = []
    mean_d = []
    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        i = idx_v[ident_v == id]
        d = len(i) / (i[-1] - i[0] + 1)
        p = np.mean(decibel(np.max(sign_v[ident_v == id], axis=1)))
        mean_d.append(d)
        mean_p.append(p)

    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0, wspace=0, height_ratios=[1, 3], width_ratios=[3, 1])

    ax = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_t.xaxis.set_visible(False)
    ax_r = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_r.yaxis.set_visible(False)
    ax.set_xlabel('power [db]', fontsize=12)
    ax.set_ylabel('density', fontsize=12)

    power_n, power_bins = np.histogram(valid_power, bins=500)
    ax_t.bar(power_bins[:-1] + (power_bins[1] - power_bins[0])/2, power_n, align='center', width=(power_bins[1] - power_bins[0])*0.9)
    percentiles = np.percentile(valid_power, (5, 10, 25, 50, 75, 90, 95))
    for p in percentiles:
        ax_t.plot([p, p], [0, np.max(power_n)], lw=2, color='k')

    most_common_valid_power = power_bins[np.argmax(power_n)]
    pct99_power = np.percentile(valid_power, 99.5)
    ax_t.plot([most_common_valid_power, most_common_valid_power], [0, np.max(power_n)], '--', color='red')
    ax_t.plot([pct99_power, pct99_power], [0, np.max(power_n)], '--', color='red')
    ax_t.plot([most_common_valid_power - (pct99_power-most_common_valid_power), most_common_valid_power - (pct99_power-most_common_valid_power)], [0, np.max(power_n)], '--', color='red')

    density_n, density_bins = np.histogram(density_v[valid_v ==1], bins=20)
    ax_r.barh(density_bins[:-1] + (density_bins[1] - density_bins[0])/2, density_n, align='center', height=(density_bins[1] - density_bins[0])*0.9)
    percentiles = np.percentile(density_v[valid_v ==1], (5, 10, 25, 50, 75, 90, 95))
    for p in percentiles:
        ax_r.plot([0, np.max(density_n)], [p, p], lw=2, color='k')


    H, xedges, yedges = np.histogram2d(valid_power, density_v[valid_v ==1], bins=(power_bins, density_bins))
    H = H.T
    X, Y = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0])/2, yedges[:-1] + (yedges[1] - yedges[0])/2)

    CS = ax.contour(X, Y, H/np.max(H), levels=[0.05, .2, .5])

    #ax.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.plot(mean_p, mean_d, 'k.', alpha=.8)
    plt.show()

    density_th = np.percentily(density_v[valid_v == 1], 5)
    dB_th = 2*most_common_valid_power - pct99_power

    # ToDo: Here I need flexible parameters...
    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        f = fund_v[ident_v == id]
        i = idx_v[ident_v == id]

        density = len(i) / (i[-1] - i[0] + 1)
        # mean_power = np.mean(decibel(np.max(sign_v[ident_v == id], axis=1)))
        mean_power = np.mean(np.max(sign_v[ident_v == id], axis=1))

        # if density < 0.1 or mean_power <= -85:
        if density < density_th or mean_power <= dB_th:
            valid_v[ident_v == id] = 0

    ident_v = connect_with_overlap(fund_v, ident_v, valid_v, idx_v, times)
    # pass
        # # valid_ids = np.array(valid_ids)
        # d_med_f = np.abs(valid_ids[:, 1] - valid_ids[:, 1][:, np.newaxis])
        # # d_med_f[d_med_f == 0] = 100
        #
        # idx0s, idx1s = np.unravel_index(np.argsort(d_med_f, axis=None), np.shape(d_med_f))
        # similar_mask = idx0s == idx1s
        # idx0s = idx0s[~similar_mask]
        # idx1s = idx1s[~similar_mask]
        #
        # for enu, (idx0, idx1) in enumerate(zip(idx0s, idx1s)):
        #     if np.abs(valid_ids[idx0, 1] - valid_ids[idx1, 1]) > f_th:
        #         break
        #     # print(f'{enu:.0f}: ids = {valid_ids[idx0, 0]:.0f} {valid_ids[idx1, 0]:.0f}; df = {np.abs(valid_ids[idx0, 1] - valid_ids[idx1, 1]):.1f}Hz')
        #     id0 = valid_ids[idx0, 0]
        #     id1 = valid_ids[idx1, 0]
        #     taken_idxs0 = idx_v[(ident_v == id0)]
        #     taken_idxs1 = idx_v[(ident_v == id1)]
        #     double_idx = np.in1d(taken_idxs0, taken_idxs1)
        #     combine = True
        #
        #     # if len(taken_idxs0)*0.01 > len(taken_idxs1):
        #     #     combine = True
        #     # elif len(taken_idxs1)*0.01 > len(taken_idxs0):
        #     #     combine = True
        #     # elif np.sum(double_idx) < len(taken_idxs1)*0.01 and np.sum(double_idx) < len(taken_idxs0)*0.01:
        #     #     combine = True
        #     # else:
        #     #     pass
        #
        #     if double_idx.any():
        #         combine = False
        #
        #     if not combine:
        #         continue
        #
        #     if valid_ids[idx0, 1] < valid_ids[idx1, 1]:
        #         # valid_ids[idx1, 0] = valid_ids[idx0, 0]
        #         valid_ids[:, 0][valid_ids[:, 0] == id1] = id0
        #         ident_v[ident_v == id1] = id0
        #     else:
        #         # valid_ids[idx0, 0] = valid_ids[idx1, 0]
        #         valid_ids[:, 0][valid_ids[:, 0] == id0] = id1
        #         ident_v[ident_v == id0] = id1
        #
        # previous_valid_ids = np.unique(valid_ids[:, 0])
        ####################################################
        # fig = plt.figure(figsize=(30/2.54, 18/2.54))
        # gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, width_ratios=[3, 1], hspace=0)
        # ax = []
        # ax.append(fig.add_subplot(gs[0, 0]))
        # ax.append(fig.add_subplot(gs[0, 1], sharey = ax[0]))
        #
        # ax[1].plot(kde, convolve_f)
        # ax[1].plot([kde_th, kde_th], [convolve_f[0], convolve_f[-1]], '--', color='k', lw=2)
        #
        # for id in tqdm(np.unique(ident_v[~np.isnan(ident_v)])):
        #     f = fund_v[(ident_v == id) & (window_t_mask)]
        #     if len(f) <= 1:
        #         continue
        #     t = times[idx_v[(ident_v == id) & (window_t_mask)]]
        #     if np.min(np.abs(valid_f - f[:, np.newaxis])) <= (convolve_f[1]-convolve_f[0]) / 2:
        #         c = np.random.rand(3)
        #         a = 1
        #     else:
        #         c = 'grey'
        #         a = .2
        #
        #     ax[0].plot(t, f, color=c, alpha = a, marker='.')
        #
        # ax[0].set_ylim(np.min(ff)-10, np.max(ff)+10)
        # plt.show()

    # ToDo: check density

    fig = plt.figure(figsize=(30/2.54, 18/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))

    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        f = fund_v[ident_v == id]
        i = idx_v[ident_v == id]

        ax[0].text(times[i[0]], f[0], f'{id:.0f}', ha='center', va='bottom')

        density = len(i) / (i[-1] - i[0] + 1)
        mean_power = np.mean(decibel(np.max(sign_v[ident_v == id], axis=1)))
        print(f'{id}; density: {density:.3f}; power: {mean_power:.2f}dB')

        if density >= 0.1 and mean_power > -85:
            ax[0].plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], marker='.')
        else:
            ax[0].plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], color='grey')


    fig = plt.figure(figsize=(30/2.54, 18/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))

    # for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v))])):
    for id in tqdm(np.unique(loaded_ident_v[(~np.isnan(loaded_ident_v))])):
        ax[0].plot(times[idx_v[loaded_ident_v == id]], fund_v[loaded_ident_v == id], marker='.')
    plt.show()



    embed()
    quit()




    pass

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()