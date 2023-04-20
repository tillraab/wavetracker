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


def get_valid_ids_by_freq_dist(times, idx_v, ident_v, fund_v, valid_v, old_valid_ids, i0, stride, f_th, kde_th):
    plot=True if not kde_th else False
    # plot=True

    window_t_mask = (times[idx_v] >= i0) & (times[idx_v] < i0 + stride)

    ff = fund_v[(~np.isnan(ident_v)) & (window_t_mask)]

    # ToDo: min_freq & max_freq + buffer (.cfg)
    convolve_f = np.arange(400, 1200, 0.1)
    g = gauss(convolve_f, ff, sigma=2*f_th, size=1, norm=True)
    kde = np.sum(g, axis=0)

    if not kde_th:
        kde_th = np.max(g) * len(times[(times >= times[0]) & (times < times[0] + stride)]) * 0.05

    ################### illustation ###################
    if plot:
        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, width_ratios=[3, 1], hspace=0)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[0, 1], sharey = ax[0]))
        ax[1].plot(kde, convolve_f)
        ax[1].plot([kde_th, kde_th], [convolve_f[0], convolve_f[-1]], '--', color='k', lw=2)
        ax[0].set_ylim(np.min(ff)-10, np.max(ff)+10)
    ###################################################

    valid_f = convolve_f[kde > kde_th]
    valid_ids = []

    for id in np.unique(ident_v[(~np.isnan(ident_v)) & (window_t_mask)]):
        f = fund_v[(ident_v == id) & (window_t_mask)]
        if len(f) <= 1:
            continue

        valid = False
        if np.min(np.abs(valid_f - f[:, np.newaxis])) <= (convolve_f[1] - convolve_f[0]) / 2:
            valid = True
        elif id in old_valid_ids:
            valid = True
        else:
            pass

        if valid:
            valid_ids.append([id, np.median(f), times[idx_v[(ident_v == id)]][0]])
            valid_v[ident_v == id] = 1

        if plot:
            c = np.random.rand(3) if valid else 'grey'
            a = 1 if valid else 0.2
            t = times[idx_v[(ident_v == id) & (window_t_mask)]]
            ax[0].plot(t, f, color=c, alpha = a, marker='.')
            # if valid:
            #     ax[0].text(t[0], f[0], f'{id:.0f}', ha='center', va='bottom')
    # plt.show()
    return kde_th, np.array(valid_ids)


def connect_by_similarity(times, idx_v, ident_v, fund_v, sign_v, valid_v, valid_ids, f_th, i0, stride):

    window_t_mask = (times[idx_v] >= i0) & (times[idx_v] < i0 + stride)

    d_med_f = np.abs(valid_ids[:, 1] - valid_ids[:, 1][:, np.newaxis])

    # embed()
    # quit()

    med_power_id = np.zeros(len(valid_ids))
    for enu, id in enumerate(valid_ids[:, 0]):
        med_power_id[enu] = np.median(np.max(sign_v[(ident_v == id) & (window_t_mask)], axis=1))

    # d_med_power_id = np.abs(med_power_id - med_power_id[:, np.newaxis])

    idx0s, idx1s = np.unravel_index(np.argsort(d_med_f, axis=None), np.shape(d_med_f))
    similar_mask = idx0s == idx1s
    idx0s = idx0s[~similar_mask]
    idx1s = idx1s[~similar_mask]
    # entities = np.sum(d_med_f[idx0s, idx1s] < f_th)

    # idxs = np.sort(np.array([idx0s[:entities], idx1s[:entities]]), axis=0)
    # idx0s, idx1s = idxs

    # power_sorter = np.argsort(np.max(med_power_id[idxs], axis=0))

    # idx0s = idx0s[power_sorter]
    # idx1s = idx1s[power_sorter]


    for enu, (idx0, idx1) in enumerate(zip(idx0s, idx1s)):
        if np.abs(valid_ids[idx0, 1] - valid_ids[idx1, 1]) > f_th:
            break
        id0 = valid_ids[idx0, 0]
        id1 = valid_ids[idx1, 0]

        taken_idxs0 = idx_v[(ident_v == id0)]
        taken_idxs1 = idx_v[(ident_v == id1)]

        more_id = id0 if len(taken_idxs0) > len(taken_idxs1) else id1
        less_id = id1 if more_id == id0 else id0

        double_idx = np.intersect1d(taken_idxs0, taken_idxs1)

        if len(double_idx) > len(taken_idxs1)*0.01 and len(double_idx) > len(taken_idxs0)*0.01:
            continue

        fund_v_m = fund_v[(ident_v == more_id) & (window_t_mask)]
        fund_v_l = fund_v[(ident_v == less_id) & (window_t_mask)]

        idx_v_m = idx_v[(ident_v == more_id) & (window_t_mask)]
        idx_v_l = idx_v[(ident_v == less_id) & (window_t_mask)]

        d_idx_v = np.intersect1d(idx_v_m, idx_v_l)

        join_idx_v = np.concatenate(( idx_v_m, idx_v_l[~np.in1d(idx_v_l, d_idx_v)]))
        help_v = np.ones_like(join_idx_v)
        help_v[:len(idx_v_m)] = 0
        join_fund_v = np.concatenate(( fund_v_m, fund_v_l[~np.in1d(idx_v_l, d_idx_v)]))

        sorter = np.argsort(join_idx_v)
        help_v = help_v[sorter]
        join_fund_v = join_fund_v[sorter]
        join_idx_v = join_idx_v[sorter]

        if np.any(np.abs(np.diff(join_fund_v)[np.diff(help_v != 0)]) > f_th):
            continue

        ident_v[(np.in1d(idx_v, np.array(double_idx))) & (ident_v == less_id)] = np.nan
        valid_v[(np.in1d(idx_v, np.array(double_idx))) & (ident_v == less_id)] = 0

        if valid_ids[idx0, 2] < valid_ids[idx1, 2]:
            valid_ids[:, 0][valid_ids[:, 0] == id1] = id0
            ident_v[ident_v == id1] = id0
        else:
            valid_ids[:, 0][valid_ids[:, 0] == id0] = id1
            ident_v[ident_v == id0] = id1

    previous_valid_ids = np.unique(valid_ids[:, 0])

    return previous_valid_ids, ident_v


def connect_with_overlap(fund_v, ident_v, valid_v, idx_v, times):
    time_tol = 5*60
    freq_tol = 2.5

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

        if np.sum(double_idx) > len(taken_idxs1)*0.01 and np.sum(double_idx) > len(taken_idxs0)*0.01:
            continue

        i0 = idx_v[(ident_v == id0) & (times[idx_v] > overlap_t[0]) & (times[idx_v] < overlap_t[1])]
        i1 = idx_v[(ident_v == id1) & (times[idx_v] > overlap_t[0]) & (times[idx_v] < overlap_t[1])]

        freq_dists = np.abs(f0 - f1[:, np.newaxis])
        time_dist = times[np.abs(i0 - i1[:, np.newaxis])]
        valid_freq_dists = freq_dists[time_dist <= time_tol]
        # mean_freq_dist = np.sum(freq_dists) / np.prod(freq_dists.shape)
        mean_freq_dist = np.mean(valid_freq_dists)

        # min_dists = np.concatenate(( np.min(freq_dists, axis=0), np.min(freq_dists, axis=1) ))

        # if np.all(min_dists) < freq_tol or mean_freq_dist < freq_tol:
        #     connections_candidates.append([id0, id1, mean_freq_dist])
        connections_candidates.append([id0, id1, mean_freq_dist])

    connections_candidates = np.array(connections_candidates)

    for pair_no in np.argsort(connections_candidates[:, 2]):
        id0 = connections_candidates[pair_no, 0]
        id1 = connections_candidates[pair_no, 1]

        taken_idxs0 = idx_v[(ident_v == id0)]
        taken_idxs1 = idx_v[(ident_v == id1)]

        more_id = id0 if len(taken_idxs0) > len(taken_idxs1) else id1
        less_id = id1 if more_id == id0 else id0

        double_idx = np.intersect1d(taken_idxs0, taken_idxs1)

        #ToDo:check for zig-zag

        if len(double_idx) < len(taken_idxs1)*0.01 or len(double_idx) < len(taken_idxs0)*0.01:
            ident_v[(np.in1d(idx_v, np.array(double_idx))) & (ident_v == less_id)] = np.nan
            ident_v[ident_v == less_id] = more_id

            connections_candidates[:, 0][connections_candidates[:, 0] == less_id] = more_id
            connections_candidates[:, 1][connections_candidates[:, 1] == less_id] = more_id

    return ident_v

def power_density_filter(valid_v, sign_v, ident_v, idx_v, fund_v, times):
    ##############################################################
    dps = 1 / (times[1] - times[0])

    valid_power = np.max(sign_v[valid_v == 1], axis=1)
    density_v = np.zeros_like(valid_v)

    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        i = idx_v[ident_v == id]
        id_densities = 1 / np.diff(i)
        id_densities = np.concatenate((id_densities, np.array([np.median(id_densities)])))
        #print(id, len(i), np.max(id_densities), np.min(id_densities))
        density_v[ident_v == id] = id_densities

    if True:
        fig, ax = plt.subplots()
        ax.plot(valid_power, density_v[valid_v == 1], '.')

    power_n, power_bins = np.histogram(valid_power, bins=500)
    most_common_valid_power = power_bins[np.argmax(power_n)]
    pct99_power = np.percentile(valid_power, 99.5)

    if True:
        fig = plt.figure(figsize=(20 / 2.54, 20 / 2.54))
        gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0, wspace=0,
                               height_ratios=[1, 3], width_ratios=[3, 1])

        ax = fig.add_subplot(gs[1, 0])
        ax_t = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_t.xaxis.set_visible(False)
        ax_r = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_r.yaxis.set_visible(False)
        ax.set_xlabel('power [db]', fontsize=12)
        ax.set_ylabel('density', fontsize=12)

        ax_t.bar(power_bins[:-1] + (power_bins[1] - power_bins[0])/2, power_n, align='center', width=(power_bins[1] - power_bins[0])*0.9)
        percentiles = np.percentile(valid_power, (5, 10, 25, 50, 75, 90, 95))
        for p in percentiles:
            ax_t.plot([p, p], [0, np.max(power_n)], lw=2, color='k')

        ax_t.plot([most_common_valid_power, most_common_valid_power], [0, np.max(power_n)], '--', color='red')
        ax_t.plot([pct99_power, pct99_power], [0, np.max(power_n)], '--', color='red')
        ax_t.plot([most_common_valid_power - (pct99_power-most_common_valid_power), most_common_valid_power - (pct99_power-most_common_valid_power)], [0, np.max(power_n)], '--', color='red')


        density_n, density_bins = np.histogram(density_v[valid_v == 1], bins=20)

        ax_r.barh(density_bins[:-1] + (density_bins[1] - density_bins[0])/2, density_n, align='center', height=(density_bins[1] - density_bins[0])*0.9)
        percentiles = np.percentile(density_v[valid_v ==1], (5, 10, 25, 50, 75, 90, 95))
        for p in percentiles:
            ax_r.plot([0, np.max(density_n)], [p, p], lw=2, color='k')

        H, xedges, yedges = np.histogram2d(valid_power, density_v[valid_v ==1], bins=(power_bins, density_bins))
        H = H.T
        X, Y = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0])/2, yedges[:-1] + (yedges[1] - yedges[0])/2)
        CS = ax.contour(X, Y, H/np.max(H), levels=[0.05, .2, .5])

        mean_d = []
        mean_p = []
        for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
            i = idx_v[ident_v == id]
            d = len(i) / (i[-1] - i[0] + 1)
            p = np.mean(np.max(sign_v[ident_v == id], axis=1))
            mean_d.append(d)
            mean_p.append(p)
        ax.plot(mean_p, mean_d, 'k.', alpha=.8)

    density_th = 0.1
    dB_th = 2*most_common_valid_power - pct99_power
    dB_th = dB_th if dB_th > -100 else -100

    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        i = idx_v[ident_v == id]
        density = len(i) / (i[-1] - i[0] + 1)
        p = np.max(sign_v[(ident_v == id)], axis=1)
        mean_power = np.mean(p)
        if density < density_th or mean_power <= dB_th:
            valid_v[ident_v == id] = 0
            if mean_power <= dB_th and density > density_th and (i[-1] - i[0]) / dps > 10 * 60:
                fig, ax = plt.subplots(2, 1)
                ii = np.arange(i[0], i[-1] + 1)
                pp = np.interp(ii, i, p)

                ppp = np.convolve(pp, np.ones(int(60 * dps)) / int(10 * 60 * dps), mode='same')
                corr_fac = np.convolve(np.ones(len(pp)), np.ones(int(60 * dps)) / int(10 * 60 * dps), mode='same')
                ppp *= 1/corr_fac
                ax[0].plot(i, p)
                ax[0].plot(ii, ppp, lw=2, color='k')
                ax[0].set_title(f'{len(p[p > dB_th])/ len(p):.2f} above th ({dB_th:.2f}Hz)')

                if len(p[p > dB_th])/ len(p) > 0.1:
                    up_idx = ii[:-1][(ppp[:-1] < dB_th) & (ppp[1:] > dB_th)]
                    down_idx = ii[1:][(ppp[:-1] > dB_th) & (ppp[1:] < dB_th)]

                    if len(up_idx) == 0: up_idx = [i[0]]
                    if len(down_idx) == 0: down_idx = [i[-1]]

                    if up_idx[0] > down_idx[0]:
                        up_idx = np.concatenate((np.array([ii[0]-1]), up_idx))
                    if down_idx[-1] < up_idx[-1]:
                        down_idx = np.concatenate((down_idx, np.array([ii[-1]+1])))
                    ax[0].plot(up_idx, np.ones(len(up_idx))*dB_th, 'og', markersize=10)
                    ax[0].plot(down_idx, np.ones(len(down_idx))*dB_th, 'or', markersize=10)
                    next_ident = np.nanmax(ident_v)+1
                    for ui, di in zip(up_idx, down_idx):
                        ident_v[(ident_v == id) & (idx_v >= ui) & (idx_v < di)] = next_ident
                    valid_v[ident_v == next_ident] = 1
                    ax[1].plot(idx_v[ident_v == next_ident], fund_v[ident_v == next_ident], marker='.')

                ax[1].plot(idx_v[ident_v == id], fund_v[ident_v == id], marker='.', color='k')

                plt.show()


    return valid_v

def main(folder = None):
    # load data
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
    if np.nanmin(sign_v) > 0:
        sign_v = decibel(sign_v)
    loaded_ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))

    # parameters
    valid_v = np.zeros_like(ident_v)
    stride = 10 * 60
    overlap = 0.2
    f_th = 2.5
    kde_th = None
    previous_valid_ids = np.array([])

    # embed()
    # quit()

    for i0 in tqdm(np.arange(0, times[-1], int(stride*(1-overlap))), desc='striding window analysis'):

        kde_th, valid_ids = get_valid_ids_by_freq_dist(times, idx_v, ident_v, fund_v, valid_v, previous_valid_ids, i0,
                                                       stride, f_th, kde_th)

        previous_valid_ids, ident_v = connect_by_similarity(times, idx_v, ident_v, fund_v, sign_v, valid_v, valid_ids,
                                                            f_th, i0, stride)

    ################### illustation ###################
    if True:
        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))

        for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
            f = fund_v[ident_v == id]
            i = idx_v[ident_v == id]
            ax[0].text(times[i[0]], f[0], f'{id:.0f}', ha='center', va='bottom')
            ax[0].plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], marker='.')
        ax[0].set_title('after snippet connections')
        #plt.show()
    ###################################################

    valid_v = power_density_filter(valid_v, sign_v, ident_v, idx_v, fund_v, times)

    ################### illustation ###################
    if True:
        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))

        for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
            f = fund_v[ident_v == id]
            i = idx_v[ident_v == id]
            ax[0].text(times[i[0]], f[0], f'{id:.0f}', ha='center', va='bottom')
            ax[0].plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], marker='.')
        ax[0].set_title('after power_density filter')
        #plt.show()
    ###################################################

    ident_v = connect_with_overlap(fund_v, ident_v, valid_v, idx_v, times)

    ################### illustation ###################
    if True:
        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))

        for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
            f = fund_v[ident_v == id]
            i = idx_v[ident_v == id]

            ax[0].text(times[i[0]], f[0], f'{id:.0f}', ha='center', va='bottom')

            density = len(i) / (i[-1] - i[0] + 1)
            mean_power = np.mean(np.max(sign_v[ident_v == id], axis=1))
            print(f'{id}; density: {density:.3f}; power: {mean_power:.2f}dB')

            ax[0].plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], marker='.')

        ax[0].set_title('after global connect')
        # plt.show()

        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))

        # for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v))])):
        for id in np.unique(loaded_ident_v[(~np.isnan(loaded_ident_v))]):
            ax[0].plot(times[idx_v[loaded_ident_v == id]], fund_v[loaded_ident_v == id], marker='.')
        ax[0].set_title('original')
        plt.show()
    ###################################################

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()