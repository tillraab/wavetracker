import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thunderfish.powerspectrum import decibel
from IPython import embed
from tqdm import tqdm

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


def main():
    folder = "/home/raab/data/cleanup_test/2023-03-02-09_54"
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
    th = None
    old_valid_ids = np.array([])

 #   for i0 in tqdm(np.arange(3*60*60, times[-1], int(stride*(1-overlap)))):
    for i0 in tqdm(np.arange(0, times[-1], int(stride*(1-overlap)))):
        # fig = plt.figure(figsize=(30/2.54, 18/2.54))
        # gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, width_ratios=[3, 1], hspace=0)
        # ax = []
        # ax.append(fig.add_subplot(gs[0, 0]))
        # ax.append(fig.add_subplot(gs[0, 1], sharey = ax[0]))

        window_t_mask = (times[idx_v] >= i0) & (times[idx_v] < i0 + stride)

        ff = fund_v[(~np.isnan(ident_v)) & (window_t_mask)]
        #tt = times[idx_v[(~np.isnan(ident_v)) & (window_t_mask)]]

        convolve_f = np.arange(400, 1200, 0.1)
        g = gauss(convolve_f, ff, sigma=f_th, size=1, norm=True)
        kde = np.sum(g, axis=0)

        if not th:
            th = np.max(g) * len(times[(times >= times[0]) & (times < times[0] + stride)]) * 0.05

        # ax[1].plot(kde, convolve_f)
        # ax[1].plot([th, th], [convolve_f[0], convolve_f[-1]], '--', color='k', lw=2)

        valid_f = convolve_f[kde > th]

        valid_ids = []
        for id in tqdm(np.unique(ident_v[~np.isnan(ident_v)])):
            f = fund_v[(ident_v == id) & (window_t_mask)]
            if len(f) <= 1:
                continue
            t = times[idx_v[(ident_v == id) & (window_t_mask)]]
            valid = False
            if np.min(np.abs(valid_f - f[:, np.newaxis])) <= (convolve_f[1]-convolve_f[0]) / 2:
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
        ###########################

        valid_ids = np.array(valid_ids)
        d_med_f = np.abs(valid_ids[:, 1] - valid_ids[:, 1][:, np.newaxis])
        # d_med_f[d_med_f == 0] = 100

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

        old_valid_ids = np.unique(valid_ids[:, 0])
        ####################################################
        # fig = plt.figure(figsize=(30/2.54, 18/2.54))
        # gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, width_ratios=[3, 1], hspace=0)
        # ax = []
        # ax.append(fig.add_subplot(gs[0, 0]))
        # ax.append(fig.add_subplot(gs[0, 1], sharey = ax[0]))
        #
        # ax[1].plot(kde, convolve_f)
        # ax[1].plot([th, th], [convolve_f[0], convolve_f[-1]], '--', color='k', lw=2)
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

    # for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v))])):
    for id in tqdm(np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])):
        f = fund_v[ident_v == id]
        i = idx_v[ident_v == id]

        # ax[0].text(times[i[0]], f[0], f'{density:.2f}; {np.percentile(f, 95) - np.percentile(f, 5):.2f}Hz', ha='center', va='bottom')
        # ax[0].plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], marker='.')
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
    main()