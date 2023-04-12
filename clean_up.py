import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

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
    folder = "/home/raab/data/breeding_grizzly_mount+/derived_data/2023-03-02-09_54"
    fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
    idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
    ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))

    fine_time = np.load(os.path.join(folder, 'fine_times.npy'))
    fine_freq = np.load(os.path.join(folder, 'fine_freqs.npy'))
    fine_spec_shape = np.load(os.path.join(folder, 'fine_spec_shape.npy'))
    fine_spec = np.memmap(os.path.join(folder, 'fine_spec.npy'), dtype='float', mode='r',
                               shape=(fine_spec_shape[0], fine_spec_shape[1]), order='F')
    for i0 in np.arange(3*60*60, times[-1], 10*60):
        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, width_ratios=[3, 1], hspace=0)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[0, 1], sharey = ax[0]))
        mask = (times[idx_v] >= i0) & (times[idx_v] < i0 + 10 * 60)
        ff = fund_v[(~np.isnan(ident_v)) & (mask)]
        tt = times[idx_v[(~np.isnan(ident_v)) & (mask)]]

        # bins = np.arange(np.floor(np.min(ff)), np.ceil(np.max(ff))+1)
        # n, bins = np.histogram(ff, bins=bins)
        # ax[1].barh(bins[:-1]+(bins[1]-bins[0])/2, n, align="center")

        convolve_f = np.arange(400, 1200, 0.1)
        g = gauss(convolve_f, ff, sigma=5, size=1, norm=True)
        th = np.max(g) * len(tt) * 0.01
        kde = np.sum(g, axis=0)
        # mean_kde = np.mean(kde)
        # std_kde = np.std(kde)
        kde_p5 = np.percentile(kde, 10)
        # embed()
        # quit()
        ax[1].plot(kde, convolve_f)
        # ax[1].plot([mean_kde + std_kde, mean_kde + std_kde], [convolve_f[0], convolve_f[-1]], '--', color='k', lw=2)
        ax[1].plot([th, th], [convolve_f[0], convolve_f[-1]], '--', color='k', lw=2)

        for id in np.unique(ident_v[~np.isnan(ident_v)]):
            f = fund_v[(ident_v == id) & (mask)]
            t = times[idx_v[(ident_v==id) & (mask)]]
            ax[0].plot(t, f)

        plt.show()
    pass

if __name__ == '__main__':
    main()