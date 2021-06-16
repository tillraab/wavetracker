import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from plottools.colors import *
colors_params(colors_muted, colors_tableau)
import os
from IPython import embed
from tqdm import tqdm
from PyQt5.QtCore import *

from thunderfish.powerspectrum import decibel

class Emit_progress():
    progress = pyqtSignal(float)

class Display_agorithm():
    def __init__(self, fund_v, ident_v, idx_v, sign_v, times, a_error_distribution, error_dist_i0s, error_dist_i1s):
        self.fund_v = fund_v
        self.sign_v = sign_v
        self.ident_v = ident_v
        self.tmp_ident_v = None
        self.idx_v = idx_v
        self.times = times
        # self.spec = np.load("/home/raab/thesis/code/tracking_display/spec.npy")
        self.spec = np.load("/home/raab/writing/2021_tracking/data/2016-04-10-11_12/spec.npy")

        self.a_error_dist = a_error_distribution
        self.error_dist_i0s = error_dist_i0s
        self.error_dist_i1s = error_dist_i1s

        self.tmp_ident_v_state = []
        self.handles = {}
        self.itter_counter = 0

        self.tmp_trace_handels = {}
        self.trace_handels = {}
        self.origin_idx = None
        self.target_idx = None
        self.alt_idx = None

    def plot_a_error_dist(self):
        from plottools.tag import tag

        X, Y = np.meshgrid(np.arange(8), np.arange(8))

        ####
        fig = plt.figure(figsize=(17.5 / 2.54, 10 / 2.54))
        gs = gridspec.GridSpec(3, 4, left=0.1, bottom=0.15, right=0.95, top=0.95, hspace=0.4, wspace=0.4, height_ratios=[2, 2, 1.5])
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0]))

        ax.append(fig.add_subplot(gs[0, 1]))
        ax.append(fig.add_subplot(gs[1, 1]))

        ax.append(fig.add_subplot(gs[0, 2]))
        ax.append(fig.add_subplot(gs[1, 2]))

        ax.append(fig.add_subplot(gs[0, 3]))
        ax.append(fig.add_subplot(gs[1, 3]))

        # gs2 = gridspec.GridSpec(1, 1, left=0.25, bottom=0.1, right=0.85, top=0.3, hspace=0.3, wspace=0.3)
        ax.append(fig.add_subplot(gs[2, :]))
        ####

        mask = np.argsort(self.a_error_dist)
        ex = np.array(np.floor(np.linspace(10, len(mask)-1, 4)), dtype=int)
        ex[-1] -= 20
        ex_color = ['forestgreen', 'gold', 'darkorange', 'firebrick']

        ex_i0 = self.error_dist_i0s[mask[ex]]
        ex_i1 = self.error_dist_i1s[mask[ex]]

        for enu, i0, i1 in zip(np.arange(len(ex_i0)), ex_i0, ex_i1):
            s0 = self.sign_v[i0].reshape(8, 8)
            #, aspect='auto'
            ax[enu*2].imshow(s0[::-1], alpha=0.7, cmap='jet', vmax=1, vmin=0, interpolation='gaussian', zorder=1)
            s1 = self.sign_v[i1].reshape(8, 8)
            ax[enu*2+1].imshow(s1[::-1], alpha=0.7, cmap='jet', vmax=1, vmin=0, interpolation='gaussian',
                           zorder=1)
            for x, y in zip(X, Y):
                ax[enu*2].plot(x, y, '.', color='k', markersize=2)
                ax[enu*2+1].plot(x, y, '.', color='k', markersize=2)
            y0, y1 = ax[enu * 2].get_ylim()
            #ax[enu * 2].arrow(8.5, 3.5, 2, 0, head_width=.7, head_length=.7, clip_on=False, color=ex_color[enu], lw=2.5)
            ax[enu * 2].arrow(3.5, 8.25, 0, .8, head_width=.7, head_length=.7, clip_on=False, color=ex_color[enu], lw=2)
            ax[enu * 2].set_ylim(y0, y1)

            ax[enu*2].set_xticks([])
            ax[enu*2+1].set_xticks([])

            ax[enu*2].set_yticks([])
            ax[enu*2+1].set_yticks([])

        ax[-1].plot(self.a_error_dist[mask], np.linspace(0, 1, len(self.a_error_dist)), color='midnightblue', clip_on=False)
        for enu in range(4):
            ax[-1].plot(self.a_error_dist[mask[ex[enu]]], np.linspace(0, 1, len(self.a_error_dist))[ex[enu]], 'o', color=ex_color[enu], clip_on=False, markersize=6)
        ax[-1].set_ylim(0, 1)
        ax[-1].set_yticks([0, 1])

        ax[-1].set_xlim(0, np.max(self.a_error_dist))
        ax[-1].set_ylabel('field error', fontsize=12)
        ax[-1].set_xlabel(r'$\Delta$ field amplitude', fontsize=12)
        ax[-1].tick_params(labelsize=10)

        # fig.tag(axes=[ax[0], ax[8]], labels=['A', 'E'], fontsize=15, yoffs=2, xoffs=-6)
        # fig.tag(axes=[ax[2], ax[4], ax[6]], labels=['B', 'C', 'D'], fontsize=15, yoffs=2, xoffs=-3)

        plt.savefig('amplitude_error_dist.pdf')
        plt.close()

    def plot_assign(self, origin_idx, tartget_idx0, alt_target_idx):
        test_alt_idx = alt_target_idx[0] if alt_target_idx[0] != tartget_idx0 else alt_target_idx[1]

        if np.abs(self.fund_v[origin_idx] - self.fund_v[tartget_idx0]) >= np.abs(self.fund_v[origin_idx] - self.fund_v[test_alt_idx]):

            fig = plt.figure(figsize=(20/2.54, 12/2.54))
            gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.15, right=0.75, top=0.75)
            ax = fig.add_subplot(gs[0, 0])

            ax.imshow(decibel(self.spec)[::-1], extent=[self.times[0], self.times[-1], 0, 2000],
                      aspect='auto', alpha=0.7, cmap='jet', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)

            ax.plot(self.times[self.idx_v], self.fund_v, '.', color='grey', markersize=3)
            ax.plot(self.times[self.idx_v[origin_idx]], self.fund_v[origin_idx], '.', color='k', markersize=10)
            ax.plot(self.times[self.idx_v[tartget_idx0]], self.fund_v[tartget_idx0], '.', color='green', markersize=10)

            for alt_idx in alt_target_idx:
                if alt_idx != tartget_idx0:
                    ax.plot(self.times[self.idx_v[alt_idx]], self.fund_v[alt_idx], '.', color='red', markersize=10)

            help_idx = np.concatenate((np.array([origin_idx, tartget_idx0]), np.array(alt_target_idx)))
            xs = self.times[self.idx_v[help_idx]]
            ys = self.fund_v[help_idx]

            ax.set_xlim(np.min(xs) - 5, np.max(xs) + 5)
            ax.set_ylim(np.min(ys) - 30, np.max(ys) + 30)

            gs2 = gridspec.GridSpec(1, 1, left = 0.6, bottom=0.6, right=0.95, top=0.95)
            ax_ins = fig.add_subplot(gs2[0, 0])

            ax_ins.plot(np.arange(0, 2.5, 0.001), boltzmann(np.arange(0, 2.5, 0.001), alpha=1, beta=0, x0=.35, dx=.08), color='midnightblue')

            ax_ins.set_xlim(-.025, 1)
            ax_ins.set_xticks([0, 0.5, 1])
            ax_ins.set_ylim(-.025, 1)
            ax_ins.set_yticks([0, 1])

            df_target = np.abs(self.fund_v[tartget_idx0] - self.fund_v[origin_idx])
            f_error = boltzmann(df_target, alpha=1, beta=0, x0=.35, dx=.08)

            ax_ins.plot([df_target, df_target], [-.025, f_error], color='green', lw=4)
            ax_ins.plot([-.025, df_target], [f_error, f_error], color='green', lw=4)

            for alt_idx in alt_target_idx:
                if alt_idx != tartget_idx0:
                    df_target = np.abs(self.fund_v[alt_idx] - self.fund_v[origin_idx])
                    f_error = boltzmann(df_target, alpha=1, beta=0, x0=.35, dx=.08)

                    ax_ins.plot([df_target, df_target], [-.025, f_error], color='red', lw=4)
                    ax_ins.plot([-.025, df_target], [f_error, f_error], color='red', lw=4)

            plt.show()

    def static_tmp_id_tracking(self, min_i0, max_i1):
        t0 = self.times[self.idx_v[min_i0]]

        self.combo_fig = plt.figure(figsize=(9.5/2.54, 14/2.54))
        gs = gridspec.GridSpec(3, 1, left=.2, bottom=.1, right=.95, top=.9, height_ratios=[2, 2, 2], hspace=0.3)
        self.combo_ax = []
        # self.combo_ax.append(self.combo_fig.add_subplot(gs[0, 0]))
        self.combo_ax.append(self.combo_fig.add_subplot(gs[0, 0]))
        self.combo_ax.append(self.combo_fig.add_subplot(gs[1, 0], sharex = self.combo_ax[0]))
        self.combo_ax.append(self.combo_fig.add_subplot(gs[2, 0], sharex = self.combo_ax[0]))

        for i in np.arange(3):
            self.combo_ax[i].imshow(decibel(self.spec)[::-1], extent=[self.times[0] - t0, self.times[-1] - t0, 0, 2000],
                                    aspect='auto', alpha=0.7, cmap='Greys', vmax=-50, vmin=-110,
                                    interpolation='gaussian', zorder=1)

            self.combo_ax[i].set_xlim(self.times[self.idx_v[min_i0]] - t0, self.times[self.idx_v[max_i1]] - t0)
            self.combo_ax[i].set_ylim(905, 930)
            # self.combo_ax[i+1].plot([10, 10], [890, 930], '--', lw=1, color='k')
            # self.combo_ax[i+1].plot([20, 20], [890, 930], '--', lw=1, color='k')

            for id in self.tmp_ident_v_state[i][~np.isnan(self.tmp_ident_v_state[i])]:
                # if 880 < self.fund_v[self.idx_v[self.tmp_ident_v_state[i] == id]][0] < 930:
                self.combo_ax[i].plot(self.times[self.idx_v[self.tmp_ident_v_state[i] == id]] - t0,
                                        self.fund_v[self.tmp_ident_v_state[i] == id], marker='.', markersize=4)


        for ax in self.combo_ax[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        self.combo_ax[-1].set_xlabel('time [s]', fontsize=10)
        self.combo_ax[1].set_ylabel('frequency [Hz]', fontsize=10)

        self.combo_ax[0].fill_between([0, 10], [935, 935], [938, 938], color='grey', clip_on=False)
        self.combo_ax[0].fill_between([10, 20], [935, 935], [938, 938], color='k', clip_on=False)
        self.combo_ax[0].fill_between([20, 30], [935, 935], [938, 938], color='grey', clip_on=False)

        for x0 in [0, 10, 20, 30]:
            con = ConnectionPatch(xyA=(x0, 930), xyB=(x0, 905), coordsA="data", coordsB="data",
                                  axesA=self.combo_ax[0], axesB=self.combo_ax[-1], color="k", linestyle='-', zorder=10, lw=1)
            self.combo_ax[-1].add_artist(con)
            self.combo_ax[0].plot([x0, x0], [930, 935], color='k', lw=1, clip_on=False)


        self.combo_ax[0].set_xlim(0, 30)
        for a in self.combo_ax:
            a.tick_params(labelsize=9)

        plt.savefig('./tmp_ident_tracking.png', dpi=300)
        plt.show()
        self.tmp_ident_v_state = []

    def static_tmp_id_assign_init(self):
        self.fig2 = plt.figure(figsize=(17.5/2.54, 12/2.54))
        gs = gridspec.GridSpec(2, 2, left=.2, bottom=.1, right=.95, top=.9, hspace=0.3, wspace=0.3)
        self.ax2 = []
        # self.combo_ax.append(self.combo_fig.add_subplot(gs[0, 0]))
        self.ax2.append(self.fig2.add_subplot(gs[0, 0]))
        self.ax2.append(self.fig2.add_subplot(gs[0, 1]))
        self.ax2.append(self.fig2.add_subplot(gs[1, 0]))
        self.ax2.append(self.fig2.add_subplot(gs[1, 1]))

        for a in self.ax2:
            a.imshow(decibel(self.spec)[::-1], extent=[self.times[0], self.times[-1], 0, 2000], aspect='auto',
                     alpha=0.7, cmap='Greys', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)
            a.set_ylim(905, 930)

    def life_tmp_ident_init(self, min_i0, max_i1):

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(decibel(self.spec)[::-1], extent=[self.times[0], self.times[-1], 0, 2000],
                  aspect='auto', alpha=0.7, cmap='jet', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)

        self.ax.set_xlim(self.times[self.idx_v[min_i0]], self.times[self.idx_v[max_i1]])
        self.ax.set_ylim(880, 950)
        # self.fig.canvas.draw()
        plt.pause(0.05)


    def life_tmp_ident_update(self, tmp_indet_v, new=None, update=None, delete=None):
        if new:
            self.handles[new], = self.ax.plot(self.times[self.idx_v[tmp_indet_v == new]], self.fund_v[tmp_indet_v == new], marker='.')
        if update:
            self.handles[update].set_data(self.times[self.idx_v[tmp_indet_v == update]], self.fund_v[tmp_indet_v == update])
        if delete:
            self.handles[delete].remove()
            del self.handles[delete]

        # self.fig.canvas.draw()
        plt.pause(0.05)



def aux():
    pass
    # ids = np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)])
    # id_comb = []
    # id_comb_freqs = []
    # id_comb_idx = []
    # id_comb_df = []
    # id_comb_part_df = []
    # id_comb_overlap = []
    # for id0 in range(len(ids)):
    #     id0_med_freq = np.median(tmp_fund_v[tmp_ident_v == ids[id0]])
    #
    #     for id1 in range(id0 + 1, len(ids)):
    #         id_comb.append((id0, id1))
    #         id1_med_freq = np.median(tmp_fund_v[tmp_ident_v == ids[id1]])
    #         id_comb_df.append(np.abs(id1_med_freq - id0_med_freq))
    #
    #         # no overlap + 5 sec ?!
    #         if np.max(tmp_idx_v[tmp_ident_v == ids[id0]]) < np.min(tmp_idx_v[tmp_ident_v == ids[id1]]):
    #             idx0_n = int(tmp_idx_v[tmp_ident_v == ids[id0]][-1] + idx_comp_range / 2)
    #             idx0_0 = int(idx0_n - idx_comp_range / 2)
    #
    #             idx1_0 = int(tmp_idx_v[tmp_ident_v == ids[id1]][0] - idx_comp_range / 2)
    #             idx1_n = int(idx1_0 + idx_comp_range / 2)
    #
    #             idx1_0 = idx1_0 if idx1_0 > 0 else 0
    #             idx1_n = idx1_n if idx1_n > 0 else 0
    #             idx0_0 = idx0_0 if idx0_0 > 0 else 0
    #             idx0_n = idx0_n if idx0_n > 0 else 0
    #
    #             id0_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id0]) & (tmp_idx_v >= idx0_0) & (tmp_idx_v <= idx0_n)]
    #             id0_part_freq = tmp_fund_v[id0_part_idx]
    #
    #             id1_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id1]) & (tmp_idx_v >= idx1_0) & (tmp_idx_v <= idx1_n)]
    #             id1_part_freq = tmp_fund_v[id1_part_idx]
    #
    #             id_comb_part_df.append(np.abs(np.median(id0_part_freq) - np.median(id1_part_freq)))
    #             id_comb_freqs.append([id0_part_freq, id1_part_freq])
    #             id_comb_idx.append([id0_part_idx, id1_part_idx])
    #             # ToDo: maybe id_comb_idx
    #
    #             # id0 < id1
    #             id_comb_overlap.append(-1*(np.min(tmp_idx_v[tmp_ident_v == ids[id1]]) - np.max(tmp_idx_v[tmp_ident_v == ids[id0]])))  # ToDo: neg. values for time distance
    #
    #
    #         elif np.max(tmp_idx_v[tmp_ident_v == ids[id1]]) < np.min(tmp_idx_v[tmp_ident_v == ids[id0]]):
    #             idx1_n = int(tmp_idx_v[tmp_ident_v == ids[id1]][-1] + idx_comp_range / 2)
    #             idx1_0 = int(idx1_n - idx_comp_range / 2)
    #
    #             idx0_0 = int(tmp_idx_v[tmp_ident_v == ids[id0]][0] - idx_comp_range / 2)
    #             idx0_n = int(idx0_0 + idx_comp_range / 2)
    #
    #             idx1_0 = idx1_0 if idx1_0 > 0 else 0
    #             idx1_n = idx1_n if idx1_n > 0 else 0
    #             idx0_0 = idx0_0 if idx0_0 > 0 else 0
    #             idx0_n = idx0_n if idx0_n > 0 else 0
    #
    #             id0_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id0]) & (tmp_idx_v >= idx0_0) & (tmp_idx_v <= idx0_n)]
    #             id0_part_freq = tmp_fund_v[id0_part_idx]
    #
    #             id1_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id1]) & (tmp_idx_v >= idx1_0) & (tmp_idx_v <= idx1_n)]
    #             id1_part_freq = tmp_fund_v[id1_part_idx]
    #
    #             id_comb_part_df.append(np.abs(np.median(id0_part_freq) - np.median(id1_part_freq)))
    #             id_comb_freqs.append([id0_part_freq, id1_part_freq])
    #             id_comb_idx.append([id0_part_idx, id1_part_idx])
    #
    #             # id1 < id0
    #             id_comb_overlap.append(-1*(np.min(tmp_idx_v[tmp_ident_v == ids[id0]]) - np.max(tmp_idx_v[tmp_ident_v == ids[id1]])))
    #
    #         # overlap + 5 sec ?!
    #         elif (np.min(tmp_idx_v[tmp_ident_v == ids[id0]]) <= np.min(tmp_idx_v[tmp_ident_v == ids[id1]])) and (
    #                 np.max(tmp_idx_v[tmp_ident_v == ids[id0]]) >= np.min(tmp_idx_v[tmp_ident_v == ids[id1]])):
    #
    #             ioi = [np.min(tmp_idx_v[tmp_ident_v == ids[id0]]), np.max(tmp_idx_v[tmp_ident_v == ids[id0]]),
    #                    np.min(tmp_idx_v[tmp_ident_v == ids[id1]]), np.max(tmp_idx_v[tmp_ident_v == ids[id1]])]
    #             ioi = np.array(ioi)[np.argsort(ioi)]
    #             id_comb_overlap.append(ioi[2] - ioi[1] + 1)
    #
    #             idx0_n = int(ioi[2] + idx_comp_range / 2) if int(ioi[2] + idx_comp_range / 2) > 0 else 0
    #             idx0_0 = int(ioi[1] - idx_comp_range / 2) if int(ioi[1] - idx_comp_range / 2) > 0 else 0
    #
    #             idx1_0 = int(ioi[1] - idx_comp_range / 2) if int(ioi[1] - idx_comp_range / 2) > 0 else 0
    #             idx1_n = int(ioi[2] + idx_comp_range / 2) if int(ioi[2] + idx_comp_range / 2) > 0 else 0
    #
    #             id0_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id0]) & (tmp_idx_v >= idx0_0) & (tmp_idx_v <= idx0_n)]
    #             id0_part_freq = tmp_fund_v[id0_part_idx]
    #
    #             id1_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id1]) & (tmp_idx_v >= idx1_0) & (tmp_idx_v <= idx1_n)]
    #             id1_part_freq = tmp_fund_v[id1_part_idx]
    #
    #             id_comb_part_df.append(np.abs(np.median(id0_part_freq) - np.median(id1_part_freq)))
    #             id_comb_freqs.append([id0_part_freq, id1_part_freq])
    #             id_comb_idx.append([id0_part_idx, id1_part_idx])
    #
    #             # id0 < id1
    #
    #
    #         elif (np.min(tmp_idx_v[tmp_ident_v == ids[id1]]) <= np.min(tmp_idx_v[tmp_ident_v == ids[id0]])) and (
    #                 np.max(tmp_idx_v[tmp_ident_v == ids[id1]]) >= np.min(tmp_idx_v[tmp_ident_v == ids[id0]])):
    #
    #             ioi = [np.min(tmp_idx_v[tmp_ident_v == ids[id0]]), np.max(tmp_idx_v[tmp_ident_v == ids[id0]]),
    #                    np.min(tmp_idx_v[tmp_ident_v == ids[id1]]), np.max(tmp_idx_v[tmp_ident_v == ids[id1]])]
    #             ioi = np.array(ioi)[np.argsort(ioi)]
    #             id_comb_overlap.append(ioi[2] - ioi[1] + 1)
    #
    #             idx1_n = int(ioi[2] + idx_comp_range / 2) if int(ioi[2] + idx_comp_range / 2) > 0 else 0
    #             idx1_0 = int(ioi[1] - idx_comp_range / 2) if int(ioi[1] - idx_comp_range / 2) > 0 else 0
    #
    #             idx0_0 = int(ioi[1] - idx_comp_range / 2) if int(ioi[1] - idx_comp_range / 2) > 0 else 0
    #             idx0_n = int(ioi[2] + idx_comp_range / 2) if int(ioi[2] + idx_comp_range / 2) > 0 else 0
    #
    #
    #             id0_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id0]) & (tmp_idx_v >= idx0_0) & (tmp_idx_v <= idx0_n)]
    #             id0_part_freq = tmp_fund_v[id0_part_idx]
    #
    #             id1_part_idx = np.arange(len(tmp_ident_v))[(tmp_ident_v == ids[id1]) & (tmp_idx_v >= idx1_0) & (tmp_idx_v <= idx1_n)]
    #             id1_part_freq = tmp_fund_v[id1_part_idx]
    #
    #             id_comb_part_df.append(np.abs(np.median(id0_part_freq) - np.median(id1_part_freq)))
    #             id_comb_freqs.append([id0_part_freq, id1_part_freq])
    #             id_comb_idx.append([id0_part_idx, id1_part_idx])
    #
    #             # id1 < id0
    #
    #         else:
    #             print('found a non existing cases')
    #             embed()
    #             quit()

    # embed()
    # quit()
    # id_comb_part_df = np.array(id_comb_part_df)
    # sorting_mask = np.argsort(id_comb_part_df)[:len(id_comb_part_df[id_comb_part_df <= 25])]
    #
    # for i, (id0, id1) in enumerate(np.array(id_comb)[sorting_mask]):
    #     comb_f = np.concatenate(id_comb_freqs[sorting_mask[i]])
    #
    #     bins = np.arange((np.min(comb_f) // .1) * .1, (np.max(comb_f) // .1) * .1 + .1, .1)
    #     bc = bins[:-1] + (bins[1:] - bins[:-1]) / 2
    #
    #     n0, bins = np.histogram(id_comb_freqs[sorting_mask[i]][0], bins=bins)
    #
    #     n1, bins = np.histogram(id_comb_freqs[sorting_mask[i]][1], bins=bins)
    #
    #     greater_mask = n0 >= n1
    #
    #     overlapping_counts = np.sum(np.concatenate((n1[greater_mask], n0[~greater_mask])))
    #
    #     pct_overlap = np.max([overlapping_counts / np.sum(n1), overlapping_counts / np.sum(n0)])
    #
    #     if pct_overlap >= 0:
    #
    #         fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
    #         # embed()
    #         # quit()
    #         for j in range(len(ids)):
    #             if ids[j] == ids[id0]:
    #                 ax[0].plot(tmp_idx_v[tmp_ident_v == ids[j]], tmp_fund_v[tmp_ident_v == ids[j]], marker='.', markersize=4,
    #                            color='red', alpha = 0.5)
    #                 ax[0].plot(tmp_idx_v[id_comb_idx[sorting_mask[i]][0]], id_comb_freqs[sorting_mask[i]][0], marker='o', markersize=4,
    #                            color='red')
    #
    #             elif ids[j] == ids[id1]:
    #                 ax[0].plot(tmp_idx_v[tmp_ident_v == ids[j]], tmp_fund_v[tmp_ident_v == ids[j]], marker='.', markersize=4,
    #                            color='blue', alpha = 0.5)
    #                 ax[0].plot(tmp_idx_v[id_comb_idx[sorting_mask[i]][1]], id_comb_freqs[sorting_mask[i]][1], marker='o',
    #                            markersize=4,
    #                            color='blue')
    #             else:
    #                 ax[0].plot(tmp_idx_v[tmp_ident_v == ids[j]], tmp_fund_v[tmp_ident_v == ids[j]], marker='.', markersize=4,
    #                            color='grey')
    #
    #         ax[1].set_title('%.2f' % pct_overlap)
    #         ax[1].bar(bc, n0, color='red', alpha=.5, width=.08)
    #         ax[1].bar(bc, n1, color='blue', alpha=.5, width=.08)
    #         ax[0].set_ylim(np.mean(ax[1].get_xlim()) - 5, np.mean(ax[1].get_xlim()) + 5)
    #         plt.show()
    # plt.show(block=False)
    # plt.waitforbuttonpress()
    # plt.close(fig)

    # if id_comb_overlap[sorting_mask[i]] > 0:
    #     embed()
    #     quit()
    # len_id0 = len(tmp_ident_v[tmp_ident_v == ids[id0]])
    # len_id1 = len(tmp_ident_v[tmp_ident_v == ids[id1]])
    #
    # overlapping_idx = list(set(tmp_idx_v[tmp_ident_v == ids[id0]]) & set(tmp_idx_v[tmp_ident_v == ids[id1]]))

    #### this is new and in progress --- end ####

def freq_tracking_v5(fundamentals, signatures, times, freq_tolerance= 2.5, n_channels=64, max_dt=10., ioi_fti=False,
                     freq_lims=(400, 1200), emit = False, visualize=False):
    """
    Sorting algorithm which sorts fundamental EOD frequnecies detected in consecutive powespectra of single or
    multielectrode recordings using frequency difference and frequnency-power amplitude difference on the electodes.

    Signal tracking and identity assiginment is accomplished in four steps:
    1) Extracting possible frequency and amplitude difference distributions.
    2) Esitmate relative error between possible datapoint connections (relative amplitude and frequency error based on
    frequency and amplitude error distribution).
    3) For a data window covering the EOD frequencies detected 10 seconds before the accual datapoint to assigne
    identify temporal identities based on overall error between two datapoints from smalles to largest.
    4) Form tight connections between datapoints where one datapoint is in the timestep that is currently of interest.

    Repeat these steps until the end of the recording.
    The temporal identities are only updated when the timestep of current interest reaches the middle (5 sec.) of the
    temporal identities. This is because no tight connection shall be made without checking the temporal identities.
    The temnporal identities are used to check if the potential connection from the timestep of interest to a certain
    datsapoint is the possibly best or if a connection in the futur will be better. If a future connection is better
    the thight connection is not made.

    Parameters
    ----------
    fundamentals: 2d-arraylike / list
        list of arrays of fundemantal EOD frequnecies. For each timestep/powerspectrum contains one list with the
        respectivly detected fundamental EOD frequnecies.
    signatures: 3d-arraylike / list
        same as fundamentals but for each value in fundamentals contains a list of powers of the respective frequency
        detected of n electrodes used.
    times: array
        respective time vector.
    freq_tolerance: float
        maximum frequency difference between two datapoints to be connected in Hz.
    n_channels: int
        number of channels/electodes used in the analysis.,
    return_tmp_idenities: bool
        only returne temporal identities at a certain timestep. Dependent on ioi_fti and only used to check algorithm.
    ioi_fti: int
        Index Of Interest For Temporal Identities: respective index in fund_v to calculate the temporal identities for.
    a_error_distribution: array
        possible amplitude error distributions for the dataset.
    f_error_distribution: array
        possible frequency error distribution for the dataset.
    fig: mpl.figure
        figure to plot the tracking progress life.
    ax: mpl.axis
        axis to plot the tracking progress life.
    freq_lims: double
        minimum/maximum frequency to be tracked.

    Returns
    -------
    fund_v: array
        flattened fundamtantals array containing all detected EOD frequencies in the recording.
    ident_v: array
        respective assigned identites throughout the tracking progress.
    idx_v: array
        respective index vectro impliing the time of the detected frequency.
    sign_v: 2d-array
        for each fundamental frequency the power of this frequency on the used electodes.
    a_error_distribution: array
        possible amplitude error distributions for the dataset.
    f_error_distribution: array
        possible frequency error distribution for the dataset.
    idx_of_origin_v: array
        for each assigned identity the index of the datapoint on which basis the assignement was made.
    """

    def clean_up(fund_v, ident_v):
        """
        deletes/replaces with np.nan those identities only consisting from little data points and thus are tracking
        artefacts. Identities get deleted when the proportion of the trace (slope, ratio of detected datapoints, etc.)
        does not fit a real fish.

        Parameters
        ----------
        fund_v: array
            flattened fundamtantals array containing all detected EOD frequencies in the recording.
        ident_v: array
            respective assigned identites throughout the tracking progress.
        idx_v: array
            respective index vectro impliing the time of the detected frequency.
        times: array
            respective time vector.

        Returns
        -------
        ident_v: array
            cleaned up identities vector.

        """
        # print('clean up')
        for ident in np.unique(ident_v[~np.isnan(ident_v)]):
            if np.median(np.abs(np.diff(fund_v[ident_v == ident]))) >= 0.25:
                ident_v[ident_v == ident] = np.nan
                continue

            if len(ident_v[ident_v == ident]) <= 10:
                ident_v[ident_v == ident] = np.nan
                continue

        return ident_v

    def get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, idx_comp_range, show=False):
        """
        extract temporal identities for a datasnippted of 2*index compare range of the original tracking algorithm.
        for each data point in the data window finds the best connection within index compare range and, thus connects
        the datapoints based on their minimal error value until no connections are left or possible anymore.

        Parameters
        ----------
        i0_m: 2d-array
            for consecutive timestamps contains for each the indices of the origin EOD frequencies.
        i1_m: 2d-array
            respectively contains the indices of the target EOD frequencies, laying within index compare range.
        error_cube: 3d-array
            error values for each combination from i0_m and the respective indices in i1_m.
        fund_v: array
            flattened fundamtantals array containing all detected EOD frequencies in the recording.
        idx_v: array
            respective index vectro impliing the time of the detected frequency.
        i: int
            loop variable and current index of interest for the assignment of tight connections.
        ioi_fti: int
            index of interest for temporal identities.
        dps: float
            detections per second. 1. / 'temporal resolution of the tracking'
        idx_comp_range: int
            index compare range for the assignment of two data points to each other.

        Returns
        -------
        tmp_ident_v: array
            for each EOD frequencies within the index compare range for the current time step of interest contains the
            temporal identity.
        errors_to_v: array
            for each assigned temporal identity contains the error value based on which this connection was made.

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
        # tmp_idx_v -= min_i0

        embed()
        quit()
        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube, axis=None), np.shape(cp_error_cube))

        made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections[~np.isnan(cp_error_cube)] = 1
        # made_connections[~np.isnan(cp_error_cube)] = 0

        layers = layers + 1

        i_non_nan = len(cp_error_cube[layers - 1, idx0s, idx1s][~np.isnan(cp_error_cube[layers - 1, idx0s, idx1s])])

        if show:
            da.life_tmp_ident_init(min_i0, max_i1)

        for enu, layer, idx0, idx1 in zip(np.arange(i_non_nan), layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):
            if enu in np.array(np.floor(np.linspace(0, i_non_nan, 5)), dtype=int)[1:3]:
                if show:
                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v
                    da.tmp_ident_v_state.append(tmp_ident_v_ret)

            if np.isnan(cp_error_cube[layer - 1, idx0, idx1]):
                break
            # _____ some control functions _____ ###

            if not ioi_fti:
                if tmp_idx_v[i1_m[layer][idx1]] - i > idx_comp_range * 3:
                    continue
            else:
                if idx_v[i1_m[layer][idx1]] - idx_v[ioi_fti] > idx_comp_range * 3:
                    continue

            if np.isnan(tmp_ident_v[i0_m[layer][idx0]]):
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    # if np.abs(tmp_fund_v[i0_m[layer][idx0]] - tmp_fund_v[i1_m[layer][idx1]]) > 0.5: # ToDo: why it thin not freq_th?
                    #     continue

                    tmp_ident_v[i0_m[layer][idx0]] = next_tmp_identity
                    tmp_ident_v[i1_m[layer][idx1]] = next_tmp_identity
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1
                    next_tmp_identity += 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

                    if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                        if show:
                            da.itter_counter += 1
                            da.life_tmp_ident_update(tmp_ident_v_ret, new = tmp_ident_v[i0_m[layer][idx0]])
                    # TODO: ALL NEW
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
                    if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                        if show:
                            da.itter_counter += 1
                            da.life_tmp_ident_update(tmp_ident_v_ret, update = tmp_ident_v[i1_m[layer][idx1]])
                    # TODO: UPDATE I1

            else:
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    if tmp_idx_v[i1_m[layer][idx1]] in tmp_idx_v[mask]:
                        continue

                    same_id_idx = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    f_after = tmp_fund_v[same_id_idx[same_id_idx > i1_m[layer][idx1]]]
                    f_before = tmp_fund_v[same_id_idx[same_id_idx < i1_m[layer][idx1]]]
                    compare_freqs = []
                    if len(f_after) > 0:
                        compare_freqs.append(f_after[0])
                    if len(f_before) > 0:
                        compare_freqs.append(f_before[-1])
                    if len(compare_freqs) == 0:
                        continue
                    else:
                        if np.all(np.abs(np.array(compare_freqs) - tmp_fund_v[i1_m[layer][idx1]]) > 0.5):
                            continue

                    tmp_ident_v[i1_m[layer][idx1]] = tmp_ident_v[i0_m[layer][idx0]]
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v
                    if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                        if show:
                            da.itter_counter += 1
                            da.life_tmp_ident_update(tmp_ident_v_ret, update = tmp_ident_v[i0_m[layer][idx0]])
                    # TODO: UPDATE I0
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

                    del_idx = tmp_ident_v[i0_m[layer][idx0]]
                    tmp_ident_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]] = tmp_ident_v[i1_m[layer][idx1]]

                    if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                        errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                        not_made_connections[layer - 1, idx0, idx1] = 0
                        made_connections[layer - 1, idx0, idx1] = 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v
                    if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                        if show:
                            da.life_tmp_ident_update(tmp_ident_v_ret, update = tmp_ident_v[i1_m[layer][idx1]], delete = del_idx)
                    # TODO: UPDATE I1; delete i0

            origin_idx = i0_m[layer][idx0]
            targets_mask = np.arange(len(cp_error_cube[layer - 1, idx0, :]))[~np.isnan(cp_error_cube[layer - 1, idx0, :])]
            target_idxs = i1_m[layer][targets_mask]
            target_idx = i1_m[layer][idx1]

            # if len(tmp_idx_v[target_idxs][tmp_idx_v[target_idxs] == tmp_idx_v[target_idx]]) >= 2:
            #     alternatives = target_idxs[np.arange(len(target_idxs))[tmp_idx_v[target_idxs] == tmp_idx_v[target_idx]]]
            #     da.plot_assign(origin_idx + min_i0, target_idx + min_i0, alternatives + min_i0)


        tmp_ident_v_ret = np.full(len(fund_v), np.nan)
        tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

        if show:
            plt.close()
            da.tmp_ident_v_state.append(tmp_ident_v_ret)
            da.static_tmp_id_tracking(min_i0, max_i1)
        #### this is new and in progress ####
        # ToDo: cut those strange ones... NOPE identify potential new partner first!!! s.u.

        #print(da.itter_counter)
        plt.show()

        return tmp_ident_v_ret, errors_to_v

    def get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims, freq_tolerance):
        f_error_distribution = []
        a_error_distribution = []

        i0s = []
        i1s = []

        for i in tqdm(range(start_idx, int(start_idx + idx_comp_range * 3)), desc='error dist'):
            i0_v = np.arange(len(idx_v))[
                (idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
            i1_v = np.arange(len(idx_v))[
                (idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (
                            fund_v <= freq_lims[1])]  # indices of possible targets

            if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
                continue

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < freq_lims[0] or fund_v[i0_v[enu0]] > freq_lims[1]:
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < freq_lims[0] or fund_v[i1_v[enu1]] > freq_lims[1]:
                        continue
                    # if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                    #     continue
                    a_error_distribution.append(np.sqrt(np.sum(
                        [(sign_v[i0_v[enu0]][k] - sign_v[i1_v[enu1]][k]) ** 2 for k in
                         range(len(sign_v[i0_v[enu0]]))])))
                    f_error_distribution.append(np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]))
                    i0s.append(i0_v[enu0])
                    i1s.append(i1_v[enu1])

        return np.array(a_error_distribution), np.array(f_error_distribution), np.array(i0s), np.array(i1s)

    def reshape_data():
        detection_time_diff = times[1] - times[0]
        dps = 1. / detection_time_diff
        fund_v = np.hstack(fundamentals)
        ident_v = np.full(len(fund_v), np.nan)  # fish identities of frequencies
        idx_of_origin_v = np.full(len(fund_v), np.nan)  # ToDo: necessary ? lets see

        idx_v = []  # temportal indices
        sign_v = []  # power of fundamentals on all electrodes
        for enu, funds in enumerate(fundamentals):
            idx_v.extend(np.ones(len(funds)) * enu)
            sign_v.extend(signatures[enu])
        idx_v = np.array(idx_v, dtype=int)
        sign_v = np.array(sign_v)

        original_sign_v = sign_v
        if np.shape(sign_v)[1] > 2:
            sign_v = (sign_v - np.min(sign_v, axis=1).reshape(len(sign_v), 1)) / (
                    np.max(sign_v, axis=1).reshape(len(sign_v), 1) - np.min(sign_v, axis=1).reshape(len(sign_v), 1))

        idx_comp_range = int(np.floor(dps * max_dt))  # maximum compare range backwards for amplitude signature comparison

        return fund_v, ident_v, idx_v, sign_v, original_sign_v, idx_of_origin_v, idx_comp_range, dps

    def create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims, update=False):
        if update:
            i0_m.pop(0)
            i1_m.pop(0)
            error_cube.pop(0)
        else:
            error_cube = []  # [fundamental_list_idx, freqs_to_assign, target_freqs]
            i0_m = []
            i1_m = []

        if update:
            Citt = [cube_app_idx]
        else:
            # Citt = np.arange(start_idx, int(start_idx + idx_comp_range * 3))
            Citt = np.arange(start_idx, int(start_idx + idx_comp_range * 2))

        for i in Citt:
            i0_v = np.arange(len(idx_v))[
                (idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
            i1_v = np.arange(len(idx_v))[
                (idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (
                            fund_v <= freq_lims[1])]  # indices of possible targets

            i0_m.append(i0_v)
            i1_m.append(i1_v)

            if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
                error_cube.append(np.array([[]]))
                continue

            error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < freq_lims[0] or fund_v[i0_v[enu0]] > freq_lims[1]:  # ToDo:should be obsolete
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < freq_lims[0] or fund_v[i1_v[enu1]] > freq_lims[
                        1]:  # ToDo:should be obsolete
                        continue

                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue

                    a_error = np.sqrt(
                        np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
                    f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                    t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                    error = estimate_error(a_error, f_error, a_error_distribution)
                    error_matrix[enu0, enu1] = np.sum(error)
            error_cube.append(error_matrix)

        if update:
            cube_app_idx += 1
        else:
            cube_app_idx = len(error_cube)

        return error_cube, i0_m, i1_m, cube_app_idx

    def assign_tmp_ids(ident_v, tmp_ident_v, idx_v, fund_v, error_cube, idx_comp_range, next_identity, i0_m, i1_m,
                       freq_lims, show=False):
        if show:
            da.static_tmp_id_assign_init()
            da.tmp_ident_v = tmp_ident_v

        max_shape = np.max([np.shape(layer) for layer in error_cube], axis=0)
        cp_error_cube = np.full((len(error_cube), max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube):
            cp_error_cube[enu, :np.shape(error_cube[enu])[0], :np.shape(error_cube[enu])[1]] = layer

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube[:idx_comp_range], axis=None),
                                                np.shape(cp_error_cube[:idx_comp_range]))

        i_non_nan = len(cp_error_cube[layers - 1, idx0s, idx1s][~np.isnan(cp_error_cube[layers - 1, idx0s, idx1s])])
        min_i0 = np.min(np.hstack(i0_m))
        max_i1 = np.max(np.hstack(i1_m))

        p_ident_v = ident_v[min_i0:max_i1 + 1]
        p_tmp_ident_v = tmp_ident_v[min_i0:max_i1 + 1]
        p_idx_v = idx_v[min_i0:max_i1 + 1]
        p_fund_v = fund_v[min_i0:max_i1 + 1]

        p_i0_m = np.array(i0_m, dtype=object) - min_i0
        p_i1_m = np.array(i1_m, dtype=object) - min_i0

        already_assigned = []
        for layer, idx0, idx1 in zip(layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):
            idents_to_assigne = p_ident_v[~np.isnan(p_tmp_ident_v) & (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]

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

            if freq_lims:
                if p_fund_v[p_i0_m[layer][idx0]] > freq_lims[1] or p_fund_v[p_i0_m[layer][idx0]] < freq_lims[0]:
                    continue
                if p_fund_v[p_i1_m[layer][idx1]] > freq_lims[1] or p_fund_v[p_i1_m[layer][idx1]] < freq_lims[0]:
                    continue

            if np.isnan(p_ident_v[p_i0_m[layer][idx0]]):
                continue

            idxs_i0 = p_idx_v[(p_ident_v == p_ident_v[p_i0_m[layer][idx0]]) & (p_idx_v > i + idx_comp_range) &
                              (p_idx_v <= i + idx_comp_range * 2)]
            idxs_i1 = p_idx_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) & (np.isnan(p_ident_v)) &
                              (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]

            if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                continue

            if p_i1_m[layer][idx1] in already_assigned:
                continue

            #########################################################
            # if show:
                # origin_idx = p_i0_m[layer][idx0]
                # target_idx = p_i1_m[layer][idx1]
                # alt_mask = cp_error_cube[]
                # cp_error_cube[layer - 1, idx0, :]
                # embed()
                # quit()
                # targets_mask = np.arange(len(cp_error_cube[layer - 1, idx0, :]))[~np.isnan(cp_error_cube[layer - 1, idx0, :])]
                # target_idxs = p_i1_m[layer][targets_mask]
                # target_idx = p_i1_m[layer][idx1]
                #
                # if len(p_tmp_ident_v[target_idxs][p_tmp_ident_v[target_idxs] == p_tmp_ident_v[target_idx]]) >= 2:
                # #     alternatives = target_idxs[np.arange(len(target_idxs))[tmp_idx_v[target_idxs] == tmp_idx_v[target_idx]]]
                # #     da.plot_assign(origin_idx + min_i0, target_idx + min_i0, alternatives + min_i0)
            #########################################################

            already_assigned.append(p_i1_m[layer][idx1])

            p_ident_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) &
                      (np.isnan(p_ident_v)) & (p_idx_v > i + idx_comp_range) &
                      (p_idx_v <= i + idx_comp_range * 2)] = p_ident_v[p_i0_m[layer][idx0]]

        for ident in np.unique(p_tmp_ident_v[~np.isnan(p_tmp_ident_v)]):
            if len(p_ident_v[p_tmp_ident_v == ident][~np.isnan(p_ident_v[p_tmp_ident_v == ident])]) == 0:
                p_ident_v[(p_tmp_ident_v == ident) & (p_idx_v > i + idx_comp_range) & (
                        p_idx_v <= i + idx_comp_range * 2)] = next_identity
                next_identity += 1

        return ident_v, next_identity

    if emit:
        Emit = Emit_progress()

    fund_v, ident_v, idx_v, sign_v, original_sign_v, idx_of_origin_v, idx_comp_range, dps = reshape_data()

    start_idx = 0 if not ioi_fti else idx_v[ioi_fti]  # Index Of Interest for temporal identities

    a_error_distribution, f_error_distribution, error_dist_i0s, error_dist_i1s = \
        get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims,
                               freq_tolerance=freq_tolerance)

    error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m=None, i1_m=None, error_cube=None, freq_lims=freq_lims,
                                                             cube_app_idx=None)


    da = Display_agorithm(fund_v, ident_v, idx_v, sign_v, times, a_error_distribution, error_dist_i0s, error_dist_i1s)

    # amplitude error with 4 examples
    # da.plot_a_error_dist()

    next_identity = 0
    next_cleanup = int(idx_comp_range * 120)

    for i in tqdm(np.arange(len(fundamentals)), desc='tracking'):
        if emit == True:
            Emit_progress.progress.emit(i / len(fundamentals) * 100)

        if len(np.hstack(i0_m)) == 0 or len(np.hstack(i1_m)) == 0:
            error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims,
                                                                     update=True)
            start_idx += 1
            continue

        if i >= next_cleanup:  # clean up every 10 minutes
            ident_v = clean_up(fund_v, ident_v)
            next_cleanup += int(idx_comp_range * 120)

        if i % idx_comp_range == 0:  # next total sorting step
            # a_error_distribution, f_error_distribution, error_dist_i0s, error_dist_i1s = \
            #     get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims, freq_tolerance)
            # da.a_error_dist = a_error_distribution
            # da.error_dist_i0s = error_dist_i0s
            # da.error_dist_i1s = error_dist_i1s

            #show_plotting = True if 17150 <= times[i] < 17160 else False
            if visualize:
                show_plotting = True if 17160 <= times[i] < 17170 else False
            else:
                show_plotting = False
            tmp_ident_v, errors_to_v = get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti,
                                                          idx_comp_range, show=show_plotting)

            if i == 0:
                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    ident_v[(tmp_ident_v == ident) & (idx_v <= i + idx_comp_range)] = next_identity
                    next_identity += 1

            # assing tmp identities ##################################

            ident_v, next_identity = assign_tmp_ids(ident_v, tmp_ident_v, idx_v, fund_v, error_cube, idx_comp_range,
                                                    next_identity, i0_m, i1_m, freq_lims, show=show_plotting)

        error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims,
                                                                 update=True)
        start_idx += 1

    ident_v = clean_up(fund_v, ident_v)

    return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v, original_sign_v


def estimate_error(a_error, f_error, a_error_distribution):
    a_weight = 2. / 3
    f_weight = 1. / 3
    # a_weight = 3. / 4
    # f_weight = 1. / 4
    if len(a_error_distribution) > 0:
        a_e = a_weight * len(a_error_distribution[a_error_distribution < a_error]) / len(a_error_distribution)
    else:
        a_e = 1
    f_e = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.35, dx=.08)
    # f_e = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.25, dx=.15)

    #f_e = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.50, dx=.10)

    # newset:
    #f_e = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.25, dx=.10)
    # plt.plot(f_error, boltzmann(f_error, alpha=1, beta=0, x0=.35, dx=.08))
    # plt.ylim(0, 1)
    # plt.show()

    return [a_e, f_e, 0]


def boltzmann(t, alpha=0.25, beta=0.0, x0=4, dx=0.85):
    """
    Calulates a boltzmann function.

    Parameters
    ----------
    t: array
        time vector.
    alpha: float
        max value of the boltzmann function.
    beta: float
        min value of the boltzmann function.
    x0: float
        time where the turning point of the boltzmann function occurs.
    dx: float
        slope of the boltzman function.

    Returns
    -------
    array
        boltzmann function of the given time array base on the other parameters given.
    """

    boltz = (alpha - beta) / (1. + np.exp(- (t - x0) / dx)) + beta
    return boltz


def load_example_data():

    # folder = "/home/raab/paper_create/2021_tracking/data/2016-04-10-11_12"
    # folder = "/home/raab/writing/2021_tracking/code/tracking_display/2016-04-10-11_12"
    folder = "/home/raab/writing/2021_tracking/data/2016-04-10-11_12"

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


def back_shape_data(fund_v, sign_v, idx_v, times):
    # t0 = 17160
    # # t0 = 1000
    # t1 = 17250
    # # t1 = 1200
    # mask = np.arange(len(idx_v))[(times[idx_v] >= t0) & (times[idx_v] <= t1)]

    mask = np.arange(len(idx_v))
    fundamentals = []
    signatures = []

    f = []
    s = []
    for i in tqdm(mask, desc='reshape data'):
        if i == mask[0]:
            f.append(fund_v[i])
            s.append(sign_v[i])
        else:
            if idx_v[i] != idx_v[i - 1]:
                fundamentals.append(np.array(f))
                f = []
                signatures.append(np.array(s))
                s = []
            f.append(fund_v[i])
            s.append(sign_v[i])
    fundamentals.append(f)
    signatures.append(s)

    return fundamentals, signatures


def plot_tracked_traces(ident_v, fund_v, idx_v, times):
    fig = plt.figure(figsize=(20/2.54, 12/2.54), facecolor='white')
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, bottom=0.1, right=.95, top=.95)
    ax = plt.subplot(gs[0, 0])

    for id in np.unique(ident_v[~np.isnan(ident_v)]):
        c = np.random.rand(3)
        ax.plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], color = c, marker='.')

    plt.show()


def main():
    fund_v, sign_v, idx_v, times, start_time, end_time = \
        load_example_data()

    fundamentals, signatures = back_shape_data(fund_v, sign_v, idx_v, times)

    fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v, original_sign_v = \
        freq_tracking_v5(fundamentals, signatures, times, visualize=True)

    plot_tracked_traces(ident_v, fund_v, idx_v, times)

    embed()
    quit()

if __name__ == '__main__':
    main()
