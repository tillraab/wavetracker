import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from plottools.colors import *
from plottools.tag import tag
colors_params(colors_muted, colors_tableau)
import os
import sys
from IPython import embed
from tqdm import tqdm
from PyQt5.QtCore import *

from thunderfish.powerspectrum import decibel

class Emit_progress():
    progress = pyqtSignal(float)

def gauss(t, shift, sigma, size, norm = False):
    g = np.exp(-((t - shift) / sigma) ** 2 / 2) * size
    if norm:
        g = g / np.sum(g) / (t[1] - t[0])
        # print(np.sum(g) * (t[1] - t[0]))
    return g

class Validate():
    def __init__(self):
        self.a_error_dist = None

        self.error_col = {}
        self.error_col['hit'] = []
        self.error_col['originID'] = []
        self.error_col['targetID'] = []
        self.error_col['alternID'] = []
        self.error_col['target_dfreq'] = []
        self.error_col['target_dfield'] = []
        self.error_col['target_freq_e'] = []
        self.error_col['target_field_e'] = []
        self.error_col['target_signal_e'] = []
        self.error_col['altern_dfreq'] = []
        self.error_col['altern_dfield'] = []
        self.error_col['altern_freq_e'] = []
        self.error_col['altern_field_e'] = []
        self.error_col['altern_signal_e'] = []

    def save_dict(self):
        np.save('./quantification/error_col.npy', self.error_col)
        np.save('./quantification/a_error_dist.npy', self.a_error_dist)

    def hist_kde(self, target_param, altern_param, sigma_factor = 1/10):
        help_array = np.concatenate((target_param, altern_param))
        error_steps = np.linspace(0, np.max(help_array) * 501 / 500, 500)

        kde_target = np.zeros(len(error_steps))
        for e in tqdm(target_param, desc='target'):
            kde_target += gauss(error_steps, e, np.std(target_param) * sigma_factor, 1, norm=True)

        kde_altern = np.zeros(len(error_steps))
        for e in tqdm(altern_param, desc='altern'):
            kde_altern += gauss(error_steps, e, np.std(altern_param) * sigma_factor, 1, norm=True)


        bin_edges = np.linspace(0, np.max(help_array), int(5 * (1/sigma_factor)))

        n_tar, _ = np.histogram(target_param, bin_edges)
        n_tar = n_tar / np.sum(n_tar) / (bin_edges[1] - bin_edges[0])
        n_alt, _ = np.histogram(altern_param, bin_edges)
        n_alt = n_alt / np.sum(n_alt) / (bin_edges[1] - bin_edges[0])

        return error_steps, kde_target, kde_altern, bin_edges, n_tar, n_alt

    def roc_analysis(self, error_steps, target_param, altern_param):
        true_pos = np.ones(len(error_steps))
        false_pos = np.ones(len(error_steps))
        for i in tqdm(range(len(error_steps)), desc='ROC'):
            true_pos[i] = len(np.array(target_param)[np.array(target_param) < error_steps[i]]) / len(target_param)
            false_pos[i] = len(np.array(altern_param)[np.array(altern_param) < error_steps[i]]) / len(altern_param)
        auc_value = np.sum(true_pos[:-1] * np.diff(false_pos))

        return true_pos, false_pos, auc_value

    def error_dist_and_auc_display(self):

        fig = plt.figure(figsize=(15/2.54, 10/2.54))
        gs = gridspec.GridSpec(2, 2, left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0.6, hspace=0.6, width_ratios=[2, 1])
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0]))
        ax_auc = []
        ax_auc.append(fig.add_subplot(gs[0, 1]))
        ax_auc.append(fig.add_subplot(gs[1, 1]))

        ax_m = []
        ax_m.append(ax[0].twinx())
        ax_m.append(ax[1].twinx())
        # ax_m = ax[0].twinx()
        ax_m[0].plot(np.linspace(0, 2.5, 1000), boltzmann(np.linspace(0, 2.5, 1000), alpha=1, beta=0, x0=.35, dx=.08), color='k')
        ax_m[0].set_ylim(bottom=0)
        ax_m[0].set_yticks([0, 1])
        ax_m[0].set_ylabel(r'$\varepsilon_{f}$', fontsize=10)

        ax_m[1].plot(self.a_error_dist[np.argsort(self.a_error_dist)], np.linspace(0, 1, len(self.a_error_dist)), color='k')
        ax_m[1].set_ylim(bottom=0)
        ax_m[1].set_yticks([0, 1])
        ax_m[1].set_ylabel(r'$\varepsilon_{S}$', fontsize=10)

        for enu, key0, key1, name in zip(np.arange(2), ['target_dfreq', 'target_dfield'], ['altern_dfreq', 'altern_dfield'], ['dfreq', 'dfield']):
            # for enu, key0, key1 in zip(np.arange(2), ['target_freq_e', 'target_field_e'], ['altern_freq_e', 'altern_field_e']):
            sigma_factor = 1 / 2 if enu in [0] else 1 / 10
            error_steps, kde_target, kde_altern, bin_edges, n_tar, n_alt = \
                self.hist_kde(self.error_col[key0], self.error_col[key1], sigma_factor)

            true_pos, false_pos, auc_value = self.roc_analysis(error_steps, self.error_col[key0], self.error_col[key1])

            np.save('./quantification/error_steps_%s.npy' % name, error_steps)
            np.save('./quantification/kde_target_%s.npy' % name, kde_target)
            np.save('./quantification/kde_altern_%s.npy' % name, kde_altern)
            np.save('./quantification/bin_edges_%s.npy' % name, bin_edges)
            np.save('./quantification/n_tar_%s.npy' % name, n_tar)
            np.save('./quantification/n_alt_%s.npy' % name, n_alt)

            np.save('./quantification/true_pos_%s.npy' % name, true_pos)
            np.save('./quantification/false_pos_%s.npy' % name, false_pos)
            np.save('./quantification/auc_value_%s.npy' % name, auc_value)

            print(len(self.error_col[key0]))
            target_handle, = ax[enu].plot(error_steps, kde_target / len(self.error_col[key0]), lw=2)
            altern_handle, = ax[enu].plot(error_steps, kde_altern / len(self.error_col[key1]), lw=2)

            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_tar, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.4, color=target_handle.get_color(), align='center')
            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_alt, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.4, color=altern_handle.get_color(), align='center')

            ax[enu].set_ylabel('KDE')
            ax[enu].set_xlim(error_steps[0], error_steps[-1])
            ax[enu].set_ylim(0, np.max(np.concatenate((n_tar, n_alt))) * 1.1)

            ax_auc[enu].fill_between(false_pos, np.zeros(len(false_pos)), true_pos, color='#999999')
            ax_auc[enu].plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
            ax_auc[enu].text(0.95, 0.05, '%.1f' % (auc_value * 100) + ' %', fontsize=9, color='k', ha='right', va='bottom')
            ax_auc[enu].set_xlim(0, 1)
            ax_auc[enu].set_ylim(0, 1)

            ax_auc[enu].set_xticks([0, 1])
            ax_auc[enu].set_yticks([0, 1])

        ax[0].set_xlabel(r'$\Delta f$ [Hz]', fontsize=10)
        ax[1].set_xlabel(r'field difference ($\Delta S$)', fontsize=10)

        ax_auc[0].set_ylabel('true positive', fontsize=10)
        ax_auc[1].set_ylabel('true positive', fontsize=10)
        ax_auc[1].set_xlabel('false positive', fontsize=10)

        for a in np.hstack([ax, ax_auc, ax_m]):
            a.tick_params(labelsize=9)
        fig.tag(axes=[ax], labels=['A', 'B'], fontsize=15, yoffs=2, xoffs=-6)
        plt.savefig('freq_field_difference.pdf')

        # fig.tag(axes=[ax_auc], labels=['B', 'D'], fontsize=15, yoffs=2, xoffs=-6)

        ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ######

        fig = plt.figure(figsize=(15/2.54, 14/2.54))
        gs = gridspec.GridSpec(3, 2, left=0.15, bottom=0.1, right=0.95, top=0.95, hspace=0.6, wspace=0.4, height_ratios=[4, 4, 4], width_ratios=[2.5, 1])
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0]))
        ax.append(fig.add_subplot(gs[2, 0]))

        ax_roc = []
        ax_roc.append(fig.add_subplot(gs[0, 1]))
        ax_roc.append(fig.add_subplot(gs[1, 1]))
        ax_roc.append(fig.add_subplot(gs[2, 1]))

        # ax_auc = fig.add_subplot(fig.add_subplot(gs[-1, :]))

        for enu, key0, key1, name in zip(np.arange(5),
                                         ['target_freq_e', 'target_field_e', 'target_signal_e'],
                                         ['altern_freq_e', 'altern_field_e', 'altern_signal_e'],
                                         ['freq_e', 'field_e', 'signal_e']):

            sigma_factor = 1 / 2 if enu in [0] else 1 / 10
            error_steps, kde_target, kde_altern, bin_edges, n_tar, n_alt = \
                self.hist_kde(self.error_col[key0], self.error_col[key1], sigma_factor)

            true_pos, false_pos, auc_value = self.roc_analysis(error_steps, self.error_col[key0], self.error_col[key1])

            # fig = plt.figure(figsize=(17.5/2.54, 7/2.54))
            # gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.2, top=0.95, right=0.95, width_ratios=[2, 1])
            # ax = fig.add_subplot(gs[0, 0])
            # ax_auc = fig.add_subplot(gs[0, 1])

            np.save('./quantification/error_steps_%s.npy' % name, error_steps)
            np.save('./quantification/kde_target_%s.npy' % name, kde_target)
            np.save('./quantification/kde_altern_%s.npy' % name, kde_altern)
            np.save('./quantification/bin_edges_%s.npy' % name, bin_edges)
            np.save('./quantification/n_tar_%s.npy' % name, n_tar)
            np.save('./quantification/n_alt_%s.npy' % name, n_alt)

            np.save('./quantification/true_pos_%s.npy' % name, true_pos)
            np.save('./quantification/false_pos_%s.npy' % name, false_pos)
            np.save('./quantification/auc_value_%s.npy' % name, auc_value)

            target_handle, = ax[enu].plot(error_steps, kde_target / len(self.error_col[key0]))
            altern_handle, = ax[enu].plot(error_steps, kde_altern / len(self.error_col[key1]))

            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_tar, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.5, color=target_handle.get_color())
            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_alt, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.5, color=altern_handle.get_color())

            ax[enu].set_ylabel('KDE', fontsize=10)
            # ax.set_xlabel(key0)
            help_array = np.concatenate((self.error_col[key0], self.error_col[key1]))
            ax[enu].set_xlim(0, np.percentile(help_array, 95))
            ax[enu].set_ylim(0, np.max(np.concatenate((n_tar, n_alt))) * 1.1)

            ax_roc[enu].fill_between(false_pos, np.zeros(len(false_pos)), true_pos, color='grey')
            ax_roc[enu].plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
            ax_roc[enu].text(0.95, 0.05, '%.1f' % (auc_value * 100), fontsize=10, color='k', ha='right', va='bottom')
            ax_roc[enu].set_xlim(0, 1)
            ax_roc[enu].set_ylim(0, 1)
            ax_roc[enu].set_xticks([0, 1])
            ax_roc[enu].set_yticks([0, 1])
            ax_roc[enu].set_ylabel('true positive', fontsize=10)
            if enu == 2:
                ax_roc[enu].set_xlabel('false positive', fontsize=10)
        ax[0].set_xlabel(r'$\varepsilon_{f}$', fontsize=10)
        ax[1].set_xlabel(r'$\varepsilon_{S}$', fontsize=10)
        ax[2].set_xlabel(r'$\varepsilon$', fontsize=10)

        for a in np.concatenate((ax, ax_roc)):
            a.tick_params(labelsize=9)
        fig.tag(axes=ax, fontsize=15, yoffs=1, xoffs=-6)
        plt.savefig('freq_field_signal_error.pdf')

    def which_is_best(self):

        for enu, key0, key1 in zip(np.arange(5),
                                   ['target_dfreq', 'target_dfield', 'target_freq_e', 'target_field_e', 'target_signal_e'],
                                   ['altern_dfreq', 'altern_dfield', 'altern_freq_e', 'altern_field_e', 'altern_signal_e']):


            fig = plt.figure(figsize=(17.5/2.54, 7/2.54))
            gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.2, top=0.95, right=0.95, width_ratios=[2, 1])
            ax = fig.add_subplot(gs[0, 0])
            ax_auc = fig.add_subplot(gs[0, 1])

            help_array = np.concatenate((self.error_col[key0], self.error_col[key1]))
            error_steps = np.linspace(0, np.max(help_array)*501/500, 500)

            kde_target = np.zeros(len(error_steps))
            sigma_factor = 1/2 if enu in [0, 2] else 1/10

            for e in self.error_col[key0]:
                kde_target += gauss(error_steps, e, np.std(self.error_col[key0]) * sigma_factor, 1, norm=True)

            kde_altern = np.zeros(len(error_steps))
            for e in self.error_col[key1]:
                kde_altern += gauss(error_steps, e, np.std(self.error_col[key1]) * sigma_factor, 1, norm=True)

            target_handle, = ax.plot(error_steps, kde_target / len(self.error_col[key0]))
            altern_handle, = ax.plot(error_steps, kde_altern / len(self.error_col[key1]))

            ###########################################################################################################

            bin_edges = np.linspace(0, np.percentile(help_array, 100), 50)

            n, _ = np.histogram(self.error_col[key0], bin_edges)
            n = n / np.sum(n) / (bin_edges[1] - bin_edges[0])
            n_alt, _ = np.histogram(self.error_col[key1], bin_edges)
            n_alt = n_alt / np.sum(n_alt) / (bin_edges[1] - bin_edges[0])

            ax.bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.5, color=target_handle.get_color())
            ax.bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_alt, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.5, color=altern_handle.get_color())
            ###########################################################################################################

            ax.set_ylabel('kde')
            ax.set_xlabel(key0)
            ax.set_xlim(left=0)

            ###
            true_pos = np.ones(len(error_steps))
            false_pos = np.ones(len(error_steps))
            for i in range(len(error_steps)):
                true_pos[i] = len(np.array(self.error_col[key0])[np.array(self.error_col[key0]) < error_steps[i]]) / len(self.error_col[key0])
                false_pos[i] = len(np.array(self.error_col[key1])[np.array(self.error_col[key1]) < error_steps[i]]) / len(self.error_col[key1])
            auc_value = np.sum(true_pos[:-1] * np.diff(false_pos))


            # ax_auc.plot(false_pos, true_pos, color='k', lw=1)
            ax_auc.fill_between(false_pos, np.zeros(len(false_pos)), true_pos, color='grey')
            ax_auc.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
            ax_auc.text(0.95, 0.05, '%.1f' % (auc_value * 100), fontsize=10, color='k', ha='right', va='bottom')
            ax_auc.set_xlim(0, 1)
            ax_auc.set_ylim(0, 1)

class Display_agorithm():
    def __init__(self, fund_v, ident_v, idx_v, sign_v, times, a_error_distribution, error_dist_i0s, error_dist_i1s):
        self.fund_v = fund_v
        self.sign_v = sign_v
        self.ident_v = ident_v
        self.tmp_ident_v = None
        self.final_ident_v = None
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
        self.origin_idx = []
        self.target_idx = []
        self.alt_idx = []

        self.tracking_i = None
        self.idx_comp_range = None

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

    def plot_assign(self, origin_idx, tartget_idx0, alternatives):
        #test_alt_idx = alt_target_idx[0] if alt_target_idx[0] != tartget_idx0 else alt_target_idx[1]
        alt_idx0 = alternatives[0]

        if np.abs(self.fund_v[origin_idx] - self.fund_v[tartget_idx0]) >= np.abs(self.fund_v[origin_idx] - self.fund_v[alt_idx0]):

            fig = plt.figure(figsize=(20/2.54, 12/2.54))
            gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.15, right=0.75, top=0.75)
            ax = fig.add_subplot(gs[0, 0])

            ax.imshow(decibel(self.spec)[::-1], extent=[self.times[0], self.times[-1], 0, 2000],
                      aspect='auto', alpha=0.7, cmap='jet', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)

            ax.plot(self.times[self.idx_v], self.fund_v, '.', color='grey', markersize=3)
            ax.plot(self.times[self.idx_v[origin_idx]], self.fund_v[origin_idx], '.', color='k', markersize=10)
            ax.plot(self.times[self.idx_v[tartget_idx0]], self.fund_v[tartget_idx0], '.', color='green', markersize=10)

            for alt_idx in alternatives:
                if alt_idx != tartget_idx0 and alt_idx != alt_idx0:
                    ax.plot(self.times[self.idx_v[alt_idx]], self.fund_v[alt_idx], '.', color='red', markersize=10)
            ax.plot(self.times[self.idx_v[alt_idx0]], self.fund_v[alt_idx0], '.', color='red', markersize=10, markeredgecolor='k')

            help_idx = np.concatenate((np.array([origin_idx, tartget_idx0]), np.array(alternatives)))
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

            for alt_idx in alternatives:
                if alt_idx != tartget_idx0:
                    df_target = np.abs(self.fund_v[alt_idx] - self.fund_v[origin_idx])
                    f_error = boltzmann(df_target, alpha=1, beta=0, x0=.35, dx=.08)

                    ax_ins.plot([df_target, df_target], [-.025, f_error], color='red', lw=4)
                    ax_ins.plot([-.025, df_target], [f_error, f_error], color='red', lw=4)

            plt.pause(2)
            plt.close('all')

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
        self.combo_fig.tag(axes=self.combo_ax, labels=['A', 'B', 'C'], fontsize=15, yoffs=1.5, xoffs=-7)

        plt.savefig('./fig5_tmp_ident_tracking.jpg', dpi=300)
        plt.show()
        self.tmp_ident_v_state = []

    def static_tmp_id_assign_init(self):

        t0 = self.times[self.tracking_i]
        for oi, ti, ai in zip(self.origin_idx, self.target_idx, self.alt_idx):
            self.fig2 = plt.figure(figsize=(17.5/2.54, 12/2.54))
            gs = gridspec.GridSpec(2, 2, left=.15, bottom=.1, right=.95, top=.9, hspace=0.3, wspace=0.2)
            self.ax2 = []
            # self.combo_ax.append(self.combo_fig.add_subplot(gs[0, 0]))
            self.ax2.append(self.fig2.add_subplot(gs[0, 0]))
            self.ax2.append(self.fig2.add_subplot(gs[0, 1]))
            self.ax2.append(self.fig2.add_subplot(gs[1, 0]))
            self.ax2.append(self.fig2.add_subplot(gs[1, 1]))

            for a in self.ax2:
                a.imshow(decibel(self.spec)[::-1], extent=[self.times[0]-t0, self.times[-1]-t0, 0, 2000], aspect='auto',
                         alpha=0.4, cmap='Greys', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)
                a.set_ylim(905, 930)

            # tmp_ident_time0 = self.times[self.idx_v[~np.isnan(self.tmp_ident_v)][0]]
            # self.ax2[0].set_xlim(tmp_ident_time0-10, tmp_ident_time0+30)

            # tmp_ids
            for tmp_id in np.unique(self.tmp_ident_v[~np.isnan(self.tmp_ident_v)]):
                Cmask = np.arange(len(self.idx_v))[(self.tmp_ident_v == tmp_id) &
                                                   (self.idx_v > self.tracking_i + self.idx_comp_range) &
                                                   (self.idx_v <= self.tracking_i + 2*self.idx_comp_range)]

                h, = self.ax2[0].plot(self.times[self.idx_v[self.tmp_ident_v == tmp_id]] -t0,
                                      self.fund_v[self.tmp_ident_v == tmp_id], lw=4, alpha=0.4)
                for a in self.ax2[:-1]:

                    c = h.get_color()
                    a.plot(self.times[self.idx_v[Cmask]] -t0, self.fund_v[Cmask], marker='.', color=c, markersize=4)
            # ids before connect
            for Cax in self.ax2[1:3]:
                for id in np.unique(self.ident_v[~np.isnan(self.ident_v)]):
                    Cax.plot(self.times[self.idx_v[self.ident_v == id]] -t0, self.fund_v[self.ident_v == id], marker='.', markersize=4)

            # ids after connect
            for id in np.unique(self.final_ident_v[~np.isnan(self.final_ident_v)]):
                self.ax2[3].plot(self.times[self.idx_v[self.final_ident_v == id]]-t0,
                                 self.fund_v[self.final_ident_v == id], marker='.', markersize=4)

            # connection points

            self.ax2[2].plot(self.times[self.idx_v[oi]] -t0, self.fund_v[oi], marker='o', markersize=6, color='k')
            self.ax2[2].plot(self.times[self.idx_v[ti]] -t0, self.fund_v[ti], marker='o', markersize=6, color='forestgreen')
            self.ax2[2].plot(self.times[self.idx_v[ai]] -t0, self.fund_v[ai], marker='o', markersize=6, color='firebrick')

            self.ax2[2].annotate("", xy=(self.times[self.idx_v[ti]] -t0, self.fund_v[ti]), xycoords='data',
                        xytext=(self.times[self.idx_v[oi]] -t0, self.fund_v[oi]), textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3, rad=-0.7",
                                        lw=2),
                        )

            # cosmetics
            for Cax, Cax2 in zip(self.ax2[:2], self.ax2[2:]):
                Cax.fill_between([0, 10], [932, 932], [934, 934], color='grey', clip_on=False)
                Cax.fill_between([10, 20], [932, 932], [934, 934], color='k', clip_on=False)
                Cax.fill_between([20, 30], [932, 932], [934, 934], color='grey', clip_on=False)

                for x0 in [0, 10, 20, 30]:
                    con = ConnectionPatch(xyA=(x0, 930), xyB=(x0, 905), coordsA="data", coordsB="data",
                                          axesA=Cax, axesB=Cax2, color="grey", linestyle='-',
                                          zorder=10, lw=1)
                    Cax2.add_artist(con)
                    Cax.plot([x0, x0], [930, 932], color='grey', lw=1, clip_on=False)

                Cax.set_xticks(np.arange(-10, 31, 10))
                Cax.set_xticklabels([])
                Cax2.set_xticks(np.arange(-10, 31, 10))
                Cax2.set_xticklabels(np.arange(-10, 31, 10))
                Cax2.set_xlabel('time [s]', fontsize=10)

                Cax.set_xlim(-10, 30)
                Cax2.set_xlim(-10, 30)
                Cax.tick_params(labelsize=9)
                Cax2.tick_params(labelsize=9)

            self.ax2[0].set_ylabel('frequency [Hz]', fontsize=10)
            self.ax2[2].set_ylabel('frequency [Hz]', fontsize=10)

            plt.setp(self.ax2[1].get_yticklabels(), visible=False)
            plt.setp(self.ax2[3].get_yticklabels(), visible=False)
        try:
            self.fig2.tag(axes=[self.ax2[0], self.ax2[2]], labels=['A', 'C'], fontsize=15, yoffs=2, xoffs=-8)
            self.fig2.tag(axes=[self.ax2[1], self.ax2[3]], labels=['B', 'D'], fontsize=15, yoffs=2, xoffs=-3)

            plt.savefig('fig_6assign_tmp_identities.jpg', dpi=300)
            plt.savefig('assign_tmp_identities2.jpg', dpi=300)
            plt.show()
        except:
            pass

    def finalize_tmp_id_assign(self, final_ident_v):
        for id in np.unique(final_ident_v[~np.isnan(final_ident_v)]):
            self.ax2[3].plot(self.times[self.idx_v[final_ident_v == id]], self.fund_v[final_ident_v == id], marker='.',
                             markersize=4)
        plt.show()

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

def freq_tracking_v5(fundamentals, signatures, times, freq_tolerance= 2.5, n_channels=64, max_dt=10., ioi_fti=False,
                     freq_lims=(200, 1200), emit = False, **kwargs):
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

    def get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, idx_comp_range, visualize=False, validate=False, validated_ident_v= None, **kwargs):
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
        def collect_validation_data():
            origin_idx = i0_m[layer][idx0] + min_i0
            target_idx = i1_m[layer][idx1] + min_i0

            collect = False
            if i + idx_comp_range < idx_v[origin_idx] < i + idx_comp_range * 2:
                if i + idx_comp_range < idx_v[target_idx] < i + idx_comp_range * 2:
                    collect = True

            if collect == False:
                return

            alt_i = np.arange(len(i1_m[layer]))[~np.isnan(error_cube[layer][idx0, :])]
            alt_e = error_cube[layer][idx0, :][alt_i]
            alt_idxs = i1_m[layer][~np.isnan(error_cube[layer][idx0, :])][np.argsort(alt_e)] + min_i0
            alt_idxs = alt_idxs[(alt_e > error_cube[layer][idx0, idx1])]  # alternatives with larger error
            # embed()
            # quit()
            if len(np.unique(validated_ident_v[alt_idxs])) > 1:
                alternatives = alt_idxs[
                    validated_ident_v[alt_idxs] != validated_ident_v[target_idx]]  # alternative with other id (all)
                alt_idx0 = alt_idxs[validated_ident_v[alt_idxs] != validated_ident_v[target_idx]][
                    0]  # alternative with other id (best)
                # alternatives = alt_idxs[np.arange(len(alt_idxs))[validated_ident_v[alt_idxs] != validated_ident_v[target_idx]]] # alternatives with othe id (all)

                a_error_target = np.sqrt(
                    np.sum([(sign_v[origin_idx][j] - sign_v[target_idx][j]) ** 2 for j in range(n_channels)]))
                f_error_target = np.abs(fund_v[origin_idx] - fund_v[target_idx])
                [a_e_target, f_e_target, t_e] = estimate_error(a_error_target, f_error_target, a_error_distribution)

                a_error_alt = np.sqrt(
                    np.sum([(sign_v[origin_idx][j] - sign_v[alt_idx0][j]) ** 2 for j in range(n_channels)]))
                f_error_alt = np.abs(fund_v[origin_idx] - fund_v[alt_idx0])
                [a_e_alt, f_e_alt, t_e] = estimate_error(a_error_alt, f_error_alt, a_error_distribution)

                if validate:
                    if validated_ident_v[origin_idx] == validated_ident_v[target_idx]:
                        va.error_col['hit'].append(True)

                        va.error_col['originID'].append(validated_ident_v[origin_idx])
                        va.error_col['targetID'].append(validated_ident_v[target_idx])

                        va.error_col['target_dfreq'].append(f_error_target)
                        va.error_col['target_dfield'].append(a_error_target)
                        va.error_col['target_freq_e'].append(f_e_target)
                        va.error_col['target_field_e'].append(a_e_target)
                        va.error_col['target_signal_e'].append(np.sum([a_e_target, f_e_target]))

                        va.error_col['alternID'].append(validated_ident_v[alt_idx0])

                        va.error_col['altern_dfreq'].append(f_error_alt)
                        va.error_col['altern_dfield'].append(a_error_alt)
                        va.error_col['altern_freq_e'].append(f_e_alt)
                        va.error_col['altern_field_e'].append(a_e_alt)
                        va.error_col['altern_signal_e'].append(np.sum([a_e_alt, f_e_alt]))

                    elif validated_ident_v[origin_idx] == validated_ident_v[alt_idx0]:
                        va.error_col['hit'].append(False)

                        va.error_col['originID'].append(validated_ident_v[origin_idx])
                        va.error_col['targetID'].append(validated_ident_v[alt_idx0])

                        va.error_col['target_dfreq'].append(f_error_alt)
                        va.error_col['target_dfield'].append(a_error_alt)
                        va.error_col['target_freq_e'].append(f_e_alt)
                        va.error_col['target_field_e'].append(a_e_alt)
                        va.error_col['target_signal_e'].append(np.sum([a_e_alt, f_e_alt]))

                        va.error_col['alternID'].append(validated_ident_v[target_idx])

                        va.error_col['altern_dfreq'].append(f_error_target)
                        va.error_col['altern_dfield'].append(a_error_target)
                        va.error_col['altern_freq_e'].append(f_e_target)
                        va.error_col['altern_field_e'].append(a_e_target)
                        va.error_col['altern_signal_e'].append(np.sum([a_e_target, f_e_target]))

                    else:
                        return
            return

        show = True if (17160 <= times[i] < 17170) and visualize else False


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

        if show:
            da.life_tmp_ident_init(min_i0, max_i1)

        for enu, layer, idx0, idx1 in zip(np.arange(i_non_nan), layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):
            if show:
                if enu in np.array(np.floor(np.linspace(0, i_non_nan, 5)), dtype=int)[1:3]:
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
                    tmp_ident_v[i0_m[layer][idx0]] = next_tmp_identity
                    tmp_ident_v[i1_m[layer][idx1]] = next_tmp_identity
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1
                    next_tmp_identity += 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

                    if show:
                        if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                            da.itter_counter += 1
                            da.life_tmp_ident_update(tmp_ident_v_ret, new = tmp_ident_v[i0_m[layer][idx0]])
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

                    if show:
                        if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                            da.itter_counter += 1
                            da.life_tmp_ident_update(tmp_ident_v_ret, update = tmp_ident_v[i1_m[layer][idx1]])

            else:
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    if tmp_idx_v[i1_m[layer][idx1]] in tmp_idx_v[mask]:
                        continue

                    # same_id_idx = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    # f_after = tmp_fund_v[same_id_idx[same_id_idx > i1_m[layer][idx1]]]
                    # f_before = tmp_fund_v[same_id_idx[same_id_idx < i1_m[layer][idx1]]]
                    # compare_freqs = []
                    # if len(f_after) > 0:
                    #     compare_freqs.append(f_after[0])
                    # if len(f_before) > 0:
                    #     compare_freqs.append(f_before[-1])
                    # if len(compare_freqs) == 0:
                    #     continue
                    # else:
                    #     if np.all(np.abs(np.array(compare_freqs) - tmp_fund_v[i1_m[layer][idx1]]) > 0.5):
                    #         continue

                    tmp_ident_v[i1_m[layer][idx1]] = tmp_ident_v[i0_m[layer][idx0]]
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1

                    tmp_ident_v_ret = np.full(len(fund_v), np.nan)
                    tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

                    if show:
                        if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                            da.itter_counter += 1
                            da.life_tmp_ident_update(tmp_ident_v_ret, update = tmp_ident_v[i0_m[layer][idx0]])
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

                    if show:
                        if 850 < tmp_fund_v[i0_m[layer][idx0]] < 950:
                            da.life_tmp_ident_update(tmp_ident_v_ret, update = tmp_ident_v[i1_m[layer][idx1]], delete = del_idx)

            if validate:
                collect_validation_data()

        tmp_ident_v_ret = np.full(len(fund_v), np.nan)
        tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

        if show:
            plt.close()
            da.tmp_ident_v_state.append(tmp_ident_v_ret)
            da.static_tmp_id_tracking(min_i0, max_i1)
            plt.show()

        return tmp_ident_v_ret, errors_to_v

    def get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims):
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
                    a_error_distribution.append(np.sqrt(np.sum(
                        [(sign_v[i0_v[enu0]][k] - sign_v[i1_v[enu1]][k]) ** 2 for k in
                         range(len(sign_v[i0_v[enu0]]))])))
                    f_error_distribution.append(np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]))
                    i0s.append(i0_v[enu0])
                    i1s.append(i1_v[enu1])

        return np.array(a_error_distribution), np.array(f_error_distribution), np.array(i0s), np.array(i1s)

    def create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims, update=False):
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
                       freq_lims, visualize=False, **kwargs):

        show = True if (17160 <= times[i] < 17170) and visualize else False

        if show:
            da.tmp_ident_v = np.copy(tmp_ident_v)
            da.ident_v = np.copy(ident_v)
            da.tracking_i = i
            da.idx_comp_range = idx_comp_range
            # da.static_tmp_id_assign_init()

        max_shape = np.max([np.shape(layer) for layer in error_cube], axis=0)
        cp_error_cube = np.full((len(error_cube), max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube):
            cp_error_cube[enu, :np.shape(error_cube[enu])[0], :np.shape(error_cube[enu])[1]] = layer

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube[:idx_comp_range], axis=None),
                                                np.shape(cp_error_cube[:idx_comp_range]))

        i_non_nan = len(cp_error_cube[layers, idx0s, idx1s][~np.isnan(cp_error_cube[layers, idx0s, idx1s])])
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

            already_assigned.append(p_i1_m[layer][idx1])

            p_ident_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) &
                      (np.isnan(p_ident_v)) & (p_idx_v > i + idx_comp_range) &
                      (p_idx_v <= i + idx_comp_range * 2)] = p_ident_v[p_i0_m[layer][idx0]]

            # print(fund_v[i0_m[layer][idx0]])
            if show:
                origin_idx = i0_m[layer][idx0]
                target_idx = i1_m[layer][idx1]
                alt_i = np.arange(len(i1_m[layer]))[~np.isnan(error_cube[layer][idx0, :])]
                alt_e = error_cube[layer][idx0, :][alt_i]
                alt_idxs = i1_m[layer][~np.isnan(error_cube[layer][idx0, :])][np.argsort(alt_e)]
                alt_idxs = alt_idxs[(idx_v[alt_idxs] >= i + idx_comp_range+1) &
                                    (alt_e > error_cube[layer][idx0, idx1])]
                if len(np.unique(tmp_ident_v[alt_idxs])) > 1:
                    if fund_v[origin_idx] >= 900:
                        print('yay')
                        da.origin_idx.append(origin_idx)
                        da.target_idx.append(target_idx)
                        da.alt_idx.append(alt_idxs[tmp_ident_v[alt_idxs] != tmp_ident_v[target_idx]][0])

        for ident in np.unique(p_tmp_ident_v[~np.isnan(p_tmp_ident_v)]):
            if len(p_ident_v[p_tmp_ident_v == ident][~np.isnan(p_ident_v[p_tmp_ident_v == ident])]) == 0:
                p_ident_v[(p_tmp_ident_v == ident) & (p_idx_v > i + idx_comp_range) & (
                        p_idx_v <= i + idx_comp_range * 2)] = next_identity
                next_identity += 1

        if show:
            da.final_ident_v = np.copy(ident_v)
            da.static_tmp_id_assign_init()
            #da.finalize_tmp_id_assign(np.copy(ident_v))

        return ident_v, next_identity

    def display_and_validation(validate=False, visualize=False, **kwargs):
        va = Validate() if validate else type('', (object,),{})()
        if visualize:
            da = Display_agorithm(fund_v, ident_v, idx_v, sign_v, times, a_error_distribution, error_dist_i0s, error_dist_i1s)
        else:
            da = type('', (object,),{})()

        show_plotting=False
        return va, da, show_plotting

    def validation(validate=False, **kwargs):
        if validate:
            va.which_is_best()
            va.a_error_dist = a_error_distribution
            va.save_dict()
            va.error_dist_and_auc_display()

    # if emit:
    #     Emit = Emit_progress()

    fund_v, ident_v, idx_v, sign_v, original_sign_v, idx_of_origin_v, idx_comp_range, dps = reshape_data()

    start_idx = 0 if not ioi_fti else idx_v[ioi_fti]  # Index Of Interest for temporal identities

    a_error_distribution, f_error_distribution, error_dist_i0s, error_dist_i1s = \
        get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims)

    error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m=None, i1_m=None, error_cube=None, freq_lims=freq_lims,
                                                             cube_app_idx=None)

    va, da, show_plotting = display_and_validation(**kwargs)

    next_identity = 0
    next_cleanup = int(idx_comp_range * 120)

    for i in tqdm(np.arange(len(fundamentals)), desc='tracking'):
        # if emit == True:
        #     Emit.progress.emit(i / len(fundamentals) * 100)

        if len(np.hstack(i0_m)) == 0 or len(np.hstack(i1_m)) == 0:
            error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims, update=True)
            start_idx += 1
            continue

        # if i >= next_cleanup:  # clean up every 10 minutes
        #     ident_v = clean_up(fund_v, ident_v)
        #     next_cleanup += int(idx_comp_range * 120)

        if i % idx_comp_range == 0:  # next total sorting step

            tmp_ident_v, errors_to_v = get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti,
                                                          idx_comp_range, **kwargs)

            if i == 0: # initial assignment of tmp_identities
                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    ident_v[(tmp_ident_v == ident) & (idx_v <= i + idx_comp_range)] = next_identity
                    next_identity += 1

            # assing tmp identities ##################################
            ident_v, next_identity = assign_tmp_ids(ident_v, tmp_ident_v, idx_v, fund_v, error_cube, idx_comp_range,
                                                    next_identity, i0_m, i1_m, freq_lims, **kwargs)

        error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims,
                                                                 update=True)
        start_idx += 1

    ident_v = clean_up(fund_v, ident_v)

    validation(**kwargs)
    # if validate:
    #     va.which_is_best()
    #     va.a_error_dist = a_error_distribution
    #     va.save_dict()
    #     va.error_dist_and_auc_display()

    return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v, original_sign_v


def estimate_error(a_error, f_error, a_error_distribution):
    a_weight = 2. / 3
    f_weight = 1. / 3
    if len(a_error_distribution) > 0:
        a_e = a_weight * len(a_error_distribution[a_error_distribution < a_error]) / len(a_error_distribution)
    else:
        a_e = 1
    f_e = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.35, dx=.08)

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


def load_example_data(folder=None):

    if folder == None:
        folder = "/home/raab/writing/2021_tracking/data/2016-04-10-11_12"

    if os.path.exists(os.path.join(folder, 'fund_v.npy')):
        fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
        sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
        validated_ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
        idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
        times = np.load(os.path.join(folder, 'times.npy'))
        start_time, end_time = np.load(os.path.join(folder, 'meta.npy'))
    else:
        fund_v, sign_v, idx_v, times, start_time, end_time  = [], [], [], [], [], []
        print('WARNING !!! files not found !')

    return fund_v, sign_v, idx_v, times, start_time, end_time, validated_ident_v


def back_shape_data(fund_v, sign_v, idx_v):
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
                signatures.append(np.array(s))
                f = []
                s = []
                counter = 1
                while idx_v[i] != idx_v[i - 1] + counter :
                    fundamentals.append(np.array(f))
                    signatures.append(np.array(s))
                    counter += 1

            f.append(fund_v[i])
            s.append(sign_v[i])
    fundamentals.append(f)
    signatures.append(s)

    return fundamentals, signatures


def plot_tracked_traces(ident_v, fund_v, idx_v, times):
    fig = plt.figure(figsize=(20/2.54, 12/2.54), facecolor='white')
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=.95, top=.95)
    ax = fig.add_subplot(gs[0, 0])

    for id in np.unique(ident_v[~np.isnan(ident_v)]):
        c = np.random.rand(3)
        ax.plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], color = c, marker='.')
    plt.show()


def get_kwargs(emit = True, visualize=True, validate=True, validated_ident_v = None):
    kwargs = {'emit': emit,
              'visualize': visualize,
              'validate':validate,
              'validated_ident_v': validated_ident_v}
    return kwargs


def main():
    if len(sys.argv) >=2:
        folder = sys.argv[1]
    else:
        folder = None

    # ---------------- Example data for tracking paper ---------------- #
    fund_v, sign_v, idx_v, times, start_time, end_time, validated_ident_v = load_example_data(folder=folder)
    fundamentals, signatures = back_shape_data(fund_v, sign_v, idx_v)
    # ----------------------------------------------------------------- #
    kwargs = get_kwargs(emit = True, visualize=True, validate=True, validated_ident_v = validated_ident_v)

    fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v, original_sign_v = \
        freq_tracking_v5(fundamentals, signatures, times, **kwargs)


    plot_tracked_traces(ident_v, fund_v, idx_v, times)

if __name__ == '__main__':
    main()
