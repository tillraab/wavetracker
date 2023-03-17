import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import time
import math
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from functools import partial, partialmethod
from tqdm import tqdm
from .config import Configuration
from .datahandler import open_raw_data
from .spectrogram import Spectrogram
from .tracking import freq_tracking_v6
from .gpu_harmonic_group import harmonic_group_pipeline, get_fundamentals

from thunderfish.harmonics import harmonic_groups, fundamental_freqs
from thunderfish.powerspectrum import decibel

try:
    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if len(tf.config.list_physical_devices('GPU')):
        available_GPU = True
except:
    available_GPU = False


class Analysis_pipeline(object):
    def __init__(self, data, samplerate, channels, dataset, data_shape, cfg, folder, verbose, gpu_use=False):
        save_path = list(folder.split(os.sep))
        save_path.insert(-1, 'derived_data')
        self.save_path = os.sep.join(save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.data = data
        self.samplerate = samplerate
        self.channels = channels
        self.dataset = dataset
        self.data_shape = data_shape
        self.cfg = cfg
        self.folder = folder

        self.verbose = verbose
        self.gpu_use = gpu_use
        self.core_count = multiprocessing.cpu_count()

        self.Spec = Spectrogram(self.folder, self.samplerate, self.data_shape, verbose=verbose,
                                gpu_use=gpu_use, **cfg.raw, **cfg.spectrogram)

        # out
        self._fund_v = []
        self._idx_v = []
        self._sign_v = []
        self.ident_v = []
        self.times = []
        pass

    @property
    def fund_v(self):
        return np.array(self._fund_v)

    @property
    def idx_v(self):
        return np.array(self._idx_v)

    @property
    def sign_v(self):
        return np.array(self._sign_v)

    def run(self):
        if self.gpu_use:
            self.pipeline_GPU()
        else:
            self.pipeline_CPU()

        if self.verbose >= 1: print(f'{"Tracking":^25}: -- freq_tolerance: {self.cfg.tracking["freq_tolerance"]} -- '
                                    f'max_dt: {self.cfg.tracking["max_dt"]}')

        self.times = self.Spec.times
        self.ident_v = freq_tracking_v6(self.fund_v, self.idx_v, self.sign_v, self.times,
                                        **self.cfg.harmonic_groups, **self.cfg.tracking)


    def extract_snippet_signals(self):
        if self.gpu_use:
            assigned_hg, peaks, log_spec = harmonic_group_pipeline(self.Spec.sum_spec, self.Spec.spec_freqs, self.cfg)
            tmp_fundamentals = get_fundamentals(assigned_hg, self.Spec.spec_freqs)
            print('\nyay')
        else:
            partial_harmonic_groups = partial(harmonic_groups, self.Spec.spec_freqs, **self.cfg.harmonic_groups)
            a = partial_harmonic_groups(self.Spec.sum_spec[:, 0]) # TEST

            pool = multiprocessing.Pool(self.core_count - 1)
            a = pool.map(partial_harmonic_groups, self.Spec.sum_spec.transpose())

            groups_per_time = [a[groups][0] for groups in range(len(a))]
            tmp_fundamentals = pool.map(fundamental_freqs, groups_per_time)
            pool.terminate()

        tmp_fund_v = np.hstack(tmp_fundamentals)
        tmp_idx_v = np.array(np.hstack([np.ones(len(f)) * enu for enu, f in enumerate(tmp_fundamentals)]), dtype=int)
        f_idx = [np.argmin(np.abs(self.Spec.spec_freqs - f)) for i in range(len(tmp_fundamentals)) for f in
                 tmp_fundamentals[i]]

        tmp_sign_v = self.Spec.spec[:, f_idx, tmp_idx_v].transpose()

        idx_0 = len(self.Spec.times) - len(self.Spec.spec_times)

        self._fund_v.extend(tmp_fund_v)
        self._idx_v.extend(tmp_idx_v + idx_0)
        self._sign_v.extend(tmp_sign_v)

    def pipeline_GPU(self):
        if self.verbose >= 1: print(f'{"Spectrogram (GPU)":^25}: -- fine spec: '
                                    f'{self.Spec.get_fine_spec} -- plotable spec: {self.Spec.get_sparse_spec}'
                                    f' -- snippet size: {self.Spec.snippet_size / self.samplerate:.2f}s')

        iterations = int(np.ceil(self.data_shape[0] / self.Spec.snippet_size))
        pbar = tqdm(total=iterations)

        iter_counter = 0
        for enu, snippet_data in enumerate(self.dataset):
            t0_snip = time.time()
            snippet_t0 = self.Spec.itter_count * self.Spec.snippet_size / self.samplerate
            if self.data.shape[0] // self.Spec.snippet_size == self.Spec.itter_count:
                self.Spec.terminate = True

            t0_spec = time.time()
            self.Spec.snippet_spectrogram(snippet_data, snipptet_t0=snippet_t0)
            t1_spec = time.time()


            t0_hg = time.time()
            self.extract_snippet_signals()
            t1_hg = time.time()

            iter_counter += 1
            t1_snip = time.time()
            if self.verbose >= 3: print(f'{" ":^25}  Progress {iter_counter / iterations:3.1%} '
                                        f'-- Spectrogram: {t1_spec - t0_spec:.2f}s '
                                        f'-- Harmonic group: {t1_hg - t0_hg:.2f}s '
                                        f'--> {t1_snip-t0_snip:.2f}s', end="\r")
            pbar.update(1)
        pbar.close()


    def pipeline_CPU(self):
        if self.verbose >= 1: print(f'{"Spectrogram (CPU)":^25}: -- fine spec: '
                                    f'{self.Spec.get_fine_spec} -- plotable spec: {self.Spec.get_sparse_spec}'
                                    f' -- snippet size: {(self.Spec.snippet_size - self.Spec.noverlap) / self.samplerate:.2f}s')
        counter = 0
        iterations = self.data.shape[0] // (self.Spec.snippet_size - self.Spec.noverlap)
        for i0 in tqdm(np.arange(0, self.data.shape[0], self.Spec.snippet_size - self.Spec.noverlap), desc="File analysis."):
            t0_snip = time.time()
            snippet_t0 = i0 / self.samplerate

            if self.data.shape[0] // (self.Spec.snippet_size - self.Spec.noverlap) * (self.Spec.snippet_size - self.Spec.noverlap) == i0:
                self.Spec.terminate = True

            t0_spec = time.time()
            snippet_data = [self.data[i0: i0 + self.Spec.snippet_size, channel] for channel in self.Spec.channel_list]
            self.Spec.snippet_spectrogram(snippet_data, snipptet_t0=snippet_t0)
            t1_spec = time.time()

            t0_hg = time.time()
            self.extract_snippet_signals()
            t1_hg = time.time()
            t1_snip = time.time()
            if self.verbose >= 3: print(f'{" ":^25}  Progress {counter / iterations:3.1%} '
                                        f'-- Spectrogram: {t1_spec - t0_spec:.2f}s '
                                        f'-- Harmonic group: {t1_hg - t0_hg:.2f}s'
                                        f'--> {t1_snip-t0_snip:.2f}s', end="\r")
            counter += 1

def main():
    example_data = "/home/raab/data/2023-02-09-08_16"
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('-f', '--folder', type=str, help='file to be analyzed', default=example_data)
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    parser.add_argument('-r', '--renew', action='store_true', help='redo all analysis; dismiss pre-saved files.')
    parser.add_argument('-l', '--logging', action='store_true', help='store sys.out in log.txt.')
    parser.add_argument('-n', '--nosave', action='store_true', help='dont save spectrograms')
    args = parser.parse_args()
    args.folder = os.path.normpath(args.folder)

    if args.logging: sys.stdout = open('./log.txt', 'w')

    if args.verbose >= 1: print(f'\n--- Running wavetracker.wavetracker ---')

    if args.verbose >= 1: print(f'{"Hardware used":^25}: {"GPU" if not (args.cpu and available_GPU) else "CPU"}')

    if args.verbose != 2: tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # load wavetracker configuration
    cfg = Configuration(args.config, verbose=args.verbose)

    # load data
    analysis_folders = []
    for dirpath, dirnames, filenames in os.walk(args.folder, topdown=True):
        for filename in [f for f in filenames if f.endswith(".raw")]:
            analysis_folders.append(dirpath)

    if args.verbose >= 1: print(f'{"Files found to analyze":^25}: {len(analysis_folders)}')

    for folder in analysis_folders:

        data, samplerate, channels, dataset, data_shape = open_raw_data(folder=folder, verbose=args.verbose,
                                                                        **cfg.spectrogram)

        Analysis = Analysis_pipeline(data, samplerate, channels, dataset, data_shape, cfg, folder, args.verbose,
                                     gpu_use=not args.cpu and available_GPU)

        if args.renew:
            Analysis.Spec.get_sparse_spec, Analysis.Spec.get_fine_spec = True, True

        if args.nosave:
            Analysis.Spec.get_sparse_spec, Analysis.Spec.get_fine_spec = False, False
        Analysis.run()

        ##########################################
        fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54))
        ax.scatter(Analysis.times[Analysis.idx_v], Analysis.fund_v, color='grey', alpha = 0.5)
        for id in np.unique(Analysis.ident_v[~np.isnan(Analysis.ident_v)]):
            c = np.random.rand(3)
            ax.plot(Analysis.times[Analysis.idx_v[Analysis.ident_v == id]], Analysis.fund_v[Analysis.ident_v == id],
                    marker='.', color=c)
        plt.show()
    #########################################
    embed()
    quit()

    sys.stdout.close()

if __name__ == '__main__':
    main()
