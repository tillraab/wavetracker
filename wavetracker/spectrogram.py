import sys
import os
import argparse
import time
import multiprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed
from functools import partial, partialmethod
from matplotlib.mlab import specgram as mspecgram
from tqdm import tqdm

from thunderfish.powerspectrum import get_window, decibel
from .config import Configuration
from .datahandler import open_raw_data

try:
    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if len(tf.config.list_physical_devices('GPU')):
        available_GPU = True
except:
    available_GPU = False


def get_step_and_overlap(overlap_frac: float,
                         nfft: int,
                         **kwargs):
    step = int(nfft * (1-overlap_frac))
    noverlap = int(nfft * overlap_frac)
    return step, noverlap


def tensorflow_spec(data, samplerate, nfft, step, verbose = 1, **kwargs):
    def conversion_to_old_scale(x):
        return x**2 * 4.05e-9

    # Run the computation on the GPU
    t0 = time.time()
    with tf.device('GPU:0'):
        # if verbose >= 3: print(f'tensor transpose: {time.time() - t0}'); t0 = time.time()

        # Compute the spectrogram using a short-time Fourier transform
        stft = tf.signal.stft(data, frame_length=nfft, frame_step=step, window_fn=tf.signal.hann_window)
        # if verbose >= 3: print(f'stft: {time.time() - t0}'); t0 = time.time()
        spectra = tf.abs(stft)
        # if verbose >= 3: print(f'abs: {time.time() - t0}'); t0 = time.time()

        # t0 = time.time()
        # embed()
        # quit()
        # spectra = spectra.numpy()
        # print(f'transfere to numpy took: {time.time()-t0:.4f}s\n')
        # if verbose == 1: print(f'result: {time.time() - t0} \n')

    freqs = np.fft.fftfreq(nfft, 1 / samplerate)[:int(nfft / 2) + 1]
    freqs[-1] = samplerate / 2
    # ToDo: this is not completely correct
    times = np.linspace(0, int(tf.shape(data)[1]) / samplerate, int(spectra.shape[1]), endpoint=False)
    return conversion_to_old_scale(spectra), freqs, times


def mlab_spec(data, samplerate, nfft, noverlap, detrend='constant', window='hann', **kwargs):
    spec, freqs, times = mspecgram(data, NFFT=nfft, Fs=samplerate,
                                  noverlap=noverlap, detrend=detrend,
                                  scale_by_freq=True,
                                  window=get_window(window, nfft))
    return spec, freqs, times


class Spectrogram(object):
    def __init__(self, samplerate, data_shape, snippet_size, nfft, overlap_frac, channels, gpu_use, verbose=0, folder=None,
                 core_count = None, **kwargs):
        # meta parameters
        if folder != None:
            save_path = list(folder.split(os.sep))
            save_path.insert(-2, 'derived_data')
            self.save_path = os.sep.join(save_path)

        self.verbose = verbose
        self.kwargs = kwargs
        self.gpu = gpu_use

        # spectrogram parameters
        self.snippet_size = snippet_size
        self.nfft = nfft
        self._overlap_frac = overlap_frac
        self.channels = data_shape[1] if channels == -1 else channels
        self.channel_list = np.arange(self.channels)
        self.samplerate = samplerate
        self.data_shape = data_shape
        self.step, self.noverlap = get_step_and_overlap(self._overlap_frac, self.nfft)

        self.core_count = multiprocessing.cpu_count() if not core_count else core_count
        self.partial_func = partial(mlab_spec, samplerate=self.samplerate, nfft=self.nfft, noverlap=self.noverlap)

        # output
        self.itter_count = 0
        self.times = np.array([])
        self.sum_spec = None
        self.spec_times = None
        self.spec_freqs = None
        self.spec = None

        # additional tasks
        ### sparse spec

        self.min_freq, self.max_freq = 0, 2000
        self.monitor_res = (1920, 1080)

        self._get_fine_spec = False
        self._get_sparse_spec = False

        if folder:
            if not os.path.exists(os.path.join(self.save_path, 'sparse_spectra.npy')):
                self._get_sparse_spec = True
                if os.path.ismount(os.sep.join(self.save_path.split(os.sep)[:-2])):
                    self.fine_spec_str = os.path.join(os.sep, 'home', os.getlogin(), 'analysis', save_path[-1], 'sparse_spectra.npy')
                    if not os.path.exists(os.path.split(self.fine_spec_str)[0]):
                        os.makedirs(os.path.split(self.fine_spec_str)[0])
                self.sparse_spectra = None
                self.sparse_time_borders, self.sparse_freq_borders = None, None
                self.sparse_time, self.sparse_freq = None, None
            else:
                self.sparse_spectra = np.load(os.path.join(self.save_path, 'sparse_spectra.npy'))
                self.sparse_time_borders, self.sparse_freq_borders = None, None
                self.sparse_time = np.load(os.path.join(self.save_path, 'sparse_time.npy'))
                self.sparse_freq = np.load(os.path.join(self.save_path, 'sparse_freq.npy'))

            ### fine spec
            # ToDo: check if already existing in save folder
            self.fine_spec_str = os.path.join(self.save_path, 'fine_spec.npy')
            self.buffer_spectra = None

            if not os.path.exists(os.path.join(self.save_path, 'fine_spec_shape.npy')):
                self._get_fine_spec = True
                self.fine_spec = None
                self.fine_spec_shape = None
                self.fine_times = np.array([])
            else:
                self.fine_spec_shape = np.load(os.path.join(self.save_path, 'fine_spec_shape.npy'))
                self.fine_spec = np.memmap(self.fine_spec_str, dtype='float', mode='r',
                                           shape=(self.fine_spec_shape[0], self.fine_spec_shape[1]), order='F')
                self.fine_times = np.load(os.path.join(self.save_path, 'fine_times.npy'))
                self.spec_freqs = np.load(os.path.join(self.save_path, 'fine_freqs.npy'))
        self.terminate = False

    @property
    def overlap_frac(self):
        return self._overlap_frac

    @overlap_frac.setter
    def overlap_frac(self, value):
        self._overlap_frac = value
        self.step, self.noverlap = get_step_and_overlap(self._overlap_frac, self.nfft)

    @property
    def get_sparse_spec(self):
        return bool(self._get_sparse_spec)

    @get_sparse_spec.setter
    def get_sparse_spec(self, get_sparse_s):
        if get_sparse_s:
            self.sparse_spectra = None
            self.sparse_time_borders, self.sparse_freq_borders = None, None
            self.sparse_time, self.sparse_freq = None, None
        self._get_sparse_spec = bool(get_sparse_s)

    @property
    def get_fine_spec(self):
        return bool(self._get_fine_spec)

    @get_fine_spec.setter
    def get_fine_spec(self, get_fine_s):
        if get_fine_s:
            self._get_fine_spec = True
            self.fine_spec = None
            self.fine_spec_shape = None
            self.fine_times = np.array([])
        self._get_fine_spec = bool(get_fine_s)


    def snippet_spectrogram(self, data_snippet, snipptet_t0):
        # ToDo: I changed some things here so that the input can be the same for both pathways

        if self.gpu:
            self.spec, self.spec_freqs, spec_times = tensorflow_spec(data_snippet, samplerate=self.samplerate,
                                                             verbose=self.verbose, step=self.step, nfft = self.nfft,
                                                             **self.kwargs)
            self.spec = np.swapaxes(self.spec, 1, 2)
            self.sum_spec = np.sum(self.spec, axis=0)
            self.itter_count += 1
        else:
            self.step, self.noverlap = get_step_and_overlap(self._overlap_frac, self.nfft)
            self.partial_func = partial(mlab_spec, samplerate=self.samplerate, nfft=self.nfft, noverlap=self.noverlap)
            pool = multiprocessing.Pool(self.core_count - 1)
            a = pool.map(self.partial_func, data_snippet)  # ret: spec, freq, time
            self.spec = np.array([a[channel][0] for channel in range(len(a))])
            self.spec_freqs = a[0][1]
            spec_times = a[0][2]
            pool.terminate()
            # self.spec_times = spec_times + (i0 / self.samplerate)
            self.sum_spec = np.sum(self.spec, axis=0)

        self.spec_times = spec_times + snipptet_t0
        self.times = np.concatenate((self.times, self.spec_times))

        if self._get_sparse_spec:
            self.create_plotable_spec()

        if self._get_fine_spec:
            self.create_fine_spec()
            self.fine_times = np.concatenate((self.fine_times, self.spec_times))

        if self.terminate:
            self.save()

    def create_plotable_spec(self):
        f1 = np.argmax(self.spec_freqs > self.max_freq)
        plot_freqs = self.spec_freqs[:f1]
        plot_spectra = self.sum_spec[:f1, :]

        if not hasattr(self.sparse_spectra, '__len__'):
            self.sparse_time_borders = np.linspace(0, self.data_shape[0] / self.samplerate, int(self.monitor_res[0]))
            self.sparse_freq_borders = np.linspace(self.min_freq, self.max_freq, int(self.monitor_res[1]))

            self.sparse_spectra = np.zeros((len(self.sparse_freq_borders) - 1, len(self.sparse_time_borders) - 1))

            recreate_matrix = False
            if (self.spec_times[1] - self.spec_times[0]) > (self.sparse_time_borders[1] - self.sparse_time_borders[0]):
                self.sparse_time_borders = np.linspace(0, self.data_shape[0] / self.samplerate,
                                                       int((self.data_shape[0] / self.samplerate) //
                                                           (self.spec_times[1] - self.spec_times[0]) + 1))
                recreate_matrix = True
            if (self.spec_freqs[1] - self.spec_freqs[0]) > (self.sparse_freq_borders[1] - self.sparse_freq_borders[0]):
                recreate_matrix = True
                self.sparse_freq_borders = np.linspace(self.min_freq, self.max_freq,
                                                       (self.max_freq - self.min_freq) //
                                                       (self.spec_freqs[1] - self.spec_freqs[0]) + 1)
            if recreate_matrix:
                self.sparse_spectra = np.zeros((len(self.sparse_freq_borders) - 1, len(self.sparse_time_borders) - 1))

        for i in range(len(self.sparse_freq_borders) - 1):
            for j in range(len(self.sparse_time_borders) - 1):
                if self.sparse_time_borders[j] > self.spec_times[-1]:
                    break
                if self.sparse_time_borders[j + 1] < self.spec_times[0]:
                    continue
                t_mask = np.arange(len(self.spec_times))[(self.spec_times >= self.sparse_time_borders[j]) & (self.spec_times < self.sparse_time_borders[j + 1])]
                f_mask = np.arange(len(plot_spectra))[(plot_freqs >= self.sparse_freq_borders[i]) & (plot_freqs < self.sparse_freq_borders[i + 1])]
                if len(t_mask) == 0 or len(f_mask) == 0:
                    continue
                self.sparse_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])

    def create_fine_spec(self):
        if not hasattr(self.fine_spec, '__len__'):
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.fine_spec = np.memmap(self.fine_spec_str, dtype='float', mode='w+',
                                       shape=(len(self.spec_freqs), len(self.spec_times)), order='F')
            self.fine_spec[:, :] = self.sum_spec
            self.fine_spec_shape = np.shape(self.fine_spec)
        else:
            if not hasattr(self.buffer_spectra, '__len__'):
                self.buffer_spectra = self.sum_spec
            else:
                self.buffer_spectra = np.append(self.buffer_spectra, self.sum_spec, axis=1)
            if np.shape(self.buffer_spectra)[1] >= 500 or self.terminate:
                self.fine_spec = np.memmap(self.fine_spec_str, dtype='float', mode='r+', shape=(
                    self.fine_spec_shape[0], self.fine_spec_shape[1] + np.shape(self.buffer_spectra)[1]), order='F')
                self.fine_spec[:, -np.shape(self.buffer_spectra)[1]:] = self.buffer_spectra
                self.fine_spec_shape = np.shape(self.fine_spec)
                self.buffer_spectra = np.empty((self.fine_spec_shape[0], 0))

    def save(self):
        if self._get_sparse_spec:
            np.save(os.path.join(self.save_path, 'sparse_spectra.npy'), self.sparse_spectra)
            self.sparse_time = self.sparse_time_borders[:-1] + np.diff(self.sparse_time_borders) / 2
            np.save(os.path.join(self.save_path, 'sparse_time.npy'), self.sparse_time)
            self.sparse_freq = self.sparse_freq_borders[:-1] + np.diff(self.sparse_freq_borders) / 2
            np.save(os.path.join(self.save_path, 'sparse_freq.npy'), self.sparse_freq)

        if self._get_fine_spec:
            np.save(os.path.join(self.save_path, 'fine_spec_shape.npy'), self.fine_spec_shape)
            np.save(os.path.join(self.save_path, 'fine_times.npy'), self.times)
            np.save(os.path.join(self.save_path, 'fine_freqs.npy'), self.spec_freqs)


def main():
    # ToDo: add example dataset to git
    example_data = "/home/raab/data/2023-02-09-08_16"
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('-f', '--folder', type=str, help='file to be analyzed', default=example_data)
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    parser.add_argument('-r', '--renew', action='store_true', help='redo all analysis; dismiss pre-saved files.')
    args = parser.parse_args()
    args.folder = os.path.normpath(args.folder)

    if args.verbose >= 1: print(f'\n--- Running wavetracker.spectrogram ---')

    if args.verbose >= 1: print(f'{"Hardware used":^25}: {"GPU" if not (args.cpu and available_GPU) else "CPU"}')

    if args.verbose < 1: tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # load wavetracker configuration
    cfg = Configuration(args.config, verbose=args.verbose)

    # load data
    data, samplerate, channels, dataset, data_shape = open_raw_data(folder=args.folder, verbose=args.verbose,
                                                                    **cfg.spectrogram)

    # Spectrogram
    Spec = Spectrogram(samplerate, data_shape, folder=args.folder, verbose=args.verbose,
                       gpu_use=not args.cpu and available_GPU, **cfg.raw, **cfg.spectrogram)
    if args.renew:
        Spec._get_sparse_spec, Spec._get_fine_spec = True, True

    if available_GPU and not args.cpu:
        if args.verbose >= 1:print(f'{"Spectrogram (GPU)":^25}: -- fine spec: {Spec._get_fine_spec} -- plotable spec: {Spec._get_sparse_spec}')

        iterations = int(np.ceil(data_shape[0] / Spec.snippet_size))
        pbar = tqdm(total=iterations)
        for snippet_data in dataset:
            # last run !
            snippet_t0 = Spec.itter_count * Spec.snippet_size / samplerate
            if data.shape[0] // Spec.snippet_size == Spec.itter_count:
                Spec.terminate = True

            Spec.snippet_spectrogram(snippet_data, snipptet_t0=snippet_t0)
            pbar.update(1)
        pbar.close()

    else:
        if args.verbose >= 1: print(f'{"Spectrogram (CPU)":^25}: -- fine spec: {Spec._get_fine_spec} -- plotable spec: {Spec._get_sparse_spec}')
        for i0 in tqdm(np.arange(0, data.shape[0], Spec.snippet_size - Spec.noverlap), desc="File analysis."):
            snippet_t0 = i0 / samplerate

            if data.shape[0] // (Spec.snippet_size - Spec.noverlap) * (Spec.snippet_size - Spec.noverlap) == i0:
                Spec.terminate = True

            snippet_data = [data[i0: i0 + Spec.snippet_size, channel] for channel in Spec.channel_list]
            Spec.snippet_spectrogram(snippet_data, snipptet_t0=snippet_t0)

    # Spec.save() # ToDo: this instead of "Spec.terminate = True" ?!?!
    embed()
    quit()









    # if available_GPU and not args.cpu:
    #     # example how to use gpu pipeline
    #     # ToDo: implement and test memmap stuff for GPU
    #     pipeline_spectrogram_gpu(dataset, samplerate=samplerate, data_shape=data_shape, verbose=args.verbose, folder=args.folder,
    #                              **cfg.raw, **cfg.spectrogram)
    # else:
    #     # example how to use cpu pipeline
    #     # ToDo: implement start and stop time
    #     pipeline_spectrogram_cpu(data, samplerate=samplerate, data_shape=data_shape, verbose=args.verbose, folder=args.folder,
    #                              **cfg.raw, **cfg.spectrogram)


if __name__ == "__main__":
    main()