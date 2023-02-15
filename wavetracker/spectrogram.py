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
        if verbose == 1: print(f'tensor transpose: {time.time() - t0}'); t0 = time.time()

        # Compute the spectrogram using a short-time Fourier transform
        stft = tf.signal.stft(data, frame_length=nfft, frame_step=step, window_fn=tf.signal.hann_window)
        if verbose == 1: print(f'stft: {time.time() - t0}'); t0 = time.time()
        spec = tf.abs(stft)
        if verbose == 1: print(f'abs: {time.time() - t0}'); t0 = time.time()

        spectra = spec.numpy()
        if verbose == 1: print(f'result: {time.time() - t0} \n')

    freqs = np.fft.fftfreq(nfft, 1 / samplerate)[:int(nfft / 2) + 1]
    freqs[-1] = samplerate / 2
    times = np.linspace(0, int(tf.shape(data)[1]) / samplerate, int(spec.shape[1]), endpoint=False)

    return conversion_to_old_scale(spectra), freqs, times


def mlab_spec(data, samplerate, nfft, noverlap, detrend='constant', window='hann', **kwargs):
    spec, freqs, times = mspecgram(data, NFFT=nfft, Fs=samplerate,
                                  noverlap=noverlap, detrend=detrend,
                                  scale_by_freq=True,
                                  window=get_window(window, nfft))
    return spec, freqs, times


def create_plotable_spec(sum_spec,
                         freqs,
                         tmp_times,
                         start_time,
                         end_time,
                         sparse_spectra=None,
                         x_borders=None,
                         y_borders=None,
                         min_freq=0,
                         max_freq=2000):

    f1 = np.argmax(freqs > max_freq)
    plot_freqs = freqs[:f1]
    plot_spectra = sum_spec[:f1, :]

    ##################################
    fig_xspan = 20.
    fig_yspan = 12.
    fig_dpi = 96.
    no_x = fig_xspan * fig_dpi * 2
    no_y = fig_yspan * fig_dpi * 2

    # if not checked_xy_borders:
    if not hasattr(sparse_spectra, '__len__'):
        x_borders = np.linspace(start_time, end_time, int(no_x))
        y_borders = np.linspace(min_freq, max_freq, int(no_y))

        sparse_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

        recreate_matrix = False
        if (tmp_times[1] - tmp_times[0]) > (x_borders[1] - x_borders[0]):
            x_borders = np.linspace(start_time, end_time, int((end_time - start_time) // (tmp_times[1] - tmp_times[0]) + 1))
            recreate_matrix = True
        if (freqs[1] - freqs[0]) > (y_borders[1] - y_borders[0]):
            recreate_matrix = True
            y_borders = np.linspace(min_freq, max_freq, (max_freq - min_freq) // (freqs[1] - freqs[0]) + 1)
        if recreate_matrix:
            sparse_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

    for i in range(len(y_borders) - 1):
        for j in range(len(x_borders) - 1):
            if x_borders[j] > tmp_times[-1]:
                break
            if x_borders[j + 1] < tmp_times[0]:
                continue
            t_mask = np.arange(len(tmp_times))[(tmp_times >= x_borders[j]) & (tmp_times < x_borders[j + 1])]
            f_mask = np.arange(len(plot_spectra))[(plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i + 1])]
            if len(t_mask) == 0 or len(f_mask) == 0:
                continue
            sparse_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])

    return sparse_spectra, x_borders, y_borders


def create_fine_spec(sum_spec,
                     tmp_times,
                     folder,
                     buffer_spectra=None,
                     fine_spec=None,
                     fine_spec_shape=None,
                     terminate=False
                     ):
    fill_spec_str = os.path.join('/home/raab/analysis/fine_specs', os.path.split(folder)[-1], 'fill_spec.npy')
    if not hasattr(fine_spec, '__len__'):
        if not os.path.exists(os.path.join('/home/raab/analysis/fine_specs', os.path.split(folder)[-1])):
            os.makedirs(os.path.join('/home/raab/analysis/fine_specs', os.path.split(folder)[-1]))
        fine_spec = np.memmap(fill_spec_str, dtype='float', mode='w+',
                              shape=(len(sum_spec), len(tmp_times)), order='F')

        fine_spec[:, :] = sum_spec
        fine_spec_shape = np.shape(fine_spec)
    else:
        if not hasattr(buffer_spectra, '__len__'):
            buffer_spectra = sum_spec
        else:
            buffer_spectra = np.append(buffer_spectra, sum_spec, axis=1)

        if np.shape(buffer_spectra)[1] >= 500 or terminate:
            fine_spec = np.memmap(fill_spec_str, dtype='float', mode='r+', shape=(
                np.shape(buffer_spectra)[0], np.shape(buffer_spectra)[1] + fine_spec_shape[1]), order='F')
            fine_spec[:, -np.shape(buffer_spectra)[1]:] = buffer_spectra
            fine_spec_shape = np.shape(fine_spec)
            buffer_spectra = np.empty((fine_spec_shape[0], 0))

    return fine_spec, buffer_spectra, fine_spec_shape



def pipeline_spectrogram_gpu(dataset, samplerate, data_shape, snippet_size, verbose=0, **kwargs):
    # create a spectorgram of the whole recording
    get_sparse_spec = True
    sparse_spectra = None
    x_borders, y_borders = None, None

    step, noverlap = get_step_and_overlap(**kwargs)
    for enu, data in enumerate(dataset):
        result, spec_freqs, spec_times = tensorflow_spec(tf.transpose(data), samplerate=samplerate, verbose=verbose,
                                                         step=step, **kwargs)
        tmp_times = spec_times + enu * snippet_size / samplerate
        sum_spec = np.swapaxes(np.sum(result, axis=0), 0, 1)

        if get_sparse_spec:
            sparse_spectra, x_borders, y_borders = \
                create_plotable_spec(sum_spec, spec_freqs, tmp_times, 0, data_shape[0]/samplerate,
                                     sparse_spectra, x_borders, y_borders, min_freq=0, max_freq=2000)


    # fig, ax = plt.subplots(1, 1, figsize=(40 / 2.54, 24 / 2.54))
    # ax.pcolormesh(x_borders[:-1], y_borders[:-1], decibel(sparse_spectra), cmap='jet')
    # ax.set_title(f'Spectrogram (all channels)')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Frequency (Hz)')
    # plt.show()


def pipeline_spectrogram_cpu(data, samplerate, data_shape, folder, nfft, snippet_size, channels, verbose=0, **kwargs):
    get_sparse_spec = False
    sparse_spectra = None
    x_borders, y_borders = None, None

    get_fine_spec = True
    buffer_spectra = None
    fine_spec = None
    fine_spec_shape = None
    terminate = False

    times = np.array([])

    if verbose >=1:  print(f'{"Spectrogram (CPU)":^25}: fine spec: {get_fine_spec}; plotable spec: {get_sparse_spec}')

    step, noverlap = get_step_and_overlap(nfft=nfft, **kwargs)

    core_count = multiprocessing.cpu_count()
    channel_list = np.arange(data.channels) if channels == -1 else np.arange(channels)
    func = partial(mlab_spec, samplerate=samplerate, nfft=nfft, noverlap=noverlap)

    for i0 in tqdm(np.arange(0, data.shape[0], snippet_size - (noverlap)), desc="File analysis."):
        pool = multiprocessing.Pool(core_count - 1)

        a = pool.map(func, [data[i0: i0 + snippet_size, channel] for channel in
                            channel_list])  # ret: spec, freq, time

        spectra = [a[channel][0] for channel in range(len(a))]
        spec_freqs = a[0][1]
        spec_times = a[0][2]
        pool.terminate()

        tmp_times = spec_times + (i0 / samplerate)
        sum_spec = np.sum(spectra, axis=0)

        times = np.concatenate((times, tmp_times))

        # when you want to have a rough spectrogram to plot
        if get_sparse_spec:
            sparse_spectra, x_borders, y_borders = \
                create_plotable_spec(sum_spec, spec_freqs, tmp_times, 0, data_shape[0]/samplerate,
                                     sparse_spectra, x_borders, y_borders, min_freq=0, max_freq=2000)

        if get_fine_spec:
            if data.shape[0] // (snippet_size - noverlap) * (snippet_size - noverlap) == i0: terminate = True

            fine_spec, buffer_spectra, fine_spec_shape = create_fine_spec(
                sum_spec, tmp_times, folder, buffer_spectra=buffer_spectra, fine_spec=fine_spec,
                fine_spec_shape=fine_spec_shape, terminate=terminate)

            if terminate:
                np.save(os.path.join('/home/raab/analysis/fine_specs', os.path.split(folder)[-1],
                                     'fine_spec_shape.npy'), fine_spec_shape)
                np.save(os.path.join('/home/raab/analysis/fine_specs', os.path.split(folder)[-1],
                                     'fine_times.npy'), times)
                np.save(os.path.join('/home/raab/analysis/fine_specs', os.path.split(folder)[-1],
                                     'fine_freqs.npy'), spec_freqs)

                f1 = np.argmax(spec_freqs > 2000)
                plot_freqs = spec_freqs[:f1]

                fig, ax = plt.subplots(1, 1, figsize=(40 / 2.54, 24 / 2.54))
                ax.pcolormesh(times, plot_freqs, decibel(fine_spec[:f1, :]), cmap='jet')
                ax.set_title(f'Spectrogram (all channels)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                plt.show()

def main():
    # ToDo: add example dataset to git
    example_data = "/home/raab/data/2023-02-09-08_16"
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('-f', '--folder', type=str, help='file to be analyzed', default=example_data)
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    args = parser.parse_args()

    if args.verbose >= 1: print(f'\n--- Running wavetracker.spectrogram ---')

    if args.verbose < 1: tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # load wavetracker configuration
    cfg = Configuration(args.config, verbose=args.verbose)

    # load data
    data, samplerate, channels, dataset, data_shape = open_raw_data(folder=args.folder, verbose=args.verbose,
                                                                    **cfg.spectrogram)

    if available_GPU and not args.cpu:
        # example how to use gpu pipeline
        # ToDo: implement and test memmap stuff for GPU
        pipeline_spectrogram_gpu(dataset, samplerate=samplerate, data_shape=data_shape, verbose=args.verbose,
                                 **cfg.raw, **cfg.spectrogram)
    else:
        # example how to use cpu pipeline
        # ToDo: implement start and stop time
        pipeline_spectrogram_cpu(data, samplerate=samplerate, data_shape=data_shape, verbose=args.verbose, folder=args.folder,
                                 **cfg.raw, **cfg.spectrogram)


if __name__ == "__main__":
    main()