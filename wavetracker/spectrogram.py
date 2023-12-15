import os
import argparse
import time
import multiprocessing
import numpy as np
from functools import partial, partialmethod
from matplotlib.mlab import specgram as mspecgram
from tqdm import tqdm

from .config import Configuration
from .datahandler import open_raw_data

from thunderfish.powerspectrum import get_window, decibel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if len(tf.config.list_physical_devices('GPU')):
        available_GPU = True
except:
    available_GPU = False


def get_step_and_overlap(overlap_frac, nfft, **kwargs):
    """
    Computes step size and overlap for spectrogram computation.

    Parameters
    ----------
        overlap_frac : float
            Fraction of how much nfft windows shall overlap during spectrogram computation.
        nfft : int
            Samples in one fft-window.
        kwargs : dict
            Excess parameters from the configuration dictionary passed to the function.

    Returns
    -------
        step : int
            Step size (samples) for fft-window in spectrogram analysis.
        noverlap : int
            Overlap (samples) of fft-windows.
    """
    step = int(nfft * (1-overlap_frac))
    noverlap = int(nfft * overlap_frac)
    return step, noverlap


def tensorflow_spec(data, samplerate, nfft, step, **kwargs):
    """
    Computes a soectrogram for a datasnippet with n samples recorded on m channels. The function is based on the
    tensorflow package and optimized for GPU use.

    Parameters
    ----------
        data : 2d-array, 2d-tensor
            Contains a snippet of raw-data from electrode (grid) recordings of electric fish. Data shape resembles
            samples (1st dimension) x channels (2nd dimension).
        samplerate : int
            Samplerate of the data.
        nfft : int
            Samples in one nfft window.
        step : int
            Samples by which consecutive nfft windows are shifted by.
        kwargs : dict
            Excess parameters from the configuration dictionary passed to the function.

    Returns
    -------
        ret_spectra : 2d-tensor
            Spectrogram computed for the given data.
        freqs : 1d-array
            Frequency array corresponding to the 2nd dimension of the computed spectrogram.
        times : 1d-array
            Time array corresponding to the 1st dimension of the computed spectrogram.
    """
    def conversion_to_old_scale(tf_spectrogram):
        """
        Scales the spectrogram computed using the tensporflow functionality to match the scale of the old way decribed
        in the function mlab_spec().

        Parameters
        ----------
            tf_spectrogram : 2d-tenspr
                Spectrogram

        Returns
        -------
            scaled_spectrogram : 2d-tensor
                Scales spectrogram that matches the the range if spectrograms returned by mlab_spec

        """
        # ToDo: kill this whole approch and write all spectrogram anaylsis in pytorch (CPU and GPU functionality).

        scaled_spectrogram = tf_spectrogram**2 * 4.05e-9
        return scaled_spectrogram

    # Run the computation on the GPU
    with tf.device('GPU:0'):
        # Compute the spectrogram using a short-time Fourier transform
        stft = tf.signal.stft(data, frame_length=nfft, frame_step=step, window_fn=tf.signal.hann_window)
        spectra = tf.abs(stft)
    ret_spectra = conversion_to_old_scale(spectra)

    # create frequency and time axis returned with the spectrogram
    freqs = np.fft.fftfreq(nfft, 1 / samplerate)[:int(nfft / 2) + 1]
    freqs[-1] = samplerate / 2
    times = np.linspace(0, int(tf.shape(data)[1]) / samplerate, int(spectra.shape[1]), endpoint=False)

    return ret_spectra, freqs, times


def mlab_spec(data, samplerate, nfft, noverlap, detrend='constant', window='hann', **kwargs):
    """
    This function maps its input parameters on the mspecgram spectrogram. The use of this function is mainly its
    utilization as patrial function (see functools) and the extraction if kwargs.

    Parameters
    ----------
        data : 2d-array
            Contains a snippet of raw-data from electrode (grid) recordings of electric fish. Data shape resembles
            samples (1st dimension) x channels (2nd dimension).
        samplerate : int
            Samplerate of the data.
        nfft : int
            Samples in one nfft window.
        noverlap : int
            Overlap (samples) of fft-windows.
        detrend: str
            If 'constant' subtract mean of data.
            If 'linear' subtract line fitted to the data.
            If 'none' do not deternd the data.
        window: str
            Function used for windowing data segements.
            One of hann, blackman, hamming, bartlett, boxcar, triang, parzen,
            bohman, blackmanharris, nuttall, fattop, barthann
            (see scipy.signal window functions).
        kwargs : dict
            Excess parameters from the configuration dictionary passed to the function.

    Returns
    -------
        spec : 2d-tensor
            Spectrogram computed for the given data.
        freqs : 1d-array
            Frequency array corresponding to the 2nd dimension of the computed spectrogram.
        times : 1d-array
            Time array corresponding to the 1st dimension of the computed spectrogram.
    """

    spec, freqs, times = mspecgram(data, NFFT=nfft, Fs=samplerate,
                                  noverlap=noverlap, detrend=detrend,
                                  scale_by_freq=True,
                                  window=get_window(window, nfft))
    return spec, freqs, times


class Spectrogram(object):
    """
    Tools to compute and collect spectrogram data while analyzing large files of electric fish. This includes the
    computation of spectrograms for data snippets and all channels separately and the storage and generation of a
    spares-  and full spectrogram which is summed up over all electrodes for the whole recording which is filled while
    different data snippets are analyzed. The sparse spectrogram is convinient for plotting larger data snippets,
    whereas the full spectrogram shows the original temporal and frequency resolution.

    The key function of this class is "snippet_spectrogram". Here, spectrograms of data snippets are computed, either
    using CPU or GPU depending on availability, and the generation and storage of spares- and full spectrograms are
    coordinated.

    """
    def __init__(self, samplerate, data_shape, snippet_size, nfft, overlap_frac, channels, gpu_use, verbose=0, folder=None,
                 core_count = None, **kwargs):
        """
        Constructs all the necessary attributes for the spectrogram analysis pipeline of the wavetracker-package to
        analyse electric grid recordings of wave-type electric fish. When a folder, corresponding to the datapath of
        the data that shall be analysed, is passed, this class checks if there is already a spare- and fine-spectrogram
        computed and stored for the respective dataset. If not, these are generated here while a full raw-file is
        processed except told else.

        Parameters
        ----------
            samplerate : int
                Samplerate of the data that shall be analyzed.
            data_shape : tuple
                Total length and channel count of the data that shall be analized,
            snippet_size : int
                Sample count that is contained in one data snippet handled by the respective spectogram functions at
                once.
            nfft : int
                Samples in one nfft window.
            overlap_frac : float
                Overlap (samples) of fft-windows.
            channels : int
                Channel count for the data the shall be analized
            gpu_use : bool
                If GPU is available chooses a different and faster analysis pathway.
            verbose : int
                Verbosity level regulating shell/logging feedback during analysis. Suggested for debugging in
                development.
            folder : str
                Folder where raw-data that shall be analysed here is stored.
            core_count : int
                CPU core count that can be used for simultaneous spectrogram analysis of different channels in one
                data-snippet.
            kwargs : dict
                Excess parameters from the configuration dictionary passed to the function.
        """
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
        """
        Get the overlap fraction of fft-windows for spectrogram analysis.
        """
        return self._overlap_frac

    @overlap_frac.setter
    def overlap_frac(self, value):
        """
        Sets the overlap fraction of fft-windows for spectrogram analysis. Since the overlap fraction directly
        influences the "step" and "noverlap" parameter, these are adjusted accordingly.

        Parameters
        ----------
            value : float
                Overlap fraction of fft-windows.
        """
        self._overlap_frac = value
        self.step, self.noverlap = get_step_and_overlap(self._overlap_frac, self.nfft)

    @property
    def get_sparse_spec(self):
        """
        Gets information whether a sparse spectrogram for the whole recording is computed on the fly while analyzing a
        whole file.
        """
        return bool(self._get_sparse_spec)

    @get_sparse_spec.setter
    def get_sparse_spec(self, get_sparse_s):
        """
        Sets whether a sparse spectrogram for the whole recording is computed on the fly while analyzing a whole file.
        If this is set to "True" associated parameters are set accordingly.

        Parameters
        ----------
            get_sparse_s : bool
                If "True" generates the sparse spectrogram during file analysis.
        """
        if get_sparse_s:
            self.sparse_spectra = None
            self.sparse_time_borders, self.sparse_freq_borders = None, None
            self.sparse_time, self.sparse_freq = None, None
        self._get_sparse_spec = bool(get_sparse_s)

    @property
    def get_fine_spec(self):
        """
        Gets information whether a fine spectrogram for the whole recording is generated and directly stored on the fly
        while analyzing a whole file.
        """
        return bool(self._get_fine_spec)

    @get_fine_spec.setter
    def get_fine_spec(self, get_fine_s):
        """
        Sets whether a fine spectrogram for the whole recording is generated and directly stored on the fly while
        analyzing a whole file. Associated parameters are set accordingly.

        Parameters
        ----------
            get_fine_s : bool
                If "True" generates and stores the fine spectrogram during file analysis.
        """
        if get_fine_s:
            self._get_fine_spec = True
            self.fine_spec = None
            self.fine_spec_shape = None
            self.fine_times = np.array([])
        self._get_fine_spec = bool(get_fine_s)


    def snippet_spectrogram(self, data_snippet, snipptet_t0):
        """
        As the key element of this class, this function computes a spectrogram for a passed data snippet.
        This is done using either CPU or GPU Hardware, depending on availability and settings. Furthermore, sparse- and
        full spectrograms for the covering the while recording can be generated.

        Parameters
        ----------
            data_snippet : 2d-array, 2d-tensor
                The current data snippet to analyse. The 1st dimension contains the data for the different recording
                channels (2nd dimension).
            snipptet_t0 : float
                Timeponit of the first datapoint in the data snippet in respect to the whole recording analized.
        """
        if self.gpu:
            self.spec, self.spec_freqs, spec_times = tensorflow_spec(data_snippet, samplerate=self.samplerate,
                                                                     step=self.step, nfft = self.nfft, **self.kwargs)
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
        """
        Create a sparse/plottable spectrogram for the whole recording (not only a data snippet). This is done by
        separating the time and frequency range of the whole recording into n time- and m frequency-segments, whereby n
        and m, the quantity of segments, correspond to the class attribute "monitor_resolution". For each combination of
        time and frequency segment the maximum value of the corresponding spectral powers in the summed up spectrogram
        of the corresponding snippet is extracted and assigned to the corresponding index in the sparse spectrogram
        matrix. Accordingly, this matrix is filled while analysing a whole recording file.

        """
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
        """
        Creates and updates a full spectrogram for the whole recording stored directly on the harddrive, via memmory
        mapping. Accordingly, this matrix is extended while analysing a whole recording file. Be cautious, since files
        get very large due to the lack of reduction in frequency and time resolution. Similar to the sparse spectrogram,
        this full spectrogram resembles the summed up spectrograms over all electrodes.

        """
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
        """
        Saves sparse and full spectrograms when the analysis is complete.
        """
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
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('file', nargs='?', type=str, help='file to be analyzed')
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    parser.add_argument('-r', '--renew', action='store_true', help='redo all analysis; dismiss pre-saved files.')
    args = parser.parse_args()

    args.file = os.path.abspath(args.file)
    folder = os.path.split(args.file)[0]

    if args.verbose >= 1: print(f'\n--- Running wavetracker.spectrogram ---')

    if args.verbose >= 1: print(f'{"Hardware used":^25}: {"GPU" if not (args.cpu and available_GPU) else "CPU"}')

    if args.verbose < 1: tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # load wavetracker configuration
    cfg = Configuration(args.config, verbose=args.verbose)

    # load data
    data, samplerate, channels, dataset, data_shape = open_raw_data(filename=args.file, verbose=args.verbose,
                                                                    **cfg.spectrogram)

    # Spectrogram
    Spec = Spectrogram(samplerate, data_shape, folder=folder, verbose=args.verbose,
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


if __name__ == "__main__":
    main()
