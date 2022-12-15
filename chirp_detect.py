import sys
import glob
import os
import multiprocessing
import numpy as np
import scipy.signal as sig
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thunderfish.dataloader import DataLoader as open_data
from thunderfish.powerspectrum import decibel, spectrogram, next_power_of_two, nfft
from functools import partial
from IPython import embed
from tqdm import tqdm

import matplotlib.mlab as mlab

# python3 chirp_detect.py /home/raab/data/2022_competition/2022-06-02-10_00/

import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def bandpass_filter(data, rate, lowf=100, highf=1100):
    """
    Bandpass filter the signal.
    """
    sos = sig.butter(2, (lowf, highf), 'bandpass', fs=rate, output='sos')
    fdata = sig.sosfiltfilt(sos, data)
    return fdata

def envelope(data, rate, freq=100.0):
    sos = sig.butter(2, freq, 'lowpass', fs=rate, output='sos')
    envelope = np.sqrt(2)*sig.sosfiltfilt(sos, np.abs(data))

    return envelope

def load_data(folder: str = ''):
    raw_file = None
    raw_files = glob.glob(os.path.join(folder, '*.raw'))
    if len(raw_files) >= 1:
        raw_file = raw_files[0]
    else:
        print('No Raw-File found!')
        exit()

    data = open_data(raw_file, 60., 10, channel=-1)
    fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
    sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
    idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
    ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))
    spectra = np.load(os.path.join(folder, 'spec.npy'))
    return data, fund_v, sign_v, idx_v, ident_v, times, spectra


def on_press(event):
    if event.key == 'q':
        plt.close('all')
    if event.key == 'x':
        plt.close('all')
        quit()


def comp_inst_freq(data, samplerate, t, kernal_size=5):
    event_idx = np.arange(len(data) - 1)[(data[:-1] <= 0) & (data[1:] > 0)]
    event_t = t[event_idx]
    a0 = data[event_idx]
    a1 = data[event_idx + 1]

    correction_ratio = a0 * -1 / (a0 * -1 + a1)
    corr_event_t = event_t + correction_ratio * 1 / samplerate
    corr_inst_f = 1 / np.diff(corr_event_t)
    smooth_freq = np.convolve(corr_inst_f, np.ones(kernal_size) / kernal_size, mode='same')

    return smooth_freq, corr_event_t


def add_sim_signal(t, smooth_freq, data, samplerate, df, envelope_cut_off_freq = 100.):
    data_env = envelope(data, samplerate, envelope_cut_off_freq)
    n_data = data / data_env
    n_data = n_data/np.percentile(n_data, 95)

    sim_sin = np.sin(2 * np.pi * t * (np.mean(smooth_freq)+df)) * 0.8
    sim_beat = n_data + sim_sin
    sim_beat_env = envelope(sim_beat, samplerate, envelope_cut_off_freq)

    return n_data, sim_beat, data_env, sim_beat_env


def main(folder):
    start_t = 3 * 60 * 60 + 5 * 60 + 23*4.5
    end_t = 3 * 60 * 60 + 10 * 60

    step_size = 4.5
    overlap = 0.5
    # start_t = 1 * 60 * 60 + 5 * 60
    # end_t = 1 * 60 * 60 + 10 * 60

    data, fund_v, sign_v, idx_v, ident_v, times, spectra = load_data(folder)

    for id in np.unique(ident_v[~np.isnan(ident_v)]):
        for t0 in tqdm(np.arange(start_t, end_t, step_size)):
            t1 = t0 + step_size + overlap

            mean_freq = np.mean(fund_v[(ident_v == id) & (times[idx_v] >= t0) & (times[idx_v] < t1)])

            s = sign_v[(ident_v == id) & (times[idx_v] >= t0) & (times[idx_v] < t1)]
            power_channels = np.argsort(np.sum(s, axis=0))[-3:][::-1]

            spectra, spec_freqs, spec_times = spec(data, data.samplerate, fres=50, overlap_frac=0, t0=t0, t1=t1)
            sf0, sf1, st0, st1 = np.argmax(spec_freqs >= 600), np.argmax(spec_freqs > 1000), \
                                 np.argmax(spec_times >= spec_times[0]), np.argmax(spec_times >= spec_times[-1])

            fig = plt.figure(1, figsize=(30/2.54, 18/2.54))
            move_figure(fig, int(screen_width*0.00), int(screen_height*0.2))
            fig.canvas.mpl_connect('key_press_event', on_press)
            gs = gridspec.GridSpec(4, 3, left=0.05, bottom=0.1, right=0.95, top=0.9)
            ax = [[], [], []]
            for i, ch in enumerate(power_channels):
                t = np.arange(int(t0*data.samplerate), int(t1*data.samplerate+1)) / data.samplerate

                fdata = bandpass_filter(data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch],
                                         data.samplerate, lowf=mean_freq-5, highf=mean_freq+5)

                fdata_env = envelope(fdata, data.samplerate, 100.)

                fdata20 = bandpass_filter(data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch],
                                         data.samplerate, lowf=mean_freq+20-5, highf=mean_freq+20+5)
                fdata20_env = envelope(fdata20, data.samplerate, 100.)

                fdata100 = bandpass_filter(data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch],
                                         data.samplerate, lowf=mean_freq+100-5, highf=mean_freq+100+5)
                fdata100_env = envelope(fdata100, data.samplerate, 100.)

                if i == 0:
                    smooth_freq, corr_event_t = comp_inst_freq(fdata, data.samplerate, t)
                    if True:
                        fig2 = plt.figure(figsize=(16/2.54, 10/2.54))
                        fig2.canvas.mpl_connect('key_press_event', on_press)
                        move_figure(fig2, int(screen_width * 0.7), int(screen_height * 0.00))
                        gs2 = gridspec.GridSpec(2, 1, left=.15, bottom=0.1, right=0.95, top=0.9)
                        ax2 = []
                        ax2.append(fig2.add_subplot(gs2[0, 0]))
                        ax2.append(fig2.add_subplot(gs2[1, 0], sharex=ax2[0]))

                        ax2[0].plot(t-t0, fdata, marker='.')
                        ax2[0].plot(t-t0, fdata_env, lw=2, color='darkorange')
                        ax2[0].plot(corr_event_t-t0, np.zeros(len(corr_event_t)), 'ok')

                        ax2[1].plot(corr_event_t[:-1][2:-3]-t0, smooth_freq[2:-3], marker='.')

                        ax2[0].set_xlim(t[0]-t0, t[-1]-t0)
                        ax2[0].set_ylabel('amplitude [mV]', fontsize=12)
                        ax2[1].set_ylabel('EODf [Hz]', fontsize=12)
                        ax2[1].set_xlabel('time [s]', fontsize=12)
                        fig2.align_ylabels()

                        ax2[0].set_title('Filtered max power electrode (EODf +- 5Hz)')

                    data_oi = fdata
                    # data_oi = data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch]
                    n_data, sim_beat, data_oi_env, sim_beat_env = add_sim_signal(t, smooth_freq, data_oi, data.samplerate, df=-1, envelope_cut_off_freq=50.)

                    if True:
                        fig2 = plt.figure(figsize=(14/2.54, 14/2.54))
                        fig2.canvas.mpl_connect('key_press_event', on_press)
                        move_figure(fig2, int(screen_width * 0.7), int(screen_height * 0.5))
                        gs2 = gridspec.GridSpec(3, 1, left=.25, bottom=0.1, right=0.95, top=0.9, hspace=0.5)
                        ax2 = []
                        ax2.append(fig2.add_subplot(gs2[0, 0]))
                        ax2.append(fig2.add_subplot(gs2[1, 0], sharex=ax2[0]))
                        ax2.append(fig2.add_subplot(gs2[2, 0], sharex=ax2[0]))

                        ax2[0].plot(t-t0, data_oi, marker='.')
                        ax2[0].plot(t-t0, data_oi_env, lw=2, color='darkorange')

                        ax2[1].plot(t-t0, n_data, marker='.')
                        ax2[2].set_ylim(-1.1, 1.1)

                        ax2[2].plot(t-t0, sim_beat, marker='.')
                        ax2[2].plot(t-t0, sim_beat_env, lw=2, color='darkorange')
                        ax2[2].set_ylim(-2.5, 2.5)

                        ax2[0].set_ylabel('amplitude [mV]', fontsize=12)
                        ax2[0].set_title('signal', fontsize=12)
                        ax2[1].set_ylabel('amplitude [a.U.]', fontsize=12)
                        ax2[1].set_title('normed signal', fontsize=12)
                        ax2[2].set_ylabel('amplitude [a.U.]', fontsize=12)
                        ax2[2].set_title('normed signal + sin() * 0.8', fontsize=12)

                        ax2[0].set_xlim(t[0]-t0, t[-1]-t0)
                        fig2.align_ylabels()

                # plotting:
                ax[i].append(fig.add_subplot(gs[0, i]))
                ax[i].append(fig.add_subplot(gs[1, i], sharex=ax[i][0]))
                ax[i].append(fig.add_subplot(gs[2, i], sharex=ax[i][0], sharey=ax[i][1]))
                ax[i].append(fig.add_subplot(gs[3, i], sharex=ax[i][0], sharey=ax[i][1]))

                c = np.random.rand(3)
                t = times[idx_v[ident_v == id]]
                f = fund_v[ident_v == id]

                ax[i][0].imshow(decibel(spectra[ch][sf0:sf1+1, st0:st1+1])[::-1],
                          extent=[spec_times[st0]-t0, spec_times[st1]-t0, spec_freqs[sf0], spec_freqs[sf1]], aspect='auto',
                          vmin=-100, vmax=-50, alpha=0.7, cmap='jet', interpolation=None)
                ax[i][0].plot(t[(t>=spec_times[st0]) & (t<=spec_times[st1])]-t0, f[(t>=spec_times[st0]) & (t<=spec_times[st1])], marker='.', color=c)

                ax[i][1].plot(np.arange(len(fdata))/data.samplerate, fdata)
                ax[i][1].plot(np.arange(len(fdata))/data.samplerate, fdata_env, lw=2, color='darkorange')

                ax[i][2].plot(np.arange(len(fdata))/data.samplerate, fdata20)
                ax[i][2].plot(np.arange(len(fdata)) / data.samplerate, fdata20_env, lw=2, color='darkorange')

                ax[i][3].plot(np.arange(len(fdata))/data.samplerate, fdata100)
                ax[i][3].plot(np.arange(len(fdata)) / data.samplerate, fdata100_env, lw=2, color='darkorange')

                ax[i][0].set_xlim(0, step_size + overlap)
                ax[i][0].set_title(ch)
                fig.suptitle('time: %.1f - %.1f' % (t0, t1))

            plt.show()

    pass

# def spectrogram(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning,
#                 overlap_frac=0.5, pad_to=None, sides='default', scale_by_freq=None, min_nfft=16):
#
#     n_fft = nfft(samplerate, fresolution)
#     noverlap = int(n_fft * overlap_frac)
#
#     spectrum, freqs, time = mlab.specgram(data, NFFT=n_fft, Fs=samplerate, detrend=detrend, window=window,
#                                           noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
#     return spectrum, freqs, time

def spec(data, sr, fres, overlap_frac, t0, t1):

    core_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(core_count // 2)

    func = partial(spectrogram, ratetime=sr, freq_resolution=fres, overlap_frac=overlap_frac)
    a = pool.map(func, [data[int(t0*sr): int(t1*sr)+1, channel] for channel in np.arange(data.channels)])  # ret: spec, freq, time
    spectra = [a[channel][0] for channel in range(len(a))]
    spec_freqs = a[0][1]
    spec_times = a[0][2]
    pool.terminate()

    spec_times += t0

    return spectra, spec_freqs, spec_times


if __name__ == '__main__':
    # main(sys.argv[1])
    main('/home/raab/data/2022_competition/2022-06-02-10_00/')
