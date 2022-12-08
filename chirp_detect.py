import sys
import glob
import os
import multiprocessing
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thunderfish.dataloader import DataLoader as open_data
from thunderfish.powerspectrum import decibel, spectrogram, next_power_of_two, nfft
from functools import partial
from IPython import embed
from tqdm import tqdm

import matplotlib.mlab as mlab

# python3 chirp_detect.py /home/raab/data/2022_competition/2022-06-02-10_00/


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



def cycle_detector(data, fdata, env, samplerate):

    t = np.arange(len(fdata))/samplerate

    n_data = data / envelope(data, samplerate, 100.)

    # event_idx = np.arange(len(data)-1)[(data[:-1] < env[:-1]) & (data[1:] > env[1:])]
    event_idx = np.arange(len(fdata)-1)[(fdata[:-1] <= 0) & (fdata[1:] > 0)]
    event_t = t[event_idx]
    a0 = fdata[event_idx]
    a1 = fdata[event_idx+1]

    t_shift = a0*-1 / (a0*-1 + a1)
    corr_event_t = event_t + t_shift*1/samplerate
    corr_inst_f = 1/np.diff(corr_event_t)
    smooth_freq = np.convolve(corr_inst_f, np.ones(5)/5, mode='same')

    inst_freq = 1/np.diff(t[event_idx])

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(t, fdata, marker='.')
    ax[0].plot(t, env, lw = 2, color='darkorange')
    ax[0].plot(corr_event_t, np.zeros(len(event_idx)), 'ok')

    ax[1].plot(corr_event_t[:-1][2:-3], corr_inst_f[2:-3], marker='.')
    ax[1].plot(corr_event_t[:-1][2:-3], smooth_freq[2:-3], marker='.')

    ax[2].plot(t, data, marker='.')
    ax[2].plot(t, envelope(data, samplerate, 100.), marker='.')

    sim_sin = np.sin(2 * np.pi * t * (np.mean(smooth_freq)-1)) * 0.8
    sim_beat = n_data + sim_sin
    sim_beat_env = envelope(sim_beat, samplerate, 100.)
    ax[3].plot(t, sim_beat, marker='.')
    ax[3].plot(t, sim_beat_env, lw = 2, color='darkorange')
    ax[3].set_ylim(-3, 3)

def on_press(event):
    if event.key == 'q':
        plt.close()
    if event.key == 'x':
        plt.close()
        quit()


def main(folder):
    start_t = 3 * 60 * 60 + 5 * 60 + 23*4.5
    end_t = 3 * 60 * 60 + 10 * 60
    # start_t = 1 * 60 * 60 + 5 * 60
    # end_t = 1 * 60 * 60 + 10 * 60

    data, fund_v, sign_v, idx_v, ident_v, times, spectra = load_data(folder)

    # spectra, spec_freqs, spec_times = spec(data, data.samplerate, fres=50, overlap_frac=0, t0=start_t, t1=end_t)
    # sum_spec = np.sum(spectra, axis=0)
    # sf0, sf1, st0, st1 = np.argmax(spec_freqs >= 600), np.argmax(spec_freqs > 1000), \
    #                      np.argmax(spec_times >= spec_times[0]), np.argmax(spec_times >= spec_times[-1])

    for id in np.unique(ident_v[~np.isnan(ident_v)]):
        for t0 in tqdm(np.arange(start_t, end_t, 4.5)):
            t1 = t0 + 5

            mean_freq = np.mean(fund_v[(ident_v == id) & (times[idx_v] >= t0) & (times[idx_v] < t1)])

            s = sign_v[(ident_v == id) & (times[idx_v] >= t0) & (times[idx_v] < t1)]
            power_channels = np.argsort(np.sum(s, axis=0))[-3:][::-1]

            spectra, spec_freqs, spec_times = spec(data, data.samplerate, fres=50, overlap_frac=0, t0=t0, t1=t1)
            sum_spec = np.sum(spectra, axis=0)
            sf0, sf1, st0, st1 = np.argmax(spec_freqs >= 600), np.argmax(spec_freqs > 1000), \
                                 np.argmax(spec_times >= spec_times[0]), np.argmax(spec_times >= spec_times[-1])
            if True:
                fig = plt.figure(figsize=(40/2.54, 24/2.54))
                fig.canvas.mpl_connect('key_press_event', on_press)
                gs = gridspec.GridSpec(4, 3, left=0.05, bottom=0.1, right=0.95, top=0.95)
                ax = [[], [], []]
                for i in range(3):
                    ax[i].append(fig.add_subplot(gs[0, i]))
                    ax[i].append(fig.add_subplot(gs[1, i], sharex=ax[i][0]))
                    ax[i].append(fig.add_subplot(gs[2, i], sharex=ax[i][0], sharey=ax[i][1]))
                    ax[i].append(fig.add_subplot(gs[3, i], sharex=ax[i][0], sharey=ax[i][1]))

            for i, ch in enumerate(power_channels):

                fdata = bandpass_filter(data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch],
                                         data.samplerate, lowf=mean_freq-5, highf=mean_freq+5)

                fdata_env = envelope(fdata, data.samplerate, 100.)

                cycle_detector(data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch], fdata, fdata_env, data.samplerate)

                fdata20 = bandpass_filter(data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch],
                                         data.samplerate, lowf=mean_freq+20-5, highf=mean_freq+20+5)
                fdata20_env = envelope(fdata20, data.samplerate, 100.)

                fdata100 = bandpass_filter(data[int(t0*data.samplerate):int(t1*data.samplerate+1), ch],
                                         data.samplerate, lowf=mean_freq+100-5, highf=mean_freq+100+5)
                fdata100_env = envelope(fdata100, data.samplerate, 100.)

                ax[i][0].imshow(decibel(spectra[ch][sf0:sf1+1, st0:st1+1])[::-1],
                          extent=[spec_times[st0], spec_times[st1], spec_freqs[sf0], spec_freqs[sf1]], aspect='auto',
                          vmin=-100, vmax=-50, alpha=0.7, cmap='jet', interpolation=None)

                # for id in np.unique(ident_v[~np.isnan(ident_v)]):
                c = np.random.rand(3)
                t = times[idx_v[ident_v == id]]
                f = fund_v[ident_v == id]

                ax[i][0].plot(t[(t>=spec_times[st0]) & (t<=spec_times[st1])], f[(t>=spec_times[st0]) & (t<=spec_times[st1])], marker='.', color=c)
                    # s = sign_v[ident_v == id]
                    # ax[1].plot(t[(t>=spec_times[t0]) & (t<=spec_times[t1])], np.argmax(s, axis=1)[(t>=spec_times[t0]) & (t<=spec_times[t1])], color=c)
                ax[i][1].plot(np.arange(len(fdata))/data.samplerate + t0, fdata)
                ax[i][1].plot(np.arange(len(fdata))/data.samplerate + t0, fdata_env, lw=2, color='darkorange')

                ax[i][2].plot(np.arange(len(fdata))/data.samplerate + t0, fdata20)
                ax[i][2].plot(np.arange(len(fdata)) / data.samplerate + t0, fdata20_env, lw=2, color='darkorange')

                ax[i][3].plot(np.arange(len(fdata))/data.samplerate + t0, fdata100)
                ax[i][3].plot(np.arange(len(fdata)) / data.samplerate + t0, fdata100_env, lw=2, color='darkorange')

                ax[i][0].set_xlim(t0, t1)
                ax[i][0].set_title(ch)
            plt.show()

    pass

def spectrogram(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning,
                overlap_frac=0.5, pad_to=None, sides='default', scale_by_freq=None, min_nfft=16):

    # nfft = 4096
    noverlap = 128
    # nfft, noverlap = nfft_noverlap(fresolution, samplerate, overlap_frac, min_nfft=min_nfft)
    n_fft = nfft(samplerate, fresolution)
    noverlap = int(n_fft * overlap_frac)

    spectrum, freqs, time = mlab.specgram(data, NFFT=n_fft, Fs=samplerate, detrend=detrend, window=window,
                                          noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    return spectrum, freqs, time

def spec(data, sr, fres, overlap_frac, t0, t1):

    core_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(core_count // 2)

    func = partial(spectrogram, samplerate=sr, fresolution=fres, overlap_frac=overlap_frac)
    a = pool.map(func, [data[int(t0*sr): int(t1*sr)+1, channel] for channel in np.arange(data.channels)])  # ret: spec, freq, time
    spectra = [a[channel][0] for channel in range(len(a))]
    spec_freqs = a[0][1]
    spec_times = a[0][2]
    pool.terminate()

    spec_times += t0

    return spectra, spec_freqs, spec_times


if __name__ == '__main__':
    # main(sys.argv[1])
    main('/home/raab/data/2022_tube_competition/2022-06-02-10_00/')
