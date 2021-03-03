import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.version import __version__
from thunderfish.powerspectrum import decibel, next_power_of_two, spectrogram
from thunderfish.dataloader import open_data, fishgrid_grids, fishgrid_spacings
from thunderfish.harmonics import harmonic_groups, fundamental_freqs
from signal_tracker import freq_tracking_v5, plot_tracked_traces, Emit_progress
from thunderfish.eventdetection import hist_threshold

import multiprocessing
from functools import partial


from IPython import embed

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SettingsHarmonicGroup(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cfg = None

        self.verbose=0
        self.low_threshold=0.0
        self.low_threshold_G = 0.0
        self.low_thresh_factor=6.0
        self.high_threshold=0.0
        self.high_threshold_G=0.0
        self.high_thresh_factor=10.0
        self.freq_tol_fac=1.0
        self.mains_freq=50.0
        self.mains_freq_tol=1.0
        self.max_divisor=2
        self.min_group_size=2
        self.max_rel_power_weight=2.0
        self.max_rel_power=0.0

        self.setGeometry(350, 200, 600, 600)
        self.setWindowTitle('Harminic groups settings')

        self.central_widget = QWidget(self)
        self.gridLayout = QGridLayout()

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 1)
        self.gridLayout.setRowStretch(3, 1)
        self.gridLayout.setRowStretch(4, 1)
        self.gridLayout.setRowStretch(5, 1)
        self.gridLayout.setRowStretch(6, 1)
        self.gridLayout.setRowStretch(7, 1)

        self.init_widgets()

        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

        self.write_cfg_dict()

    def init_widgets(self):

        self.verboseW = QLineEdit(str(0), self.central_widget)
        self.verboseL = QLabel('Verbose', self.central_widget)
        self.gridLayout.addWidget(self.verboseW, 0, 0)
        self.gridLayout.addWidget(self.verboseL, 0, 1)

        self.lowTH_W = QLineEdit(str(self.low_threshold), self.central_widget)
        self.lowTH_L = QLabel('low threshold [dB]', self.central_widget)
        self.gridLayout.addWidget(self.lowTH_W, 1, 0)
        self.gridLayout.addWidget(self.lowTH_L, 1, 1)

        self.lowTH_fac_W = QLineEdit(str(self.low_thresh_factor), self.central_widget)
        self.lowTH_fac_L = QLabel('low threshold factor', self.central_widget)
        self.gridLayout.addWidget(self.lowTH_fac_W, 2, 0)
        self.gridLayout.addWidget(self.lowTH_fac_L, 2, 1)

        self.highTH_W = QLineEdit(str(self.high_threshold), self.central_widget)
        self.highTH_L = QLabel('high threshold [dB]', self.central_widget)
        self.gridLayout.addWidget(self.highTH_W, 3, 0)
        self.gridLayout.addWidget(self.highTH_L, 3, 1)

        self.highTH_fac_W = QLineEdit(str(self.high_thresh_factor), self.central_widget)
        self.highTH_fac_L = QLabel('high threshold factor', self.central_widget)
        self.gridLayout.addWidget(self.highTH_fac_W, 4, 0)
        self.gridLayout.addWidget(self.highTH_fac_L, 4, 1)

        self.freq_tol_fac_W = QLineEdit(str(self.freq_tol_fac), self.central_widget)
        self.freq_tol_fac_L = QLabel('freq tollerance factor', self.central_widget)
        self.gridLayout.addWidget(self.freq_tol_fac_W, 5, 0)
        self.gridLayout.addWidget(self.freq_tol_fac_L, 5, 1)

        self.mains_freq_W = QLineEdit(str(self.mains_freq), self.central_widget)
        self.mains_freq_L = QLabel('Main frequencies [Hz]', self.central_widget)
        self.gridLayout.addWidget(self.mains_freq_W, 6, 0)
        self.gridLayout.addWidget(self.mains_freq_L, 6, 1)

        self.mains_freq_tol_W = QLineEdit(str(self.mains_freq_tol), self.central_widget)
        self.mains_freq_tol_L = QLabel('Main frequencies tollerance [Hz]', self.central_widget)
        self.gridLayout.addWidget(self.mains_freq_tol_W, 7, 0)
        self.gridLayout.addWidget(self.mains_freq_tol_L, 7, 1)

        self.max_divisor_W = QLineEdit(str(self.max_divisor), self.central_widget)
        self.max_divisor_L = QLabel('Max divisor', self.central_widget)
        self.gridLayout.addWidget(self.max_divisor_W, 8, 0)
        self.gridLayout.addWidget(self.max_divisor_L, 8, 1)

        self.min_group_size_W = QLineEdit(str(self.min_group_size), self.central_widget)
        self.min_group_size_L = QLabel('min. harmonic group size', self.central_widget)
        self.gridLayout.addWidget(self.min_group_size_W, 9, 0)
        self.gridLayout.addWidget(self.min_group_size_L, 9, 1)

        self.max_rel_power_weight_W = QLineEdit(str(self.max_rel_power_weight), self.central_widget)
        self.max_rel_power_weight_L = QLabel('max. rel. power weight', self.central_widget)
        self.gridLayout.addWidget(self.max_rel_power_weight_W, 10, 0)
        self.gridLayout.addWidget(self.max_rel_power_weight_L, 10, 1)

        self.max_rel_power_W = QLineEdit(str(self.max_rel_power), self.central_widget)
        self.max_rel_power_L = QLabel('max. rel. power', self.central_widget)
        self.gridLayout.addWidget(self.max_rel_power_W, 11, 0)
        self.gridLayout.addWidget(self.max_rel_power_L, 11, 1)

        space = QLabel('', self.central_widget)
        self.gridLayout.addWidget(space, 12, 0)

        Apply = QPushButton('&Apply', self.central_widget)
        Apply.clicked.connect(self.apply_settings)
        self.gridLayout.addWidget(Apply, 13, 1)

        Cancel = QPushButton('&Cancel', self.central_widget)
        Cancel.clicked.connect(self.close)
        self.gridLayout.addWidget(Cancel, 13, 2)

    def apply_settings(self):
        self.verbose = int(self.verboseW.text())
        self.low_threshold = float(self.lowTH_W.text())
        self.low_thresh_factor = float(self.lowTH_fac_W.text())
        self.high_threshold = float(self.highTH_W.text())
        self.high_thresh_factor = float(self.highTH_fac_W.text())
        self.freq_tol_fac = float(self.freq_tol_fac_W.text())
        self.mains_freq = float(self.mains_freq_W.text())
        self.mains_freq_tol = float(self.mains_freq_tol_W.text())
        self.max_divisor = int(self.max_divisor_W.text())
        self.min_group_size = int(self.min_group_size_W.text())

        self.max_rel_power_weight = float(self.max_rel_power_weight_W.text())
        self.max_rel_power = float(self.max_rel_power_W.text())

        self.write_cfg_dict()

    def write_cfg_dict(self):
        self.cfg = {}

        self.cfg.update({'verbose': self.verbose})
        self.cfg.update({'low_thresh_factor': self.low_thresh_factor})
        self.cfg.update({'high_thresh_factor': self.high_thresh_factor})
        self.cfg.update({'freq_tol_fac': self.freq_tol_fac})
        self.cfg.update({'mains_freq': self.mains_freq})
        self.cfg.update({'mains_freq_tol': self.mains_freq_tol})
        self.cfg.update({'max_divisor': self.max_divisor})
        self.cfg.update({'min_group_size': self.min_group_size})
        self.cfg.update({'max_rel_power_weight': self.max_rel_power_weight})
        self.cfg.update({'max_rel_power': self.max_rel_power})


class SettingsSpectrogram(QMainWindow):
    def __init__(self):
        super().__init__()
        # ToDo: get samplerate in here !!!
        self.samplerate = None

        self.start_time = 0
        self.end_time = -1
        self.data_snippet_sec = 15.
        self.data_snippet_idxs = 15 * 20000
        self.fresolution = 1.5
        self.overlap_frac = 0.85
        self.nffts_per_psd = 1


        self.setGeometry(350, 200, 600, 600)
        self.setWindowTitle('Harminic groups settings')

        self.central_widget = QWidget(self)
        self.gridLayout = QGridLayout()

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 1)
        self.gridLayout.setRowStretch(3, 1)
        self.gridLayout.setRowStretch(4, 1)
        self.gridLayout.setRowStretch(5, 1)
        self.gridLayout.setRowStretch(6, 1)
        self.gridLayout.setRowStretch(7, 1)

        self.init_widgets()

        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

    def init_widgets(self):
        self.StartTime = QLineEdit('%.2f' % (self.start_time / 60), self.central_widget)
        self.gridLayout.addWidget(self.StartTime, 0, 0)
        t0 = QLabel('start time [min]', self.central_widget)
        self.gridLayout.addWidget(t0, 0, 1)

        self.EndTime = QLineEdit('%.2f' % (self.end_time / 60), self.central_widget)
        self.gridLayout.addWidget(self.EndTime, 1, 0)
        t1 = QLabel('end time [min]', self.central_widget)
        self.gridLayout.addWidget(t1, 1, 1)

        self.SnippetSize = QLineEdit(str(self.data_snippet_sec), self.central_widget)
        self.gridLayout.addWidget(self.SnippetSize, 2, 0)
        snip_size = QLabel('data snippet size [sec]', self.central_widget)
        self.gridLayout.addWidget(snip_size, 2, 1)

        self.FreqResolution = QLineEdit(str(self.fresolution), self.central_widget)
        self.gridLayout.addWidget(self.FreqResolution, 3, 0)
        freqres = QLabel('frequency resolution [Hz]', self.central_widget)
        self.gridLayout.addWidget(freqres, 3, 1)

        self.Overlap = QLineEdit(str(self.overlap_frac), self.central_widget)
        self.gridLayout.addWidget(self.Overlap, 4, 0)
        overlap = QLabel('overlap fraction', self.central_widget)
        self.gridLayout.addWidget(overlap, 4, 1)

        self.NfftPerPsd = QLineEdit(str(self.nffts_per_psd), self.central_widget)
        self.gridLayout.addWidget(self.NfftPerPsd, 5, 0)
        overlap = QLabel('nffts per PSD [n]', self.central_widget)
        self.gridLayout.addWidget(overlap, 5, 1)

        if self.samplerate:
            self.real_nfft = QLineEdit('%.0f' % next_power_of_two(self.samplerate / self.fresolution),
                                       self.central_widget)
            self.temp_res = QLineEdit(
                '%.3f' % (next_power_of_two(self.samplerate / self.fresolution) * (1. - self.overlap_frac)),
                self.central_widget)
            print('%.3f' % (
            next_power_of_two(self.samplerate / self.fresolution) * (1. - self.overlap_frac) / self.samplerate))
        else:
            self.real_nfft = QLineEdit('~', self.central_widget)
            self.temp_res = QLineEdit('~', self.central_widget)
        self.real_nfft.setReadOnly(True)
        self.temp_res.setReadOnly(True)

        self.real_nfftL = QLabel('real nfft [n]', self.central_widget)
        self.temp_resL = QLabel('temp. resolution [s]', self.central_widget)

        self.gridLayout.addWidget(self.real_nfft, 6, 1)
        self.gridLayout.addWidget(self.real_nfftL, 6, 2)

        self.gridLayout.addWidget(self.temp_res, 7, 1)
        self.gridLayout.addWidget(self.temp_resL, 7, 2)

        space = QLabel('', self.central_widget)
        self.gridLayout.addWidget(space, 8, 0)

        Apply = QPushButton('&Apply', self.central_widget)
        Apply.clicked.connect(self.apply_settings)
        self.gridLayout.addWidget(Apply, 9, 1)

        Cancel = QPushButton('&Cancel', self.central_widget)
        Cancel.clicked.connect(self.close)
        self.gridLayout.addWidget(Cancel, 9, 2)

    def apply_settings(self):
        self.start_time = float(self.StartTime.text()) * 60
        self.end_time = float(self.EndTime.text()) * 60
        self.data_snippet_sec = float(self.SnippetSize.text())
        self.fresolution = float(self.FreqResolution.text())
        self.overlap_frac = float(self.Overlap.text())
        self.nffts_per_psd = int(self.NfftPerPsd.text())

        self.real_nfft.setText('%.0f' % next_power_of_two(20000. / self.fresolution))
        self.temp_res.setText('%.3f' % (next_power_of_two(20000. / self.fresolution) * (1. - self.overlap_frac) / 20000.))
        if self.samplerate:
            self.data_snippet_idxs = int(self.data_snippet_sec * self.samplerate)
            self.real_nfft.setText('%.0f' % next_power_of_two(self.samplerate / self.fresolution))
            self.temp_res.setText('%.3f' % (next_power_of_two(self.samplerate / self.fresolution) * (1. - self.overlap_frac) / self.samplerate))



class EOD_extraxt(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(float)
    # tracking_progress = pyqtSignal(float)
    return_spec = pyqtSignal()
    return_EODf = pyqtSignal()

    def __init__(self):
        super(QThread, self).__init__()

        self.HGSettings = SettingsHarmonicGroup()
        self.SpecSettings = SettingsSpectrogram()

        self.single_chennel_analysis=False
        # self.TrackingProgress = Emit_progress()
        # self.TrackingProgress.progress.connect(self.emit_tracking_progress)
        self.params()

    def params(self):

        self.current_tast = None
        self.samplerate = None
        self.channels = None
        self.channel_list = []
        self.data = None

        self.fundamentals_SCH = []
        self.signatures_SCH = []
        self.fundamentals = []
        self.signatures = []
        self.times = []

        self.tmp_spectra_SCH = None
        self.tmp_spectra = None
        self.tmp_times = None

        self.life_plotting = False

        self.folder = None
        self.all_fund_v = []
        self.all_ident_v = []
        self.all_idx_v = []
        self.all_original_sign_v = []

    def run(self):
        self.SpecSettings.apply_settings()
        self.HGSettings.apply_settings()
        if self.current_tast == 'fill_spec':
            self.fill_spec()
        else:
            self.snippet_spectrogram()

    def fill_spec(self):
        start_idx = int(self.SpecSettings.start_time * self.samplerate)
        if self.SpecSettings.end_time < 0.0:
            end_time = len(self.data) / self.samplerate
            end_idx = int(len(self.data) - 1)

            self.SpecSettings.end_time = end_time
        else:
            end_idx = int(self.SpecSettings.end_time * self.samplerate)
            if end_idx >= int(len(self.data) - 1):
                end_idx = int(len(self.data) - 1)

        last_run = False
        get_spec_plot_matrix = False

        p0 = start_idx
        pn = end_idx

        first_run = True
        pre_save_spectra = np.array([])

        while start_idx <= end_idx:
            self.progress.emit((start_idx - p0) / (end_idx - p0) * 100)
            # self.progress.setValue((start_idx - p0) / (end_idx - p0) * 100)

            if start_idx >= end_idx - self.SpecSettings.data_snippet_idxs:
                last_run = True

            core_count = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(core_count - 1)
            nfft = next_power_of_two(self.samplerate / self.SpecSettings.fresolution)

            func = partial(spectrogram, ratetime=self.samplerate, freq_resolution=self.SpecSettings.fresolution, overlap_frac=self.SpecSettings.overlap_frac)

            if len(np.shape(self.data)) == 1:
                a = pool.map(func, [self.data[start_idx: start_idx + self.SpecSettings.data_snippet_idxs]])
            else:
                a = pool.map(func, [self.data[start_idx: start_idx + self.SpecSettings.data_snippet_idxs, channel] for channel in
                                    self.channel_list])  # ret: spec, freq, time

            self.spectra = [a[channel][0] for channel in range(len(a))]
            self.spec_freqs = a[0][1]
            self.spec_times = a[0][2]
            pool.terminate()

            self.comb_spectra = np.sum(self.spectra, axis=0)
            self.tmp_times = self.spec_times + (start_idx / self.samplerate)

            ############################################
            fill_spec_str = os.path.join('/home/raab/analysis/fine_specs', os.path.split(self.folder)[-1], 'fill_spec.npy')
            if first_run:
                first_run = False
                if not os.path.exists(os.path.join('/home/raab/analysis/fine_specs', os.path.split(self.folder)[-1])):
                    os.mkdir(os.path.join('/home/raab/analysis/fine_specs', os.path.split(self.folder)[-1]))
                fill_spec = np.memmap(fill_spec_str, dtype='float', mode='w+',
                                      shape=(len(self.comb_spectra), len(self.tmp_times)), order='F')

                fill_spec[:, :] = self.comb_spectra
            else:
                if len(pre_save_spectra) == 0:
                    pre_save_spectra = self.comb_spectra
                else:
                    pre_save_spectra = np.append(pre_save_spectra, self.comb_spectra, axis=1)
                if np.shape(pre_save_spectra)[1] >= 500:
                    old_len = np.shape(fill_spec)[1]
                    fill_spec = np.memmap(fill_spec_str, dtype='float', mode='r+', shape=(
                    np.shape(pre_save_spectra)[0], np.shape(pre_save_spectra)[1] + old_len), order='F')
                    fill_spec[:, old_len:] = pre_save_spectra
                    pre_save_spectra = np.array([])

            non_overlapping_idx = (1 - self.SpecSettings.overlap_frac) * nfft
            start_idx += int(len(self.spec_times) * non_overlapping_idx)
            self.times = np.concatenate((self.times, self.tmp_times))

            if start_idx >= end_idx or last_run:
                break

        file_str = os.path.split(self.folder)[-1]
        np.save(os.path.join('/home/raab/analysis/fine_specs', file_str, 'fill_spec_shape.npy'), np.array(np.shape(fill_spec)))
        np.save(os.path.join('/home/raab/analysis/fine_specs', file_str, 'fill_times.npy'), self.times)
        np.save(os.path.join('/home/raab/analysis/fine_specs', file_str, 'fill_freqs.npy'), self.spec_freqs)

        print('')
        print('###   ###')
        print('')
        print('fill spec completed')
        print('')
        print('###   ###')
        print('')

        self.quit()

    def snippet_spectrogram(self):
        start_idx = int(self.SpecSettings.start_time * self.samplerate)
        if self.SpecSettings.end_time < 0.0:
            end_time = len(self.data) / self.samplerate
            end_idx = int(len(self.data) - 1)

            self.SpecSettings.end_time = end_time
        else:
            end_idx = int(self.SpecSettings.end_time * self.samplerate)
            if end_idx >= int(len(self.data) - 1):
                end_idx = int(len(self.data) - 1)

        last_run = False
        get_spec_plot_matrix = False

        p0 = start_idx
        pn = end_idx

        for ch in self.channel_list:
            self.fundamentals_SCH.append([])
            self.signatures_SCH.append([])

        while start_idx <= end_idx:
            self.progress.emit((start_idx - p0) / (end_idx - p0) * 100)
            # self.progress.setValue((start_idx - p0) / (end_idx - p0) * 100)

            if start_idx >= end_idx - self.SpecSettings.data_snippet_idxs:
                last_run = True

            core_count = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(core_count - 1)
            nfft = next_power_of_two(self.samplerate / self.SpecSettings.fresolution)

            func = partial(spectrogram, ratetime=self.samplerate, freq_resolution=self.SpecSettings.fresolution, overlap_frac=self.SpecSettings.overlap_frac)

            if len(np.shape(self.data)) == 1:
                a = pool.map(func, [self.data[start_idx: start_idx + self.SpecSettings.data_snippet_idxs]])  # ret: spec, freq, time
            else:
                a = pool.map(func, [self.data[start_idx: start_idx + self.SpecSettings.data_snippet_idxs, channel] for channel in
                                    self.channel_list])  # ret: spec, freq, time

            self.spectra = [a[channel][0] for channel in range(len(a))]
            self.spec_freqs = a[0][1]
            self.spec_times = a[0][2]
            pool.terminate()

            self.comb_spectra = np.sum(self.spectra, axis=0)
            self.tmp_times = self.spec_times + (start_idx / self.samplerate)

            comp_max_freq = 2000
            comp_min_freq = 0
            create_plotable_spectrogram = True

            if create_plotable_spectrogram:
                plot_freqs = self.spec_freqs[self.spec_freqs < comp_max_freq]
                plot_spectra = np.sum(self.spectra, axis=0)[self.spec_freqs < comp_max_freq]

                if self.life_plotting:
                    self.spec_ret = plot_spectra[(plot_freqs < 1200) & (plot_freqs > 400)]
                    self.freq_ret = plot_freqs[(plot_freqs < 1200) & (plot_freqs > 400)]
                    self.times_ret = self.tmp_times
                    self.return_spec.emit()

                # if not checked_xy_borders:
                if not get_spec_plot_matrix:
                    fig_xspan = 20.
                    fig_yspan = 12.
                    fig_dpi = 80.
                    no_x = fig_xspan * fig_dpi
                    no_y = fig_yspan * fig_dpi

                    min_x = self.SpecSettings.start_time
                    max_x = self.SpecSettings.end_time

                    min_y = comp_min_freq
                    max_y = comp_max_freq

                    x_borders = np.linspace(min_x, max_x, int(no_x * 2))
                    y_borders = np.linspace(min_y, max_y, int(no_y * 2))

                    self.tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))
                    self.tmp_spectra_SCH = np.array([np.zeros((len(y_borders) - 1, len(x_borders) - 1)) for ch in self.channel_list])

                    recreate_matrix = False
                    if (self.tmp_times[1] - self.tmp_times[0]) > (x_borders[1] - x_borders[0]):
                        try:
                            print('da')
                            x_borders = np.linspace(min_x, max_x, int((max_x - min_x) // (self.tmp_times[1] - self.tmp_times[0]) + 1))
                        except:
                            embed()
                            quit()
                        recreate_matrix = True
                    if (self.spec_freqs[1] - self.spec_freqs[0]) > (y_borders[1] - y_borders[0]):
                        recreate_matrix = True
                        y_borders = np.linspace(min_y, max_y, (max_y - min_y) // (self.spec_freqs[1] - self.spec_freqs[0]) + 1)
                    if recreate_matrix:
                        self.tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))
                        self.tmp_spectra_SCH = np.array([np.zeros((len(y_borders) - 1, len(x_borders) - 1)) for ch in self.channel_list])

                    get_spec_plot_matrix = True
                    # checked_xy_borders = True

                for i in range(len(y_borders) - 1):
                    # print(i/len(y_borders))
                    for j in range(len(x_borders) - 1):
                        if x_borders[j] > self.tmp_times[-1]:
                            break
                        if x_borders[j + 1] < self.tmp_times[0]:
                            continue

                        t_mask = np.arange(len(self.tmp_times))[(self.tmp_times >= x_borders[j]) & (self.tmp_times < x_borders[j + 1])]
                        f_mask = np.arange(len(plot_spectra))[(plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i + 1])]

                        if len(t_mask) == 0 or len(f_mask) == 0:
                            continue
                        # print('yay')
                        self.tmp_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])
                        for ch in self.channel_list:
                            self.tmp_spectra_SCH[ch, i, j] = np.max(self.spectra[ch][f_mask[:, None], t_mask])

            # if self.CBgroup_analysis.isChecked():
            if self.single_chennel_analysis == True:
                # self.power = self.spectra
                ####

                for ch in self.channel_list:
                    # self.fundamentals_SCH.append([])
                    # self.signatures_SCH.append([])
                    # ToDo: error here !!!
                    self.power = [np.array([]) for i in range(len(self.spec_times))]

                    for t in range(len(self.spec_times)):
                        self.power[t] = np.mean(self.spectra[ch][:, t:t + 1], axis=1)
                        # self.power[t] = np.mean(self.comb_spectra[:, t:t + 1], axis=1)
                    self.extract_fundamentals_and_signatures(channel=ch)
                ####
            else:
                self.power = [np.array([]) for i in range(len(self.spec_times))]
                for t in range(len(self.spec_times)):
                    self.power[t] = np.mean(self.comb_spectra[:, t:t + 1], axis=1)
                self.extract_fundamentals_and_signatures()

            # if self.single_chennel_analysis == True:
            #     for i in range(len(a)):
            #         self.extract_fundamentals_and_signatures(channel=i)



            non_overlapping_idx = (1 - self.SpecSettings.overlap_frac) * nfft
            start_idx += int((len(self.spec_times) - self.SpecSettings.nffts_per_psd + 1) * non_overlapping_idx)
            self.times = np.concatenate((self.times, self.tmp_times))

            # print('check 4')
            if start_idx >= end_idx or last_run:
                # self.progress.setValue(100)
                self.progress.emit(100)
                # print('done')

                print(np.shape(self.data))
                print(self.channels)
                if self.single_chennel_analysis == True:
                    self.all_fund_v = []
                    self.all_ident_v = []
                    self.all_idx_v = []
                    self.all_sign_v = []
                    self.all_a_error_distribution = []
                    self.all_f_error_distribution = []
                    self.all_idx_of_origin_v = []
                    self.all_original_sign_v = []

                    for it in range(len(self.fundamentals_SCH)):
                        print('### tracking Ch: %.0f of %.0f' % (it, len(self.fundamentals_SCH)) )

                        self.fund_v, self.ident_v, self.idx_v, self.sign_v, self.a_error_distribution, \
                        self.f_error_distribution, self.idx_of_origin_v, self.original_sign_v = \
                            freq_tracking_v5(self.fundamentals_SCH[it], self.signatures_SCH[it], self.times,
                                             freq_tolerance=self.HGSettings.freq_tol_fac, n_channels=self.channels)

                        self.all_fund_v.append(self.fund_v)
                        self.all_ident_v.append(self.ident_v)
                        self.all_idx_v.append(self.idx_v)
                        self.all_sign_v.append(self.sign_v)
                        self.all_a_error_distribution.append(self.a_error_distribution)
                        self.all_f_error_distribution.append(self.f_error_distribution)
                        self.all_idx_of_origin_v.append(self.idx_of_origin_v)
                        self.all_original_sign_v.append(self.original_sign_v)
                    # embed()
                    # quit()

                else:
                    self.fund_v, self.ident_v, self.idx_v, self.sign_v, self.a_error_distribution, \
                    self.f_error_distribution, self.idx_of_origin_v, self.original_sign_v = \
                        freq_tracking_v5(self.fundamentals, self.signatures, self.times, freq_tolerance=self.HGSettings.freq_tol_fac, n_channels=self.channels)

                self.finished.emit()
                break

    def extract_fundamentals_and_signatures(self, channel = None):
        core_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(core_count - 1)

        if channel == None:
            func = partial(harmonic_groups, self.spec_freqs, low_threshold = self.HGSettings.low_threshold_G,
                           high_threshold = self.HGSettings.high_threshold_G, min_freq = 400, max_freq=2000, **self.HGSettings.cfg)
            a = pool.map(func, self.power)

        else:
            func = partial(harmonic_groups, self.spec_freqs, low_threshold = self.HGSettings.low_threshold,
                           high_threshold = self.HGSettings.high_threshold, min_freq = 400, max_freq=2000, **self.HGSettings.cfg)
            a = pool.map(func, self.power)

        # print(a[0][5], a[0][6])
        if channel == None:
            if self.HGSettings.low_threshold_G <= 0 or self.HGSettings.high_threshold_G <= 0:
                self.HGSettings.low_threshold_G = a[0][5]
                self.HGSettings.high_threshold_G = a[0][6]
        else:
            if self.HGSettings.low_threshold <= 0 or self.HGSettings.high_threshold <= 0:
                self.HGSettings.low_threshold = a[0][5]
                self.HGSettings.high_threshold = a[0][6]

        log_spectra = decibel(np.array(self.spectra))

        if self.life_plotting:
            self.EODf_ret = []

        for p in range(len(self.power)):
            tmp_fundamentals = fundamental_freqs(a[p][0])

            if self.life_plotting:
                self.EODf_ret.append(tmp_fundamentals)

            if channel != None:
                self.fundamentals_SCH[channel].append(tmp_fundamentals)
            else:
                self.fundamentals.append(tmp_fundamentals)

            if len(tmp_fundamentals) >= 1:
                f_idx = np.array([np.argmin(np.abs(self.spec_freqs - f)) for f in tmp_fundamentals])
                tmp_signatures = log_spectra[:, np.array(f_idx), p].transpose()
            else:
                tmp_signatures = np.array([])


            if channel != None:
                self.signatures_SCH[channel].append(tmp_signatures)
            else:
                self.signatures.append(tmp_signatures)

        if self.life_plotting:
            self.return_EODf.emit()

        pool.terminate()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.EodEctractThread = EOD_extraxt()
        self.EodEctractThread.finished.connect(self.thread_finished)
        self.EodEctractThread.progress.connect(self.show_progress)
        self.EodEctractThread.return_spec.connect(self.life_spec)
        self.EodEctractThread.return_EODf.connect(self.life_eodf)
        #
        # self.EodEctractThread.tracking_progress.connect(self.show_progress)

        self.life_plot = LifePlot()

        # self.test_thread = Test_thread()
        # self.test_thread.finished.connect(self.thread_finished)
        # self.test_thread.progress.connect(self.show_progress)

        self.setGeometry(300, 150, 600, 300)  # set window proportion
        self.setWindowTitle('Fish Tracking')  # set window title

        self.central_widget = QWidget(self)
        self.gridLayout = QGridLayout()

        self.params()
        self.create_widgets()

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 1)
        self.gridLayout.setRowStretch(3, 1)

        self.gridLayout.addWidget(self.open_B, 0, 0)
        self.gridLayout.addWidget(self.open_fileL, 0, 1)
        self.gridLayout.addWidget(self.save_B, 0, 2)
        self.gridLayout.addWidget(self.auto_save_cb, 1, 2)
        self.gridLayout.addWidget(self.single_trace_cb, 1, 0)
        self.gridLayout.addWidget(self.HG_B, 3, 0)
        self.gridLayout.addWidget(self.fill_spec_B, 3, 1)
        self.gridLayout.addWidget(self.Spec_B, 4, 0)
        self.gridLayout.addWidget(self.run_B, 3, 2)
        self.gridLayout.addWidget(self.life_plot_B, 4, 2)
        self.gridLayout.addWidget(self.progress, 5, 0, 1, 3)

        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

        self.filenames = []
        self.total_file_count = 0
        self.finisched_count = 0

    def params(self):
        self.filename = None
        self.folder = None
        self.rec_datetime = None
        self.data = None
        self.samplerate = None
        self.channels = None
        self.elecs_y, self.elecs_x = None, None
        self.elecs_y_spacing, self.elecs_x_spacing = None, None

        self.fund_v = None
        self.ident_v = None
        self.idx_v = None
        self.original_sign_v = None

        self.start_time = None
        self.end_time = None

    def create_widgets(self):
        self.open_B = QPushButton('open', self.central_widget)
        self.open_B.clicked.connect(self.open)

        self.open_fileL = QLabel('N.a.', self.central_widget)

        self.HG_B = QPushButton('Harmonic groups', self.central_widget)
        self.HG_B.clicked.connect(self.HG_main)

        self.Spec_B = QPushButton('Spectrogram', self.central_widget)
        self.Spec_B.clicked.connect(self.Spec_main)

        self.progress = QProgressBar(self)

        self.run_B = QPushButton('Run', self.central_widget)
        self.run_B.clicked.connect(self.run_main)

        self.fill_spec_B = QPushButton('Calc. fine spec', self.central_widget)
        self.fill_spec_B.clicked.connect(self.run_fill_spec)

        self.save_B = QPushButton('Save', self.central_widget)
        self.save_B.clicked.connect(self.save)
        self.save_B.setEnabled(False)

        self.auto_save_cb = QCheckBox('Auto Save', self.central_widget)
        self.single_trace_cb = QCheckBox('Single Trace', self.central_widget)

        self.life_plot_B = QPushButton('Life Plot', self.central_widget)
        self.life_plot_B.clicked.connect(self.life_dialog)

    def life_dialog(self):
        if self.life_plot.running == False:
            self.life_plot.show()
            self.EodEctractThread.life_plotting = True
            self.life_plot.running = True
        else:
            self.life_plot.close()
            self.EodEctractThread.life_plotting = False
            self.life_plot.running = False

    def life_spec(self):
        life_spec = np.copy(self.EodEctractThread.spec_ret)
        life_freq = np.copy(self.EodEctractThread.freq_ret)
        life_times = np.copy(self.EodEctractThread.tmp_times)

        self.life_plot.update.spec = life_spec
        self.life_plot.update.freq = life_freq
        self.life_plot.update.times = life_times

        # self.life_plot.show()
        self.life_plot.update.start()

    def life_eodf(self):
        life_eodf = np.copy(self.EodEctractThread.EODf_ret)
        life_times = np.copy(self.EodEctractThread.tmp_times)

        self.life_plot.update_eodf.eodf = life_eodf
        self.life_plot.update_eodf.times = life_times

        self.life_plot.update_eodf.start()

    def show_progress(self, e):
        self.progress.setValue(e)

    @pyqtSlot()
    def thread_finished(self):
        # self.test_thread.wait()
        self.finisched_count += 1
        self.open_B.setText('open (%.0f/%.0f)' % (self.finisched_count, self.total_file_count))

        self.EodEctractThread.wait()
        self.HG_B.setEnabled(True)
        self.Spec_B.setEnabled(True)

        self.fund_v = self.EodEctractThread.fund_v
        self.ident_v = self.EodEctractThread.ident_v
        self.idx_v = self.EodEctractThread.idx_v
        self.original_sign_v = self.EodEctractThread.original_sign_v

        self.all_fund_v = self.EodEctractThread.all_fund_v
        self.all_ident_v = self.EodEctractThread.all_ident_v
        self.all_idx_v = self.EodEctractThread.all_idx_v
        self.all_original_sign_v = self.EodEctractThread.all_original_sign_v
        self.tmp_spectra_SCH = self.EodEctractThread.tmp_spectra_SCH

        self.times = self.EodEctractThread.times
        self.tmp_spectra = self.EodEctractThread.tmp_spectra
        self.save_B.setEnabled(True)

        if self.auto_save_cb.isChecked():
            self.save()

        self.filenames.pop(0)
        if len(self.filenames) == 0:
            print('all files analysed')
            self.close()
            quit()
        else:
            self.params()
            self.EodEctractThread.params()
            # self.filename, ok = self.filenames[0]
            self.run_main()

    def run_fill_spec(self):
        self.EodEctractThread.current_tast = 'fill_spec'
        self.run_main()

    @pyqtSlot()
    def run_main(self):
        def get_datetime(folder):
            rec_year, rec_month, rec_day, rec_time = \
                os.path.split(os.path.split(folder)[-1])[-1].split('-')
            rec_year = int(rec_year)
            rec_month = int(rec_month)
            rec_day = int(rec_day)
            try:
                rec_time = [int(rec_time.split('_')[0]), int(rec_time.split('_')[1]), 0]
            except:
                rec_time = [int(rec_time.split(':')[0]), int(rec_time.split(':')[1]), 0]

            rec_datetime = datetime.datetime(year=rec_year, month=rec_month, day=rec_day, hour=rec_time[0],
                                             minute=rec_time[1], second=rec_time[2])


            return rec_datetime

        self.filename, ok = self.filenames[0]

        if ok:
            self.open_fileL.setText(os.path.join('...', os.path.split(os.path.split(self.filename)[0])[-1]))

            self.folder = os.path.split(self.filename)[0]
            self.EodEctractThread.folder = self.folder
            self.rec_datetime = get_datetime(self.folder)

            self.data = open_data(self.filename, -1, 60.0, 10.0)
            self.EodEctractThread.data = self.data

            self.samplerate= self.data.samplerate
            self.EodEctractThread.samplerate = self.samplerate
            self.EodEctractThread.SpecSettings.samplerate = self.samplerate

            # ToDo: delete this after again... error analysis
            self.channels = self.data.channels - 1
            # self.channels = self.data.channels
            self.EodEctractThread.channels = self.data.channels - 1
            self.EodEctractThread.channel_list = np.arange(self.data.channels)

            self.open_fileL.setText(os.path.join('...', os.path.split(os.path.split(self.filename)[0])[-1]))

            if os.path.exists(os.path.join(os.path.split(self.filename)[0], 'fishgrid.cfg')):
                self.elecs_y, self.elecs_x = fishgrid_grids(self.filename)[0]
                self.elecs_y_spacing, self.elecs_x_spacing = fishgrid_spacings(self.filename)[0]

            self.HG_B.setEnabled(False)
            self.Spec_B.setEnabled(False)

            # self.test_thread.start()
            if self.single_trace_cb.isChecked():
                self.EodEctractThread.single_chennel_analysis = True
            self.EodEctractThread.start()
        else:
            print('error in data')
            self.filenames.pop(0)

            if len(self.filenames) == 0:
                quit()
            else:
                self.params()
                self.run_main()

    @pyqtSlot()
    def HG_main(self):
        self.EodEctractThread.HGSettings.show()

    @pyqtSlot()
    def Spec_main(self):
        self.EodEctractThread.SpecSettings.show()

    @pyqtSlot()
    def open(self):
        def get_datetime(folder):
            rec_year, rec_month, rec_day, rec_time = \
                os.path.split(os.path.split(folder)[-1])[-1].split('-')
            rec_year = int(rec_year)
            rec_month = int(rec_month)
            rec_day = int(rec_day)
            try:
                rec_time = [int(rec_time.split('_')[0]), int(rec_time.split('_')[1]), 0]
            except:
                rec_time = [int(rec_time.split(':')[0]), int(rec_time.split(':')[1]), 0]

            rec_datetime = datetime.datetime(year=rec_year, month=rec_month, day=rec_day, hour=rec_time[0],
                                             minute=rec_time[1], second=rec_time[2])


            return rec_datetime

        fd = QFileDialog()
        self.filename, ok = fd.getOpenFileName(self, 'Open File', '/', 'Select Raw-File (*.raw)')

        self.filenames.append((self.filename, ok))
        self.total_file_count += 1
        self.open_B.setText('open (%.0f/%.0f)' % (self.finisched_count, self.total_file_count))

        # if os.path.exists('/home/raab/data/'):
        #     self.filename, ok = fd.getOpenFileName(self, 'Open File', '/home/raab/data/', 'Select Raw-File (*.raw)')
        # else:
        #     self.filename, ok = fd.getOpenFileName(self, 'Open File', '/home/', 'Select Raw-File (*.raw)')

        # if ok:
        #     self.folder = os.path.split(self.filename)[0]
        #     self.rec_datetime = get_datetime(self.folder)
        #
        #     self.data = open_data(self.filename, -1, 60.0, 10.0)
        #     self.EodEctractThread.data = self.data
        #
        #     self.samplerate= self.data.samplerate
        #     self.EodEctractThread.samplerate = self.samplerate
        #     self.EodEctractThread.SpecSettings.samplerate = self.samplerate
        #
        #     self.channels = self.data.channels
        #     self.EodEctractThread.channels = self.data.channels
        #     self.EodEctractThread.channel_list = np.arange(self.channels)
        #
        #     self.open_fileL.setText(os.path.join('...', os.path.split(os.path.split(self.filename)[0])[-1]))
        #
        #     if os.path.exists(os.path.join(os.path.split(self.filename)[0], 'fishgrid.cfg')):
        #         self.elecs_y, self.elecs_x = fishgrid_grids(self.filename)[0]
        #         self.elecs_y_spacing, self.elecs_x_spacing = fishgrid_spacings(self.filename)[0]

    @pyqtSlot()
    def save(self):
        folder = self.folder
        if self.single_trace_cb.isChecked():
            try:
                np.save(os.path.join(folder, 'all_fund_v.npy'), self.all_fund_v)
                np.save(os.path.join(folder, 'all_sign_v.npy'), self.all_original_sign_v)
                np.save(os.path.join(folder, 'all_idx_v.npy'), self.all_idx_v)
                np.save(os.path.join(folder, 'all_ident_v.npy'), self.all_ident_v)

                np.save(os.path.join(folder, 'all_times.npy'), self.times)
                np.save(os.path.join(folder, 'all_spec.npy'), self.tmp_spectra_SCH)

                np.save(os.path.join(folder, 'meta.npy'), np.array([self.EodEctractThread.SpecSettings.start_time,
                                                                    self.EodEctractThread.SpecSettings.end_time]))
            except:
                print('alternative save folder: /home/raab/analysis/<>')
                folder = os.path.join('/home/raab/analysis', os.path.split(self.folder)[-1])
                if not os.path.exists(folder):
                    os.mkdir(folder)
                np.save(os.path.join(folder, 'all_fund_v.npy'), self.all_fund_v)
                np.save(os.path.join(folder, 'all_sign_v.npy'), self.all_original_sign_v)
                np.save(os.path.join(folder, 'all_idx_v.npy'), self.all_idx_v)
                np.save(os.path.join(folder, 'all_ident_v.npy'), self.all_ident_v)

                np.save(os.path.join(folder, 'all_times.npy'), self.times)
                np.save(os.path.join(folder, 'all_spec.npy'), self.tmp_spectra_SCH)

                np.save(os.path.join(folder, 'meta.npy'), np.array([self.EodEctractThread.SpecSettings.start_time,
                                                                    self.EodEctractThread.SpecSettings.end_time]))

        else:
            try:
                np.save(os.path.join(folder, 'fund_v.npy'), self.fund_v)
                np.save(os.path.join(folder, 'sign_v.npy'), self.original_sign_v)
                np.save(os.path.join(folder, 'idx_v.npy'), self.idx_v)
                np.save(os.path.join(folder, 'ident_v.npy'), self.ident_v)
                np.save(os.path.join(folder, 'times.npy'), self.times)
                np.save(os.path.join(folder, 'spec.npy'), self.tmp_spectra)

                np.save(os.path.join(folder, 'meta.npy'), np.array([self.EodEctractThread.SpecSettings.start_time,
                                                                    self.EodEctractThread.SpecSettings.end_time]))
            except:
                print('alternative save folder: /home/raab/analysis/<>')
                folder = os.path.join('/home/raab/analysis', os.path.split(self.folder)[-1])
                if not os.path.exists(folder):
                    os.mkdir(folder)
                np.save(os.path.join(folder, 'fund_v.npy'), self.fund_v)
                np.save(os.path.join(folder, 'sign_v.npy'), self.original_sign_v)
                np.save(os.path.join(folder, 'idx_v.npy'), self.idx_v)
                np.save(os.path.join(folder, 'ident_v.npy'), self.ident_v)
                np.save(os.path.join(folder, 'times.npy'), self.times)
                np.save(os.path.join(folder, 'spec.npy'), self.tmp_spectra)

                np.save(os.path.join(folder, 'meta.npy'), np.array([self.EodEctractThread.SpecSettings.start_time,
                                                                    self.EodEctractThread.SpecSettings.end_time]))


class LifeSpecUpdate(QThread):
    done = pyqtSignal()

    def __init__(self):
        super(QThread, self).__init__()
        self.figure = None
        self.canvas = None
        self.ax = None

        self.handle = None

        self.spec = None
        self.freq = None
        self.times = None

    def run(self):
        # print('dada')
        vmax = -50
        vmin = -100

        dt = self.times[1] - self.times[0]

        if self.handle != None:
            self.handle.remove()
            self.handle = None

        if self.handle == None:
            self.handle = self.ax.imshow(decibel(self.spec)[::-1],
                                         extent=[self.times[0], self.times[-1] + dt, 400, 1200],
                                         aspect='auto', alpha=0.7, cmap='jet', vmin=vmin, vmax=vmax,
                                         interpolation='gaussian', zorder = 1)
        self.done.emit()

class LifeEODfUpdate(QThread):
    done = pyqtSignal()

    def __init__(self):
        super(QThread, self).__init__()

        self.ax = None
        self.handle = None

        self.eodf = None
        self.times = None

    def run(self):
        # print('i try')
        t_array = []
        for enu, f in enumerate(self.eodf):
            t_array.extend(np.ones(len(f)) * self.times[enu])

        if self.handle == None:
            self.handle, = self.ax.plot(t_array, np.hstack(self.eodf), 'o', color ='k', zorder = 2)
        else:
            self.handle.set_data(t_array, np.hstack(self.eodf))
        self.ax.set_ylim(400, 1200)
        self.ax.set_xlim(self.times[0], self.times[-1] + (self.times[1] - self.times[0]))

        self.done.emit()

class LifePlot(QMainWindow):
    def __init__(self):
        self.running = False

        super(QMainWindow, self).__init__()
        self.setGeometry(400, 200, 800, 600)  # set window proportion
        self.setWindowTitle('Spectrum')  # set window title

        self.update = LifeSpecUpdate()
        # self.update.done.connect(self.cv_draw)

        self.update_eodf = LifeEODfUpdate()
        self.update_eodf.done.connect(self.cv_draw)

        self.central_widget = QWidget(self)
        self.gridLayout = QGridLayout()

        self.figure = plt.figure()
        self.update.figure = self.figure

        self.canvas = FigureCanvas(self.figure)
        self.update.canvas = self.canvas

        self.ax = self.figure.add_subplot(111)
        self.update.ax = self.ax
        self.update_eodf.ax = self.ax

        self.gridLayout.addWidget(self.canvas, 0, 0)

        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

    def cv_draw(self):
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)  # create application
    w = MainWindow()  # create window
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()