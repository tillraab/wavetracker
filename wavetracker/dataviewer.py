import queue

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import DataLoader as open_data
from IPython import embed
from .config import Configuration
from .spectrogram import *
from .datahandler import *

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import pyqtgraph as pg
import time
import queue
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if len(tf.config.list_physical_devices('GPU')):
        available_GPU = True
except ImportError:
    available_GPU = False

class ImagePlotWithHist(QWidget):
    update_data_sig = pyqtSignal(object)
    v_min_max_adapt_sig = pyqtSignal(object)
    def __init__(self, data, verbose = 0, parent=None):
        super(ImagePlotWithHist, self).__init__()

        self.data = data
        self.verbose = verbose

        self.plot_x_min, self.plot_x_max = 0., 1.
        self.data_x_min, self.data_x_max = 0., 0.
        self.x_min_for_sb = np.linspace(0, (self.data.shape[0] - (self.plot_x_max - self.plot_x_min) * self.data.samplerate) / self.data.samplerate, 100)
        self.x_max_for_sb = np.linspace(self.plot_x_max - self.plot_x_min, (self.data.shape[0]) / self.data.samplerate, 100)
        self.plot_max_d_xaxis = 5.

        self.content_layout = QGridLayout(self)

        self.plot_handels = []
        self.plot_widgets = []

        self.win = pg.GraphicsLayoutWidget()
        self.plot_handels.append(pg.ImageItem(ColorMap='viridis'))
        self.plot_widgets.append(self.win.addPlot(title=""))
        self.plot_widgets[0].addItem(self.plot_handels[0], colorMap='viridis')
        self.plot_widgets[0].setLabel('left', 'frequency [Hz]')
        self.plot_widgets[0].setLabel('bottom', 'time [s]')

        # self.sum_spec_h.addColorBar(self.sum_spec_img, colorMap='viridis', values =(self.v_min, self.v_max))
        self.power_hist = pg.HistogramLUTItem()
        self.power_hist.setImageItem(self.plot_handels[0])
        self.power_hist.gradient.loadPreset('viridis')

        self.power_hist.axis.setLabel('power [dB]')
        self.power_hist.sigLevelsChanged.connect(lambda: self.v_min_max_adapt_sig.emit(self))

        self.win.addItem(self.power_hist)

        self.content_layout.addWidget(self.win, 0, 0)

        self.update_xrange_without_xlim_grep = False

        self.plot_widgets[0].sigXRangeChanged.connect(lambda: DataViewer.plot_xlims_changed(self))


class SubplotScrollareaWidget(QScrollArea):
    update_data_sig = pyqtSignal(object)
    def __init__(self, plots_per_row, num_rows_visible, data, verbose = 0, parent=None):
        super(QScrollArea, self).__init__()
        self._scroll_ylim_per_double_click = False # can be activated

        self.verbose = verbose
        self.data = data

        self.plot_x_min, self.plot_x_max = 0., 1.
        self.data_x_min, self.data_x_max = 0., 0.
        self.x_min_for_sb = np.linspace(0, (self.data.shape[0] - (self.plot_x_max - self.plot_x_min) * self.data.samplerate) / self.data.samplerate, 100)
        self.x_max_for_sb = np.linspace(self.plot_x_max - self.plot_x_min, (self.data.shape[0]) / self.data.samplerate, 100)
        self.plot_max_d_xaxis = 5.

        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(2)

        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        self.setWidget(self.content_widget)

        self.plots_per_row = plots_per_row
        self.num_rows_visible = num_rows_visible
        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        self.subplot_height = height / self.num_rows_visible

        self.plot_handels = []
        self.plot_widgets = []

        self.update_xrange_without_xlim_grep = False

    def create_subplots(self, fn, xlabel='x', ylabel='y'):
        for channel in range(self.data.channels):
            row, col = channel // self.plots_per_row, channel % self.plots_per_row

            plot_widget = pg.PlotWidget()
            plot_widget.setMinimumHeight(int(self.subplot_height))

            plot_widget.setLabel('bottom', xlabel)
            plot_widget.setLabel('left', ylabel)


            plot_widget.addItem(subplot_h := fn())
            self.content_layout.addWidget(plot_widget, row, col, 1, 1)

            self.plot_widgets.append(plot_widget)
            self.plot_handels.append(subplot_h)

            if channel >= 1:
                plot_widget.setXLink(self.plot_widgets[0])
                plot_widget.setYLink(self.plot_widgets[0])

        self.plot_widgets[0].sigXRangeChanged.connect(lambda: DataViewer.plot_xlims_changed(self))

    @property
    def scroll_ylim_per_double_click(self):
        return self._scroll_ylim_per_double_click

    @scroll_ylim_per_double_click.setter
    def scroll_ylim_per_double_click(self, value: bool):
        self._scroll_ylim_per_double_click = bool(value)
        if value == True:
            for plot_widget in self.plot_widgets:
                plot_widget.mouseDoubleClickEvent = lambda event, p=plot_widget: self.adjust_ylim_to_double_clicked_subplot(
                    event, p)
        else:
            for plot_widget in self.plot_widgets:
                plot_widget.mouseDoubleClickEvent = (lambda *args: None)

    def adjust_ylim_to_double_clicked_subplot(self, event, plot):
        x0, x1 = self.plot_widgets[0].getAxis('bottom').range
        x0 = 0 if x0 < 0 else x0
        plot_idx = self.content_layout.indexOf(plot)
        doi = self.data[int(x0 * self.data.samplerate):int(x1* self.data.samplerate) + 1, plot_idx]

        y_min, y_max = np.min(doi), np.max(doi)
        dy = (y_max-y_max)
        y_min -= dy*0.05
        y_max += dy*0.05
        for pw in self.plot_widgets:
            pw.setYRange(y_min, y_max, padding=0)


class DataViewer(QWidget):
    kill = pyqtSignal()

    def __init__(self, data=None, cfg=None, verbose = 0, parent=None):
        super(DataViewer, self).__init__()

        self.verbose = verbose

        # params: data plotting
        self.plot_max_d_xaxis = 5.
        self.plot_current_d_xaxis = 1.
        self.x_min, self.x_max = 0., self.plot_current_d_xaxis
        self.current_data_xrange = (0, self.plot_current_d_xaxis*2., self.plot_current_d_xaxis*2.)

        self.update_xrange_by_scrollbar = False

        # params: spectrogram
        self.cfg = None
        if cfg:
            self.cfg = cfg
            self.v_min, self.v_max = self.cfg.harmonic_groups['min_good_peak_power'], -50
            self.min_freq, self.max_freq = self.cfg.harmonic_groups['min_freq'], cfg.harmonic_groups['max_freq']
            self.snippet_size = self.cfg.spectrogram['snippet_size']
            self.nfft = self.cfg.spectrogram['nfft']
            self.overlap_frac = self.cfg.spectrogram['overlap_frac']
        else:
            self.v_min, self.v_max = -100, -50
            self.min_freq, self.max_freq = 400, 1200
            self.snippet_size = 2 ** 21
            self.nfft = 2 ** 12
            self.overlap_frac = 0.9

        ### data and spec init
        self.data = data
        self.Spec = Spectrogram(data.samplerate, data.shape, snippet_size=self.snippet_size, nfft=self.nfft,
                                overlap_frac=self.overlap_frac, channels = -1, gpu_use= available_GPU)

        ### subplot layout
        self.plots_per_row = 3
        self.num_rows_visible = 3
        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        self.subplot_height = height / self.num_rows_visible

        ### layout -- traces per channel
        self.main_layout = QGridLayout(self)

        self.force_update_spec_plot = False

        self.TracesSubPlots = SubplotScrollareaWidget(plots_per_row=3, num_rows_visible=3, data = self.data, verbose=self.verbose)
        self.TracesSubPlots.create_subplots(fn=pg.PlotCurveItem, xlabel ='time [s]', ylabel='ampl. [a.U.]')
        self.TracesSubPlots.update_data_sig.connect(self.plot_update_traces)
        self.TracesSubPlots.scroll_ylim_per_double_click = True
        self.TracesSubPlots.plot_widgets[0].setXRange(self.TracesSubPlots.plot_x_min, self.TracesSubPlots.plot_x_max)
        self.main_layout.addWidget(self.TracesSubPlots, 0, 0)

        self.SpecsSubPlots = SubplotScrollareaWidget(plots_per_row=3, num_rows_visible=3, data = self.data, verbose=self.verbose)
        self.SpecsSubPlots.create_subplots(fn=pg.ImageItem, xlabel ='time [s]', ylabel='frequency [Hz]')
        self.SpecsSubPlots.update_data_sig.connect(self.plot_update_specs)
        self.SpecsSubPlots.plot_widgets[0].setXRange(self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max)
        self.SpecsSubPlots.plot_widgets[0].setYRange(self.min_freq, self.max_freq, padding=0)
        self.main_layout.addWidget(self.SpecsSubPlots, 0, 0)
        self.SpecsSubPlots.hide()

        self.SumSpecPlot = ImagePlotWithHist(data=self.data, verbose=self.verbose)
        self.SumSpecPlot.update_data_sig.connect(self.plot_update_sumspec)
        self.SumSpecPlot.v_min_max_adapt_sig.connect(self.vmin_vmax_adapt)
        self.SumSpecPlot.plot_widgets[0].setXRange(self.SumSpecPlot.plot_x_min, self.SumSpecPlot.plot_x_max)
        self.SumSpecPlot.plot_widgets[0].setYRange(self.min_freq, self.max_freq, padding=0)
        self.main_layout.addWidget(self.SumSpecPlot, 0, 0)
        self.SumSpecPlot.hide()

        ### scrollbar for x lims
        self.scroll_val = 0
        self.add_scrollbar_to_adjust_xrange()

        ### Actions
        self.define_actions()

        #
        # ## loop plot parameters
        # self.timer=QTimer()
        # self.timer.timeout.connect(self.check)
        # self.timer.start(1000)

    # def check(self):
    #     print(self.x_min)

    def plot_update_sumspec(self, obj):
        if self.verbose >= 2: print('fn plot_update_sumspec')
        x_idx_0 = int(obj.data_x_min * self.data.samplerate)
        x_idx_1 = int(obj.data_x_max * self.data.samplerate)
        if not self.force_update_spec_plot:
            self.Spec.snippet_spectrogram(self.data[x_idx_0:x_idx_1, :].T, obj.data_x_min)
        self.force_update_spec_plot = False
        obj.plot_handels[0].setImage(decibel(self.Spec.sum_spec.T),
                                   levels=[self.v_min, self.v_max], colorMap='viridis')
        obj.plot_handels[0].setRect(
            pg.QtCore.QRectF(self.Spec.spec_times[0], self.Spec.spec_freqs[0],
                             self.Spec.times[-1] - self.Spec.times[0],
                             self.Spec.spec_freqs[-1] - self.Spec.spec_freqs[0]))

        # obj.plot_widgets[0].setYRange(self.min_freq, self.max_freq)

    def plot_update_specs(self, obj):
        if self.verbose >= 2: print('fn: plot_update_specs')
        x_idx_0 = int(obj.data_x_min * self.data.samplerate)
        x_idx_1 = int(obj.data_x_max * self.data.samplerate)

        if not self.force_update_spec_plot:
            self.Spec.snippet_spectrogram(self.data[x_idx_0:x_idx_1, :].T, obj.data_x_min)
        self.force_update_spec_plot = False

        for ch in range(self.data.channels):
            obj.plot_handels[ch].setImage(decibel(self.Spec.spec[ch, :, :].T),
                                                levels=[self.v_min, self.v_max], colorMap='viridis')
            obj.plot_handels[ch].setRect(
                pg.QtCore.QRectF(self.Spec.spec_times[0], self.Spec.spec_freqs[0],
                                 self.Spec.times[-1] - self.Spec.times[0],
                                 self.Spec.spec_freqs[-1] - self.Spec.spec_freqs[0]))

        # obj.plot_widgets[0].setYRange(self.min_freq, self.max_freq, padding=0)

    def plot_update_traces(self, obj):
        if self.verbose >= 2: print('fn: plot_update_traces')
        x_idx_0 = int(obj.data_x_min * self.data.samplerate)
        x_idx_1 = int(obj.data_x_max * self.data.samplerate)

        if x_idx_1 - x_idx_0 > 10000:
            x = np.array(np.linspace(x_idx_0, x_idx_1, 10000), dtype=int)
        else:
            x = np.array(np.arange(x_idx_0, x_idx_1), dtype=int)

        for enu, plot_widget in enumerate(obj.plot_handels):
            plot_widget.setData(x / self.data.samplerate, self.data[x, enu])

        y_min = np.min(self.data[int(obj.plot_x_min * self.data.samplerate):int(obj.plot_x_max * self.data.samplerate) + 1, :])
        y_max = np.max(self.data[int(obj.plot_x_min * self.data.samplerate):int(obj.plot_x_max * self.data.samplerate) + 1, :])

        for pw in obj.plot_widgets:
            pw.setYRange(y_min, y_max, padding=0)

    @staticmethod
    def plot_xlims_changed(cls): # ToDo: move this method to the master class; emit signal which then triggers this function... .sigXRangeChanged.connect(self.xlim_changed_signal.emit(self))
        if cls.update_xrange_without_xlim_grep:
            cls.update_xrange_without_xlim_grep = False
        else:
            cls.plot_x_min, cls.plot_x_max = cls.plot_widgets[0].getAxis('bottom').range

        cls.x_min_for_sb = np.linspace(0, (cls.data.shape[0] - (cls.plot_x_max - cls.plot_x_min) * cls.data.samplerate) / cls.data.samplerate,100)
        cls.x_max_for_sb = np.linspace(cls.plot_x_max - cls.plot_x_min, (cls.data.shape[0]) / cls.data.samplerate,100)

        if cls.plot_x_min < 0:
            cls.plot_x_max -= cls.plot_x_min
            cls.plot_x_min = 0
            cls.update_xrange_without_xlim_grep = True
            cls.plot_widgets[0].setXRange(cls.plot_x_min, cls.plot_x_max, padding=0)
        elif cls.plot_x_max - cls.plot_x_min > cls.plot_max_d_xaxis:
            cls.plot_x_max = cls.plot_x_min + cls.plot_max_d_xaxis
            cls.update_xrange_without_xlim_grep = True
            cls.plot_widgets[0].setXRange(cls.plot_x_min, cls.plot_x_max, padding=0)  # triggers the same function again
        else:
            if (((cls.plot_x_min < cls.plot_x_min - (cls.plot_x_min - cls.data_x_min) * 0.5) and (cls.plot_x_min > (cls.plot_x_max - cls.plot_x_min))) or
                    (cls.plot_x_max > cls.plot_x_max + (cls.data_x_max - cls.plot_x_max) * 0.5)):
                if cls.verbose >= 1:
                    print('\nemit updating data:')
                    print(f'data_x_min: {cls.data_x_min:.2f}s; plot_x_min: {cls.plot_x_min:.2f}s')
                    print(f'data_x_max: {cls.data_x_max:.2f}s; plot_x_max: {cls.plot_x_max:.2f}s')

                cls.data_x_min = cls.plot_x_min - (cls.plot_x_max - cls.plot_x_min)
                cls.data_x_min = cls.data_x_min if cls.data_x_min > 0 else 0
                cls.data_x_max = cls.plot_x_max + (cls.plot_x_max - cls.plot_x_min)

                cls.update_data_sig.emit(cls)

    def spec_params_changed(self):
        if self.SumSpecPlot.isVisible():
            self.plot_update_sumspec(self.SumSpecPlot)
        elif self.SpecsSubPlots.isVisible():
            self.plot_update_specs(self.SpecsSubPlots)
        else:
            pass

    def add_scrollbar_to_adjust_xrange(self):
        self.x_scrollbar = QScrollBar()
        self.x_scrollbar.setOrientation(1)  # Horizontal orientation
        self.x_scrollbar.setMinimum(0)
        self.x_scrollbar.setMaximum(99)

        self.x_scrollbar.setPageStep(int(0.1* self.x_scrollbar.maximum()))
        self.scroller_position = int(0)
        self.x_scrollbar.setValue(self.scroller_position)

        self.x_scrollbar.sliderReleased.connect(lambda: self.update_plot_x_limits_by_scrollbar(self.x_scrollbar.value()))

        self.main_layout.addWidget(self.x_scrollbar, 1, 0, 1, 1)

    def define_actions(self):
        self.Act_spec_nfft_up = QAction('nfft up', self)
        # self.Act_spec_nfft_up.setEnabled(False)
        self.Act_spec_nfft_up.triggered.connect(lambda: setattr(self.Spec, "nfft", int(2**(np.log2(self.Spec.nfft)+1))))
        self.Act_spec_nfft_up.triggered.connect(self.spec_params_changed)
        self.Act_spec_nfft_up.setShortcut('Shift+F')
        self.addAction(self.Act_spec_nfft_up)

        self.Act_spec_nfft_down = QAction('nfft down', self)
        # self.Act_spec_nfft_down.setEnabled(False)
        self.Act_spec_nfft_down.triggered.connect(lambda: setattr(self.Spec, "nfft",
                                                                  int(2**(np.log2(self.Spec.nfft)-1)) if
                                                                  int(2**(np.log2(self.Spec.nfft)-1)) > 16
                                                                  else self.Spec.nfft))
        self.Act_spec_nfft_down.triggered.connect(self.spec_params_changed)
        self.Act_spec_nfft_down.setShortcut('F')
        self.addAction(self.Act_spec_nfft_down)

        self.Act_spec_overlap_up = QAction('overlap up', self)
        # self.Act_spec_overlap_up.setEnabled(False)
        self.Act_spec_overlap_up.triggered.connect(lambda: setattr(self.Spec, "overlap_frac",
                                                                   self.Spec.overlap_frac + 0.05 if self.Spec.overlap_frac < 0.95 else self.Spec.overlap_frac))
        self.Act_spec_overlap_up.triggered.connect(self.spec_params_changed)
        self.Act_spec_overlap_up.setShortcut('O')
        self.addAction(self.Act_spec_overlap_up)

        self.Act_spec_overlap_down = QAction('overlap down', self)
        # self.Act_spec_overlap_down.setEnabled(False)
        self.Act_spec_overlap_down.triggered.connect(lambda: setattr(self.Spec, "overlap_frac",
                                                                   self.Spec.overlap_frac - 0.05 if self.Spec.overlap_frac > 0.05 else self.Spec.overlap_frac))
        self.Act_spec_overlap_down.triggered.connect(self.spec_params_changed)
        self.Act_spec_overlap_down.setShortcut('Shift+O')
        self.addAction(self.Act_spec_overlap_down)
        pass

    def update_plot_x_limits_by_scrollbar(self, value): #  1-100 as set earlier
        for obj in [self.TracesSubPlots, self.SpecsSubPlots]:
            if obj.isVisible():
                obj.plot_x_min = obj.x_min_for_sb[value]
                obj.plot_x_max = obj.x_max_for_sb[value]

                if self.verbose >= 2: print(f'\nscrollbar: {obj.plot_x_min, obj.plot_x_max}')
                obj.update_xrange_without_xlim_grep = True
                obj.plot_widgets[0].setXRange(obj.plot_x_min, obj.plot_x_max, padding=0)
                return

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            if self.TracesSubPlots.isVisible():
                if (self.SpecsSubPlots.plot_x_min != self.TracesSubPlots.plot_x_min) or (
                        self.SpecsSubPlots.plot_x_max != self.TracesSubPlots.plot_x_max):
                    self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max = self.TracesSubPlots.plot_x_min, self.TracesSubPlots.plot_x_max
                    self.SpecsSubPlots.update_xrange_without_xlim_grep = True
                    self.SpecsSubPlots.plot_widgets[0].setXRange(self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max, padding=0)

                self.SpecsSubPlots.show()
                self.TracesSubPlots.hide()
            elif self.SpecsSubPlots.isVisible():
                if (self.SumSpecPlot.plot_x_min != self.SpecsSubPlots.plot_x_min) or (
                        self.SumSpecPlot.plot_x_max != self.SpecsSubPlots.plot_x_max):
                    self.SumSpecPlot.plot_x_min, self.SumSpecPlot.plot_x_max = self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max
                    self.SumSpecPlot.update_xrange_without_xlim_grep = True
                    self.SumSpecPlot.plot_widgets[0].setXRange(self.SumSpecPlot.plot_x_min, self.SumSpecPlot.plot_x_max, padding=0)

                f0, f1 = self.SpecsSubPlots.plot_widgets[0].getAxis('left').range
                self.SumSpecPlot.plot_widgets[0].setYRange(f0, f1, padding=0)
                self.SumSpecPlot.show()
                self.SpecsSubPlots.hide()

            elif self.SumSpecPlot.isVisible():
                if (self.SpecsSubPlots.plot_x_min != self.SumSpecPlot.plot_x_min) or (
                        self.SpecsSubPlots.plot_x_max != self.SumSpecPlot.plot_x_max):
                    self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max = self.SumSpecPlot.plot_x_min, self.SumSpecPlot.plot_x_max
                    self.SpecsSubPlots.update_xrange_without_xlim_grep = True
                    self.SpecsSubPlots.plot_widgets[0].setXRange(self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max, padding=0)
                elif self.force_update_spec_plot:
                    self.plot_update_specs(self.SpecsSubPlots)

                f0, f1 = self.SumSpecPlot.plot_widgets[0].getAxis('left').range
                self.SpecsSubPlots.plot_widgets[0].setYRange(f0, f1, padding=0)
                self.SpecsSubPlots.show()
                self.SumSpecPlot.hide()
            else:
                pass
        if event.key() == Qt.Key_T:
            if self.SpecsSubPlots.isVisible():
                if (self.SpecsSubPlots.plot_x_min != self.TracesSubPlots.plot_x_min) or (
                        self.SpecsSubPlots.plot_x_max != self.TracesSubPlots.plot_x_max):
                    self.TracesSubPlots.plot_x_min, self.TracesSubPlots.plot_x_max = self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max
                    self.TracesSubPlots.update_xrange_without_xlim_grep = True
                    self.TracesSubPlots.plot_widgets[0].setXRange(self.SpecsSubPlots.plot_x_min, self.SpecsSubPlots.plot_x_max, padding=0)

                self.TracesSubPlots.show()
                self.SpecsSubPlots.hide()

            if self.SumSpecPlot.isVisible():
                if (self.SumSpecPlot.plot_x_min != self.TracesSubPlots.plot_x_min) or (
                        self.SumSpecPlot.plot_x_max != self.TracesSubPlots.plot_x_max):
                    self.TracesSubPlots.plot_x_min, self.TracesSubPlots.plot_x_max = self.SumSpecPlot.plot_x_min, self.SumSpecPlot.plot_x_max
                    self.TracesSubPlots.update_xrange_without_xlim_grep = True
                    self.TracesSubPlots.plot_widgets[0].setXRange(self.SumSpecPlot.plot_x_min, self.SumSpecPlot.plot_x_max, padding=0)

                self.TracesSubPlots.show()
                self.SumSpecPlot.hide()

        if event.key() == Qt.Key_Q:
            self.kill.emit()

    def vmin_vmax_adapt(self, cls):
        # Obtain the new lookup table values
        levels = cls.power_hist.getLevels()
        self.force_update_spec_plot = True
        self.v_min, self.v_max = levels


class DataViewerStandalone(QMainWindow):
    def __init__(self, args, parent=None):
        super(DataViewerStandalone, self).__init__(parent)
        self.args = args

        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        width = rec.width()
        self.resize(int(1 * width), int(1 * height))
        self.setWindowTitle('datahandler')  # set window title

        self.central_widget = QWidget(self) # basic widget ; can be exchanged with tabwidget etc.

        self.gridLayout = QGridLayout()

        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

        self.cfg = Configuration(self.args.config, verbose=self.args.verbose)

        ### data
        if not self.args.file:
            self.open_btn = QPushButton('Open file', self.central_widget)
            self.open_btn.clicked.connect(self.open_with_file_dialog)
            self.gridLayout.addWidget(self.open_btn, 0, 0)
        else:
            self.file = self.args.file
            self.go_to_DataViewer()

    def open_with_file_dialog(self):
        fd = QFileDialog()
        self.file, _ = fd.getOpenFileName(self, "Open File", '/home', "*.raw *.wav")
        # embed()
        # quit()
        # self.folder = fd.getExistingDirectory(self, 'Select Directory')
        if self.file != '':
            self.file = os.path.abspath(self.file)
            self.open_btn.hide()
            self.go_to_DataViewer()

    def go_to_DataViewer(self):
        data, samplerate, channels, dataset, data_shape = open_raw_data(filename=self.file)
        # data_viewer_widget = DataViewer(data, self.cfg)
        data_viewer_widget = DataViewer(data, verbose=self.args.verbose)
        self.gridLayout.addWidget(data_viewer_widget, 0, 0, 1, 1)
        data_viewer_widget.kill.connect(lambda: self.close())


def main_UI():
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    # parser.add_argument('-f', '--file', type=str, help='file to be analyzed', default=None)
    parser.add_argument('file', nargs='?', type=str, help='file to be analyzed')
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    parser.add_argument('-s', '--shell', action='store_true', help='execute shell pipeline')
    args = parser.parse_args()
    if args.file:
        args.file = os.path.normpath(args.file)

    app = QApplication(sys.argv)  # create application
    w = DataViewerStandalone(args)  # create window
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main_UI()
