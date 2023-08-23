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


def multi_channel_audio_file_generator(filename: str,
                                       channels: int,
                                       data_snippet_idxs: int):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        while True:
            chunk = f.read(data_snippet_idxs * channels * 4) # 4 bytes per float32 value
            if chunk:
                chunk = tf.io.decode_raw(chunk, tf.float32, fixed_length=data_snippet_idxs * channels * 4)
                chunk = chunk.reshape([-1, channels])
                yield chunk
            else:
                break


def open_raw_data(folder: str,
                  buffersize: float = 60.,
                  backsize: float = 0.,
                  channel: int = -1,
                  snippet_size: int = 2**21,
                  verbose: int = 0,
                  logger = None,
                  **kwargs: dict):

    filename = os.path.join(folder, 'traces-grid1.raw')
    data = open_data(filename, buffersize=buffersize, backsize=backsize, channel=channel)
    samplerate = data.samplerate
    channels = data.channels
    shape = data.shape

    GPU_str = "(gpu found: TensorGenerator created)" if available_GPU else "(NO gpu: NO TensorGenerator created)"
    if verbose >= 1: print(f'{"Loading data from":^25}: {os.path.abspath(folder)}\n{" "*27 + GPU_str}')
    if logger: logger.info(f'{"Loading data from":^25}: {os.path.abspath(folder)}')
    if logger: logger.info(f'{" "*27 + GPU_str}')
    dataset = None
    if available_GPU:
        dataset = tf.data.Dataset.from_generator(
            multi_channel_audio_file_generator,
            args=(filename, channels, snippet_size),
            output_types=tf.float32,
            output_shapes=tf.TensorShape([None, channels]))

    return data, samplerate, channels, dataset, shape

def main(args):
    if args.verbose >= 1: print(f'\n--- Running wavetracker.datahandler ---')

    cfg = Configuration(args.config, verbose=args.verbose)

    data, samplerate, channels, dataset, data_shape = open_raw_data(folder=args.folder, verbose=args.verbose, **cfg.spectrogram)

    fig, ax = plt.subplots(int(np.ceil(data_shape[1] / 2)), 2, figsize=(20 / 2.54, 20 / 2.54), sharex='all', sharey='all')
    ax = np.hstack(ax)
    d = data[0: cfg.spectrogram['snippet_size'], :]
    fig.suptitle('Data loaded with thunderfish.DataLoader')
    for i in range(channels):
        ax[i].plot(np.arange(cfg.spectrogram['snippet_size']) / samplerate, d[:, i])
        ax[i].text(0.9, 0.9, f'{i}', transform=ax[i].transAxes, ha='center', va='center')
    plt.show()

    if available_GPU:
        for enu, data in enumerate(dataset.take(2)):
            fig, ax = plt.subplots(int(np.ceil(data_shape[1]/2)), 2, figsize=(20/2.54, 20/2.54), sharex='all', sharey='all')
            ax = np.hstack(ax)
            d = data.numpy()
            fig.suptitle('Data loaded with tensorflow.generator')
            for i in range(channels):
                ax[i].plot((np.arange(cfg.spectrogram['snippet_size']) + enu * cfg.spectrogram['snippet_size']) / samplerate, d[:, i])
                ax[i].text(0.9, 0.9, f'{i}', transform = ax[i].transAxes, ha='center', va='center')
            plt.show()

class MainWindow(QMainWindow):
    def __init__(self, args, parent=None):
        super(MainWindow, self).__init__(parent)
        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        width = rec.width()
        self.resize(int(1 * width), int(1 * height))
        self.setWindowTitle('datahandler')  # set window title

        self.central_widget = QWidget(self) # basic widget ; can be exchanged with tabwidget etc.

        self.gridLayout = QGridLayout()

        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

        ### data
        cfg = Configuration(args.config, verbose=args.verbose)
        data, samplerate, channels, dataset, data_shape = open_raw_data(folder=args.folder, verbose=args.verbose, **cfg.spectrogram)
        data_viewer_widget = DataViewer(data)
        self.gridLayout.addWidget(data_viewer_widget, 0, 0, 1, 1)


class DataViewer(QWidget):
    def __init__(self, data, parent=None):
        super(DataViewer, self).__init__()
        self.data = data

        # params: scrollbar
        self._rel_scroller_size = 0.1

        # params: data plotting
        self.plot_max_d_xaxis = self.data.samplerate * 2
        self.plot_current_d_xaxis = self.data.samplerate
        self.x_min_for_sb = np.linspace(0, self.data.shape[0]-self.plot_current_d_xaxis, 100)
        self.x_max_for_sb = np.linspace(self.plot_current_d_xaxis, self.data.shape[0], 100)

        # layout
        self.main_layout = QGridLayout(self)

        self.scroll_area_traces = QScrollArea() # this is a widget
        self.scroll_area_traces.setWidgetResizable(True)
        self.scroll_area_traces.setVerticalScrollBarPolicy(2)  # Always show scrollbar

        self.content_widget_traces = QWidget()  # Create a content widget for the scroll area
        self.content_layout_traces = QGridLayout(self.content_widget_traces)  # Use QGridLayout for the content widget
        self.scroll_area_traces.setWidget(self.content_widget_traces)  # Set the content widget as the scroll area's content

        self.main_layout.addWidget(self.scroll_area_traces, 0, 0)

        ###########################################
        self.scroll_area_spec = QScrollArea() # this is a widget
        self.scroll_area_spec.setWidgetResizable(True)
        self.scroll_area_spec.setVerticalScrollBarPolicy(2)  # Always show scrollbar

        self.content_widget_spec = QWidget()  # Create a content widget for the scroll area
        self.content_layout_spec = QGridLayout(self.content_widget_spec)  # Use QGridLayout for the content widget
        self.scroll_area_spec.setWidget(self.content_widget_spec)  # Set the content widget as the scroll area's content

        self.main_layout.addWidget(self.scroll_area_spec, 0, 0)
        self.scroll_area_spec.hide()
        ###########################################

        self.plots_per_row = 3
        self.num_rows_visible = 3
        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        self.subplot_height = height / self.num_rows_visible
        self.create_channel_subplots()

        self.update_by_scrollbar = False
        self.add_plot_scrollbar()

        self.current_data_xrange = (0, self.data.samplerate * 2, self.data.samplerate * 2)
        self.initial_plot()

        self.plot_widgets_trace[0].sigXRangeChanged.connect(self.update_data_in_all_subplotsplot)

        self.Spec = Spectrogram(data.samplerate, data.shape, snippet_size=2**21, nfft=2**12, overlap_frac=0.9, channels = -1, gpu_use= available_GPU)
        # Spec.snippet_spectrogram(data[self.current_data_xrange[0]:self.current_data_xrange[1], :].T, 0)

    def switch_to_spectrograms(self):
        self.Spec.snippet_spectrogram(self.data[self.current_data_xrange[0]:self.current_data_xrange[1], :].T, 0)

        print(self.current_data_xrange[2]/self.data.samplerate)
        for ch in range(self.data.channels):
            self.plot_handels_spec[ch].setImage(decibel(self.Spec.spec[ch].T), levels=[-100, -50])
            self.plot_handels_spec[ch].setRect(
                pg.QtCore.QRectF(self.Spec.spec_times[0], self.Spec.spec_freqs[0], self.Spec.times[-1] - self.Spec.times[0], self.Spec.spec_freqs[-1] - self.Spec.spec_freqs[0]))
            # self.plot_handels_spec[ch].setColorMap(pg.colormap.ColorMap("viridis").getLookupTable())


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            if not self.scroll_area_traces.isHidden():
                self.switch_to_spectrograms()
                self.scroll_area_spec.show()
                self.scroll_area_traces.hide()
            else:
                for ch in range(self.data.channels):
                    self.plot_handels_spec[ch].setImage()
                self.scroll_area_traces.show()
                self.scroll_area_spec.hide()

    def create_channel_subplots(self):
        c = 0
        self.plot_handels_trace = []
        self.plot_widgets_trace = []

        self.plot_handels_spec = []
        self.plot_widgets_spec = []

        for row in range(self.data.channels//self.plots_per_row + 1):
            for col in range(self.plots_per_row):
                if c >= self.data.channels:
                    break

                plot_widget = pg.PlotWidget()
                plot_widget.setMinimumHeight(int(self.subplot_height))
                plot_widget.mouseDoubleClickEvent = lambda event, p=plot_widget: self.adjust_ylim_to_double_clicked_subplot(event, p)
                plot_widget.setLabel('bottom', "samples")
                plot_widget.setLabel('left', "ampl [aU]")

                subplot_h = plot_widget.plot()

                self.content_layout_traces.addWidget(plot_widget, row, col, 1, 1)

                self.plot_widgets_trace.append(plot_widget)
                self.plot_handels_trace.append(subplot_h)

                if c >= 1:
                    plot_widget.setXLink(self.plot_widgets_trace[0])
                    plot_widget.setYLink(self.plot_widgets_trace[0])


                ####
                plot_widget_s = pg.PlotWidget()
                plot_widget_s.setMinimumHeight(int(self.subplot_height))
                plot_widget_s.setLabel('bottom', "time [s]")
                plot_widget_s.setLabel('left', "freq [Hz]")

                subplot_h_s = pg.ImageItem()
                plot_widget_s.addItem(subplot_h_s)

                self.content_layout_spec.addWidget(plot_widget_s, row, col, 1, 1)

                self.plot_widgets_spec.append(plot_widget_s)
                self.plot_handels_spec.append(subplot_h_s) # ToDo: add whatever needs to be added here
                c += 1

    def initial_plot(self):
        for i in range(self.data.channels):
            self.plot_handels_trace[i].setData(np.arange(self.data.samplerate * 2), self.data[:self.data.samplerate * 2, i])
        self.plot_widgets_trace[0].setXRange(0, self.plot_current_d_xaxis)
        # self.current_data_xrange = (0, self.data.samplerate*2, self.data.samplerate*2)

    def add_plot_scrollbar(self):
        self.x_scrollbar = QScrollBar()
        self.x_scrollbar.setOrientation(1)  # Horizontal orientation
        self.x_scrollbar.setMinimum(0)
        self.x_scrollbar.setMaximum(99)

        # stylesheet = f"""
        #      QScrollBar:horizontal {{
        #          background: lightgray;
        #          height: 20px;
        #      }}
        #
        #      QScrollBar::handle:horizontal {{
        #          background: gray;
        #      }}
        #
        #      QScrollBar::add-line:horizontal {{
        #          background: none;
        #      }}
        #
        #      QScrollBar::sub-line:horizontal {{
        #          background: none;
        #      }}
        #  """
        # self.x_scrollbar.setStyleSheet(stylesheet)

        self.x_scrollbar.setPageStep(int(self._rel_scroller_size * self.x_scrollbar.maximum()))
        self.scroller_position = int(0 * self.x_scrollbar.maximum())
        self.x_scrollbar.setValue(self.scroller_position)

        # self.x_scrollbar.valueChanged.connect(self._update_x_limits_by_scrollbar)
        self.x_scrollbar.sliderReleased.connect(lambda: self.update_plot_x_limits_by_scrollbar(self.x_scrollbar.value()))

        self.main_layout.addWidget(self.x_scrollbar, 1, 0, 1, 1)

    def update_plot_x_limits_by_scrollbar(self, value): #  1-100 as set earlier
        self.x_min = int(self.x_min_for_sb[value])
        self.x_max = int(self.x_max_for_sb[value])

        self.plot_current_d_xaxis = self.x_max - self.x_min
        self.x_min_for_sb = np.linspace(0, self.data.shape[0]-self.plot_current_d_xaxis, 100)
        self.x_max_for_sb = np.linspace(self.plot_current_d_xaxis, self.data.shape[0], 100)

        self.update_by_scrollbar = True
        self.plot_widgets_trace[0].setXRange(self.x_min, self.x_max, padding=0) # triggers self._update_plot

    def update_data_in_all_subplotsplot(self):
        if self.update_by_scrollbar:
            self.update_by_scrollbar = False
        else:
            self.x_min, self.x_max = self.plot_widgets_trace[0].getAxis('bottom').range
            self.plot_current_d_xaxis = self.x_max - self.x_min

        # ToDo: implement solution for when we scroll beound data end
        if self.x_min < 0:
            self.x_min = 0
            self.x_max = self.plot_current_d_xaxis
            self.plot_widgets_trace[0].setXRange(self.x_min, self.x_max, padding=0)
        elif self.x_max - self.x_min > self.plot_max_d_xaxis:
            self.x_max = self.x_min + self.plot_max_d_xaxis
            self.plot_current_d_xaxis = self.plot_max_d_xaxis
            self.plot_widgets_trace[0].setXRange(self.x_min, self.x_max, padding=0) # triggers the same function again
        else:
            self.x_min_for_sb = np.linspace(0, self.data.shape[0] - self.plot_current_d_xaxis, 100)
            self.x_max_for_sb = np.linspace(self.plot_current_d_xaxis, self.data.shape[0], 100)

            if (self.x_min < self.current_data_xrange[0] + 0.1*self.current_data_xrange[2]) or (
                    self.x_max > self.current_data_xrange[1] - 0.1*self.current_data_xrange[2]):

                plot_x_idx0 = self.x_min - self.plot_current_d_xaxis
                plot_x_idx0 = plot_x_idx0 if plot_x_idx0 >= 0 else 0
                plot_x_idx1 = self.x_max + self.plot_current_d_xaxis

                # self.current_data_xrange = (self.x_min - self.plot_current_d_xaxis,
                #                             self.x_max + self.plot_current_d_xaxis,
                #                             (self.x_max - self.x_min + 2*self.plot_current_d_xaxis))
                self.current_data_xrange = (plot_x_idx0, plot_x_idx1, plot_x_idx1 - plot_x_idx0)

                for enu, plot_widget in enumerate(self.plot_handels_trace):


                    x = np.arange(plot_x_idx0, plot_x_idx1)
                    y = self.data[plot_x_idx0:plot_x_idx1, enu]

                    if len(y) > len(x):
                        y = y[:len(x)]
                    elif len(x) > len(y):
                        x = x[:len(y)]
                    else:
                        pass
                    plot_widget.setData(x, y)

            y_min = np.min(self.data[self.x_min:self.x_max+1, :])
            y_max = np.max(self.data[self.x_min:self.x_max+1, :])
            for pw in self.plot_widgets_trace:
                pw.setYRange(y_min, y_max, padding=0)

    def adjust_ylim_to_double_clicked_subplot(self, event, plot):
        plot_idx = self.content_layout_traces.indexOf(plot)
        doi = self.data[self.current_data_xrange[0]:self.current_data_xrange[1], plot_idx]
        y_min, y_max = np.min(doi), np.max(doi)
        dy = (y_max-y_max)
        y_min -= dy*0.05
        y_max += dy*0.05
        for pw in self.plot_widgets_trace:
            pw.setYRange(y_min, y_max, padding=0)


if __name__ == '__main__':
    # run as: run as "python3 -m wavetracker.dataloader"

    # example_data = "/home/raab/data/2023-02-09-08_16"
    example_data = "/data1/data/2023_Breeding/raw_data/2023-02-09-08_16"
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('-f', '--folder', type=str, help='file to be analyzed', default=example_data)
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    parser.add_argument('-s', '--shell', action='store_true', help='execute shell pipeline')
    args = parser.parse_args()
    args.folder = os.path.normpath(args.folder)

    if args.shell:
        main(args)
    else:
        app = QApplication(sys.argv)  # create application
        w = MainWindow(args)  # create window
        w.show()
        sys.exit(app.exec_())  # exit if window is closed
