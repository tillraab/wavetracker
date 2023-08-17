import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import DataLoader as open_data
from IPython import embed
from .config import Configuration

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import pyqtgraph as pg
import time
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

        # self.show()

        ### data
        cfg = Configuration(args.config, verbose=args.verbose)
        data, samplerate, channels, dataset, data_shape = open_raw_data(folder=args.folder, verbose=args.verbose, **cfg.spectrogram)
        data_viewer_widget = DataViewer(data)
        self.gridLayout.addWidget(data_viewer_widget, 0, 0, 4, 4)

        # data_viewer_widget.plot_widgets[1].hide() # clears plot data
        # data_viewer_widget.plot_widgets[1].hide() # hides plot temporarily

class DataViewer(QWidget):
    def __init__(self, data, parent=None):
        super(DataViewer, self).__init__()
        self.data = data

        self.layout = QGridLayout()
        # ToDo: reaplace this with a scroll area
        # self.layout = QScrollArea()
        self.setLayout(self.layout)

        self._create_channel_subplots()

        self._initial_plot()

        for plot_widget in self.plot_widgets:
            plot_widget.sigXRangeChanged.connect(self._update_plot)

    def _initial_plot(self):
        for i in range(self.data.channels):
            self.plot_handels[i].setData(np.arange(self.data.samplerate), self.data[:self.data.samplerate, i])

            self.plot_widgets[0].setXRange(0, 20000)

    def _create_channel_subplots(self):
        c = 0
        self.plot_handels = []
        self.plot_widgets = []
        for row in range(self.data.channels//4 + 1):
            for col in range(4):
                if c >= self.data.channels:
                    print('breaking')
                    break
                plot_widget = pg.PlotWidget()
                subplot_h = plot_widget.plot()
                self.layout.addWidget(plot_widget, row, col, 1, 1)

                self.plot_widgets.append(plot_widget)
                self.plot_handels.append(subplot_h)

                # print(c)
                if c >= 1:
                    plot_widget.setXLink(self.plot_widgets[0])
                    plot_widget.setYLink(self.plot_widgets[0])
                c += 1

    def _update_plot(self):
        # print('yay')
        x_min, x_max = self.plot_widgets[0].getAxis('bottom').range
        x_min = 0 if x_min < 0 else x_min
        for enu, plot_widget in enumerate(self.plot_handels):
            # print(enu)
            x = np.arange(x_min, x_max + 1)
            y = self.data[x_min:x_max+1, enu]

            if len(y) > len(x):
                y = y[:len(x)]
            elif len(x) > len(y):
                x = x[:len(y)]
            else:
                pass
            plot_widget.setData(x, y)

        # self.plot_handels[0].setXRange(x_min, x_max)
        # for ch in range(len(self.plot_handels)):
        #     self.plot_handels[ch].setData(np.arange(x_min, x_max+1), self.data[x_min:x_max, ch])
        # new_x = np.linspace(x_min, x_max, 1000)
        # new_y = np.sin(new_x)

        # Update the plot data
        # self.plot.setData(new_x, new_y)

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
