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
# from .spectrogram import *

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


def open_raw_data(filename: str,
                  buffersize: float = 60.,
                  backsize: float = 0.,
                  channel: int = -1,
                  snippet_size: int = 2**21,
                  verbose: int = 0,
                  logger = None,
                  **kwargs: dict):

    folder = os.path.split(filename)[0]
    # filename = os.path.join(folder, 'traces-grid1.raw')
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

    data, samplerate, channels, dataset, data_shape = open_raw_data(filename=args.file, verbose=args.verbose, **cfg.spectrogram)

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

if __name__ == '__main__':
    example_data = "/data1/data/2023_Breeding/raw_data/2023-02-09-08_16/traces-grid1.raw"
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    # parser.add_argument('-f', '--folder', type=str, help='file to be analyzed', default=example_data)
    parser.add_argument('file', nargs='?', type=str, help='file to be analyzed')
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    parser.add_argument('-s', '--shell', action='store_true', help='execute shell pipeline')
    args = parser.parse_args()
    if args.file:
        args.file = os.path.normpath(args.file)

    main(args)