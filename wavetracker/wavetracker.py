import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from .config import Configuration
from .datahandler import open_raw_data
from .signal_tracker import freq_tracking_v5

try:
    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if len(tf.config.list_physical_devices('GPU')):
        available_GPU = True
except:
    available_GPU = False


def main():
    example_data = "/home/raab/data/2023-02-09-08_16"
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('-f', '--folder', type=str, help='file to be analyzed', default=example_data)
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('--cpu', action='store_true', help='analysis using only CPU.')
    args = parser.parse_args()
    args.folder = os.path.normpath(args.folder)

    if args.verbose >= 1: print(f'\n--- Running wavetracker.wavetracker ---')

    # load wavetracker configuration
    cfg = Configuration(args.config, verbose=args.verbose)

    # load data
    data, samplerate, channels, dataset, data_shape = open_raw_data(folder=args.folder, verbose=args.verbose,
                                                                    **cfg.spectrogram)

    # compute spectrograms
    if available_GPU:

        pass
    else:

        pass

    # harmonic group extraction

    # tracking

    # save and exit

    embed()
    quit()
    pass


if __name__ == '__main__':
    main()
