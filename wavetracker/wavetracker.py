import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from thunderfish.dataloader import DataLoader as open_data
from .config import Configuration
from .signal_tracker import freq_tracking_v5

try:
    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if len(tf.config.list_physical_devices('GPU')):
        use_GPU = True
except:
    use_GPU = False


def multi_channel_audio_file_generator(filename, channels, data_snippet_idxs):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        while True:
            chunk = f.read(data_snippet_idxs * channels * 4) # 4 bytes per float32 value
            if chunk:
                chunk = tf.io.decode_raw(chunk, tf.float32)
                chunk = chunk.reshape([-1, channels])
                yield chunk
            else:
                break

def open_raw_data(folder: str,
                  buffersize: float = 60.,
                  backsize: float = 0.,
                  channel: int = -1,
                  snippet_size: int = 2**21,
                  **kwargs: dict):

    filename = os.path.join(folder, 'traces-grid1.raw')
    data = open_data(filename, buffersize=buffersize, backsize=backsize, channel=channel)
    samplerate = data.samplerate
    channels = data.channels

    dataset = None
    if use_GPU:
        dataset = tf.data.Dataset.from_generator(
            multi_channel_audio_file_generator,
            args=(filename, channels, snippet_size),
            output_types=tf.float32,
            output_shapes=tf.TensorShape([None, channels]))

    return data, samplerate, channels, dataset

def main():
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('folder', type=str, help='file to be analyzed', default='')
    parser.add_argument('-c', "--config", type=str, help="<config>.yaml file for analysis", default=None)
    parser.add_argument('-v', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    args = parser.parse_args()

    # load wavetracker configuration
    cfg = Configuration()

    # load data
    data, samplerate, channels, dataset = open_raw_data(folder=args.folder, **cfg.data_processing)

    # compute spectrograms

    # harmonic group extraction

    # tracking

    # save and exit

    embed()
    quit()
    pass


if __name__ == '__main__':
    main()
