import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import DataLoader as open_data
from IPython import embed
from .config import Configuration


try:
    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if len(tf.config.list_physical_devices('GPU')):
        use_GPU = True
except ImportError:
    use_GPU = False


def multi_channel_audio_file_generator(filename: str,
                                       channels: int,
                                       data_snippet_idxs: int):
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
    shape = data.shape

    dataset = None
    if use_GPU:
        dataset = tf.data.Dataset.from_generator(
            multi_channel_audio_file_generator,
            args=(filename, channels, snippet_size),
            output_types=tf.float32,
            output_shapes=tf.TensorShape([None, channels]))

    return data, samplerate, channels, dataset, shape

def main():
    # run as: run as "python3 -m wavetracker.dataloader"
    file = "/home/raab/data/2023-02-09-08_16"

    cfg = Configuration()

    data, samplerate, channels, dataset, data_shape = open_raw_data(folder=file, **cfg.spectrogram)

    fig, ax = plt.subplots(int(np.ceil(data_shape[1] / 2)), 2, figsize=(20 / 2.54, 20 / 2.54), sharex='all', sharey='all')
    ax = np.hstack(ax)
    d = data[0: cfg.spectrogram['snippet_size'], :]
    fig.suptitle('Data loaded with thunderfish.DataLoader')
    for i in range(channels):
        ax[i].plot(np.arange(cfg.spectrogram['snippet_size']) / samplerate, d[:, i])
        ax[i].text(0.9, 0.9, f'{i}', transform=ax[i].transAxes, ha='center', va='center')
    plt.show()

    if use_GPU:
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
    main()