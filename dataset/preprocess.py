import os
import pickle

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

import config as cfg


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor.

        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''

        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)

        self.melW = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform_multichannel(self, multichannel_audio):
        '''Extract feature of a multichannel audio file.

        Args:
          multichannel_audio: (samples, channels_num)

        Returns:
          feature: (channels_num, frames_num, freq_bins)
        '''

        (samples, channels_num) = multichannel_audio.shape

        feature = np.array([self.transform_singlechannel(multichannel_audio[:, m]) for m in range(channels_num)])

        return feature

    def transform_singlechannel(self, audio):
        '''Extract feature of a singlechannel audio file.

        Args:
          audio: (samples,)

        Returns:
          feature: (frames_num, freq_bins)
        '''

        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func

        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio,
            n_fft=window_size,
            hop_length=hop_size,
            window=window_func,
            center=True,
            dtype=np.complex64,
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''

        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)

        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10,
            top_db=None)

        logmel_spectrogram = logmel_spectrogram.astype(np.float32)

        return logmel_spectrogram


def read_multichannel_audio(audio_path, target_fs=None):
    """
    Read the audio samples in files and resample them to fit the desired sample ratre
    """
    (multichannel_audio, sample_rate) = soundfile.read(audio_path)
    if len(multichannel_audio.shape) == 1:
        multichannel_audio = multichannel_audio.reshape(-1, 1)
    if multichannel_audio.shape[1] != cfg.audio_channels:
        multichannel_audio = np.repeat(multichannel_audio.mean(1).reshape(-1, 1), cfg.audio_channels, axis=1)

    if target_fs is not None and sample_rate != target_fs:

        channels_num = multichannel_audio.shape[1]

        multichannel_audio = np.array(
            [librosa.resample(multichannel_audio[:, i], orig_sr=sample_rate, target_sr=target_fs) for i in range(channels_num)]
                                        ).T

    return multichannel_audio, target_fs


def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def preprocess_data(audio_path_and_labels, output_dir, output_mean_std_file):
    os.makedirs(output_dir, exist_ok=True)

    feature_extractor = LogMelExtractor(
        sample_rate=cfg.sample_rate,
        window_size=cfg.window_size,
        hop_size=cfg.hop_size,
        mel_bins=cfg.mel_bins,
        fmin=cfg.fmin,
        fmax=cfg.fmax)

    all_features = []

    for (audio_path, start_times, end_times) in tqdm(audio_path_and_labels):
        bare_name = os.path.basename(os.path.splitext(audio_path)[0])

        multichannel_audio, _ = read_multichannel_audio(audio_path=audio_path, target_fs=cfg.sample_rate)
        feature = feature_extractor.transform_multichannel(multichannel_audio)
        all_features.append(feature)

        if len(start_times) > 0:
            output_path = os.path.join(output_dir, bare_name + "_mel_features_and_labels.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump({'features': feature, 'start_times': start_times, 'end_times': end_times},
                             f, protocol=pickle.HIGHEST_PROTOCOL)

    all_features = np.concatenate(all_features, axis=1)
    mean, std = calculate_scalar_of_tensor(all_features)
    with open(output_mean_std_file, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)