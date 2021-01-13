import os
import pickle
import random
import librosa
import numpy as np
import soundfile
from tqdm import tqdm

import config as cfg
from utils import plot_debug_image


class LogMelExtractor(object):
    def __init__(self, sample_rate, nfft, window_size, hop_size, mel_bins, fmin, fmax):
        '''
        Log mel feature extractor.
        '''

        self.nfft = nfft
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)

        self.melW = librosa.filters.mel(
            sr=sample_rate,
            n_fft=nfft,
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

        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio,
            n_fft=self.nfft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            window=self.window_func,
            center=True,
            dtype=np.complex64,
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''

        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)

        # Log mel spectrogram ( in decibels )
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
    if multichannel_audio.shape[1] < cfg.audio_channels:
        print(multichannel_audio.shape[1])
        multichannel_audio = np.repeat(multichannel_audio.mean(1).reshape(-1, 1), cfg.audio_channels, axis=1)
    elif cfg.audio_channels == 1:
        multichannel_audio = multichannel_audio.mean(1).reshape(-1, 1)
    elif multichannel_audio.shape[1] > cfg.audio_channels:
        multichannel_audio = multichannel_audio[:, :cfg.audio_channels]

    if target_fs is not None and sample_rate != target_fs:

        channels_num = multichannel_audio.shape[1]

        multichannel_audio = np.array(
            [librosa.resample(multichannel_audio[:, i], orig_sr=sample_rate, target_sr=target_fs) for i in range(channels_num)]
        ).T

    return multichannel_audio


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
        sample_rate=cfg.working_sample_rate,
        nfft=cfg.NFFT,
        window_size=cfg.frame_size,
        hop_size=cfg.hop_size,
        mel_bins=cfg.mel_bins,
        fmin=cfg.mel_min_freq,
        fmax=cfg.mel_max_freq)

    all_features = []

    for (audio_path, start_times, end_times, audio_name) in tqdm(audio_path_and_labels):

        multichannel_audio = read_multichannel_audio(audio_path=audio_path, target_fs=cfg.working_sample_rate)
        feature = feature_extractor.transform_multichannel(multichannel_audio)
        all_features.append(feature)

        output_path = os.path.join(output_dir, audio_name + "_mel_features_and_labels.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({'features': feature, 'start_times': start_times, 'end_times': end_times},
                        f)

    all_features = np.concatenate(all_features, axis=1)
    mean, std = calculate_scalar_of_tensor(all_features)
    with open(output_mean_std_file, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)

    # Visualize single data sample
    (audio_path, start_times, end_times, audio_name) = random.choice(audio_path_and_labels)
    analyze_data_sample(audio_path, start_times, end_times, audio_name,
                        feature_extractor, os.path.join(os.path.dirname(output_mean_std_file), "data_sample.png"))


def analyze_data_sample(audio_path, start_times, end_times, audio_name, feature_extractor, plot_path):
    from dataset.data_generator import create_event_matrix
    org_multichannel_audio, org_sample_rate = soundfile.read(audio_path)

    multichannel_audio = read_multichannel_audio(audio_path=audio_path, target_fs=cfg.working_sample_rate)
    singlechannel_audio = multichannel_audio[:, 0]
    feature = feature_extractor.transform_singlechannel(singlechannel_audio)

    event_matrix = create_event_matrix(feature.shape[0], start_times, end_times)
    file_name = f"{os.path.basename(os.path.dirname(audio_path))}_{os.path.splitext(os.path.basename(audio_path))[0]}"
    plot_debug_image(feature, target=event_matrix, plot_path=plot_path, file_name=file_name)

    signal_time = singlechannel_audio.shape[0]/cfg.working_sample_rate
    FPS = cfg.working_sample_rate / cfg.hop_size
    print(f"Data sample analysis: {audio_name}")
    print(f"\tOriginal audio: {org_multichannel_audio.shape} sample_rate={org_sample_rate}")
    print(f"\tsingle channel audio: {singlechannel_audio.shape}, sample_rate={cfg.working_sample_rate}")
    print(f"\tSignal time is (num_samples/sample_rate)={signal_time:.1f}s")
    print(f"\tSIFT FPS is (sample_rate/hop_size)={FPS}")
    print(f"\tTotal number of frames is (FPS*signal_time)={FPS*signal_time:.1f}")
    print(f"\tEach frame covers {cfg.frame_size} samples or {cfg.frame_size / cfg.working_sample_rate:.3f} seconds "
          f"padded into {cfg.NFFT} samples and allow ({cfg.NFFT}//2+1)={cfg.NFFT // 2 + 1} frequency bins")
    print(f"\tFeatures shape: {feature.shape}")


