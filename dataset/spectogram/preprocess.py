import os
import pickle
import random
import librosa
import numpy as np
import soundfile
from tqdm import tqdm

import dataset.spectogram.spectogram_configs as cfg
from dataset.dataset_utils import read_multichannel_audio
from utils.plot_utils import plot_sample_features

MEL_FILTER_BANK_MATRIX = librosa.filters.mel(
    sr=cfg.working_sample_rate,
    n_fft=cfg.NFFT,
    n_mels=cfg.mel_bins,
    fmin=cfg.mel_min_freq,
    fmax=cfg.mel_max_freq).T


def multichannel_stft(multichannel_signal):
    (samples, channels_num) = multichannel_signal.shape
    features = []
    for c in range(channels_num):
        complex_spectogram = librosa.core.stft(
                            y=multichannel_signal[:, c],
                            n_fft=cfg.NFFT,
                            win_length=cfg.frame_size,
                            hop_length=cfg.hop_size,
                            window=np.hanning(cfg.frame_size),
                            center=True,
                            dtype=np.complex64,
                            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
        features.append(complex_spectogram)
    return np.array(features)


def multichannel_complex_to_log_mel(multichannel_complex_spectogram):
    multichannel_power_spectogram = np.abs(multichannel_complex_spectogram) ** 2
    multichannel_mel_spectogram = np.dot(multichannel_power_spectogram, MEL_FILTER_BANK_MATRIX)
    multichannel_logmel_spectogram = librosa.core.power_to_db(multichannel_mel_spectogram,
                                                              ref=1.0, amin=1e-10, top_db=None).astype(np.float32)

    return multichannel_logmel_spectogram


def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def preprocess_data(audio_path_and_labels, output_dir, output_mean_std_file, preprocess_mode='logMel'):
    print("Preprocessing collected data")
    os.makedirs(output_dir, exist_ok=True)

    all_features = []

    for (audio_path, start_times, end_times, audio_name) in tqdm(audio_path_and_labels):
        multichannel_waveform = read_multichannel_audio(audio_path=audio_path, target_fs=cfg.working_sample_rate)
        feature = multichannel_stft(multichannel_waveform)
        if preprocess_mode == 'logMel':
            feature = multichannel_complex_to_log_mel(feature)
        all_features.append(feature)

        output_path = os.path.join(output_dir, audio_name + f"_{preprocess_mode}_features_and_labels.pkl")
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
                        os.path.join(os.path.dirname(output_mean_std_file), "data_sample.png"))


def analyze_data_sample(audio_path, start_times, end_times, audio_name, plot_path):
    """
    A debug function that plots a single sample and analyzes how the spectogram configuration affect the feature final size
    """
    from dataset.spectogram.spectograms_dataset import create_event_matrix
    org_multichannel_audio, org_sample_rate = soundfile.read(audio_path)

    multichannel_audio = read_multichannel_audio(audio_path=audio_path, target_fs=cfg.working_sample_rate)
    feature = multichannel_stft(multichannel_audio)
    feature = multichannel_complex_to_log_mel(feature)  # (channels, frames, mel_bins)
    event_matrix = create_event_matrix(feature.shape[1], start_times, end_times)
    plot_sample_features(feature, mode='spectogram', target=event_matrix, plot_path=plot_path, file_name=audio_name)

    signal_time = multichannel_audio.shape[0]/cfg.working_sample_rate
    FPS = cfg.working_sample_rate / cfg.hop_size
    print(f"Data sample analysis: {audio_name}")
    print(f"\tOriginal audio: {org_multichannel_audio.shape} sample_rate={org_sample_rate}")
    print(f"\tsingle channel audio: {multichannel_audio.shape}, sample_rate={cfg.working_sample_rate}")
    print(f"\tSignal time is (num_samples/sample_rate)={signal_time:.1f}s")
    print(f"\tSIFT FPS is (sample_rate/hop_size)={FPS}")
    print(f"\tTotal number of frames is (FPS*signal_time)={FPS*signal_time:.1f}")
    print(f"\tEach frame covers {cfg.frame_size} samples or {cfg.frame_size / cfg.working_sample_rate:.3f} seconds "
          f"padded into {cfg.NFFT} samples and allow ({cfg.NFFT}//2+1)={cfg.NFFT // 2 + 1} frequency bins")
    print(f"\tFeatures shape: {feature.shape}")


