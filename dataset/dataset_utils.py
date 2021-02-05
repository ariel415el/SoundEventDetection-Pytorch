import os
import pickle

import librosa
import numpy as np
import pandas as pd
import soundfile

from dataset.spectogram_features import spectogram_configs as cfg


def get_film_clap_paths_and_labels(data_root, time_margin=0.1):
    """
    Parses the Film_clap raw data and collect audio file paths , start_times and end_times of claps
    """
    result = []
    num_claps = 0
    num_audio_files = 0
    dataset_sizes = 0
    print("Collecting Film-clap dataset")
    for film_name in os.listdir(data_root):
        dirpath = os.path.join(data_root, film_name)
        meta_data_pickle = os.path.join(dirpath, f"{film_name}_parsed.pkl")

        meta_data = pickle.load(open(meta_data_pickle, 'rb'))
        for sounfile_name, evetnts_list in meta_data.items():
            soundfile_path = os.path.join(dirpath, sounfile_name)
            assert os.path.exists(soundfile_path), soundfile_path
            start_times = [e - time_margin for e in evetnts_list]
            end_times = [e + time_margin for e in evetnts_list]
            name = f"{film_name}_{os.path.splitext(os.path.basename(soundfile_path))[0]}"
            result += [(soundfile_path, start_times, end_times, name)]
            num_claps += len(start_times)
            num_audio_files += 1
        print(f"\t- {film_name} has {len(result) - dataset_sizes} samples")
        dataset_sizes = len(result)
    print(f"\tFilm clap dataset contains {num_audio_files} audio files with {num_claps} clap incidents")
    return result


def get_tau_sed_paths_and_labels(audio_dir, labels_data_dir):
    """
    Parses the Tau_sed raw data and collect audio file paths, start_times and end_times of claps
    """
    results = []
    for audio_fname in os.listdir(audio_dir):
        bare_name = os.path.splitext(audio_fname)[0]

        audio_path = os.path.join(audio_dir, audio_fname)

        df = pd.read_csv(os.path.join(labels_data_dir, bare_name + ".csv"), sep=',')
        relevant_classes = [i for i in range(len(df['sound_event_recording'].values))
                            if df['sound_event_recording'].values[i] in cfg.tau_sed_labels]

        start_times, end_times = df['start_time'].values[relevant_classes], df['end_time'].values[relevant_classes]

        results += [(audio_path, start_times, end_times, bare_name)]

    return results


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