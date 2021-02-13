import os
import json
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
import soundfile

from dataset.spectogram import spectogram_configs as cfg


def get_film_clap_paths_and_labels(data_root, time_margin=0.1):
    """
    Parses the Film_clap raw data and collect audio file paths , start_times and end_times of claps
    """
    result = []
    num_claps = 0
    num_audio_files = 0
    files_per_film = defaultdict(lambda:0)
    path_to_label = json.load(open(os.path.join(data_root, 'paths_and_labels_fixed_Meron.txt')))
    print("Collecting Film-clap dataset")
    for sound_path in path_to_label:
        soundfile_name = os.path.splitext(os.path.basename(sound_path))[0]
        film_name = os.path.basename(os.path.dirname(sound_path))
        name = f"{film_name}_{soundfile_name}"
        evemt_centers_list = path_to_label[sound_path]
        assert os.path.exists(sound_path), sound_path
        start_times = [e - time_margin for e in evemt_centers_list]
        end_times = [e + time_margin for e in evemt_centers_list]
        result += [(sound_path, start_times, end_times, name)]
        num_claps += len(start_times)
        num_audio_files += 1
        files_per_film[film_name] += 1

    for film_name in files_per_film:
        print(f"\t- {film_name} has {files_per_film[film_name]}")
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
