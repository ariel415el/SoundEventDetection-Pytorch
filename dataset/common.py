import os
import pickle
from math import ceil
import numpy as np
import pandas as pd

from dataset.spectogram_features import spectogram_configs as cfg


def get_film_clap_paths_and_labels(data_root, time_margin=0.1):
    """
    Parses the Film_clap raw data and collect audio file paths , start_times and end_times of claps
    """
    result = []
    num_claps = 0
    num_audio_files = 0
    dataset_sizes = 0
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
        print(f"Dataset : {film_name} has {len(result) - dataset_sizes} samples")
        dataset_sizes = len(result)
    print(f"Film clap dataset contains {num_audio_files} audio files with {num_claps} clap incidents")
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


def split_to_frames(signal, frame_length, overlap_length):
    """
    Split the signal into overlapping frames. pads the signal if necessary for division.
    Here a new frame of size "frame_length" samples starts every "overlap_length" samples
    :param frame_length: how many samples in each frames
    :param overlap_length: overlapping samples
    :return: numpy array of size num_frames, frame_length
    """
    num_frames = ceil(len(signal) / float(overlap_length))
    pad_size = (num_frames - 1) * overlap_length + frame_length - len(signal)
    padded_signal = np.append(signal, np.zeros(pad_size))

    # extract overlapping frames:
    frame_offsets = np.tile(np.arange(0, num_frames) * overlap_length, (frame_length, 1)).T
    frame_indices = np.tile(np.arange(0, frame_length), (num_frames, 1))
    frame_indices += frame_offsets

    return padded_signal[frame_indices]
