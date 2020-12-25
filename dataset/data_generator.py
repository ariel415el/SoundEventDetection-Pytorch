import numpy as np
import os
import pickle
from random import shuffle

import pandas as pd
import torch

import config as cfg
from dataset.download_tau_sed_2019 import download_foa_data, extract_foa_data
from dataset.preprocess import preprocess_data
from utils import human_format
cfg_descriptor = f"SR-{human_format(cfg.working_sample_rate)}_WS-{human_format(cfg.window_size)}_HS-{human_format(cfg.hop_size)}"


def create_event_matrix(frames_num, start_times, end_times):
    """
    Create a per-frame classification matrix whith 1 in times specified by start/end times and 0 elsewhere
    """
    # Researve space data
    event_matrix = np.zeros((frames_num, cfg.classes_num))

    for n in range(len(start_times)):
        start_frame = int(round(start_times[n] * cfg.frames_per_second))
        end_frame = int(round(end_times[n] * cfg.frames_per_second)) + 1

        event_matrix[start_frame: end_frame] = 1

    return event_matrix


class DataGenerator(object):
    def __init__(self, features_and_labels_dir, mean_std_file, batch_size, val_perc=0.1, seed=1234):
        d = pickle.load(open(mean_std_file, 'rb'))
        self.mean = d['mean']
        self.std = d['std']
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)

        self.frames_per_second = cfg.frames_per_second
        self.classes_num = cfg.classes_num
        self.lb_to_idx = cfg.lb_to_idx
        self.time_steps = cfg.time_steps

        # Split to train, test
        feature_names = os.listdir(features_and_labels_dir)
        shuffle(feature_names)
        val_split = int(len(feature_names)*val_perc)
        self.train_feature_names = feature_names[val_split:]
        self.validate_feature_names = feature_names[:val_split]
        print(f"Data generator initiated with {len(feature_names) - val_split} train and {val_split} val images")

        self.train_features_list = []
        self.train_event_matrix_list = []
        self.train_index_array_list = []
        frame_index = 0

        # Load training feature and targets
        for feature_name in self.train_feature_names:
            data = pickle.load(open(os.path.join(features_and_labels_dir, feature_name), 'rb'))
            feature = data['features']
            event_matrix = create_event_matrix(feature.shape[1], data['start_times'], data['end_times'])

            frames_num = feature.shape[1]
            '''Number of frames of the log mel spectrogram of an audio 
            recording. May be different from file to file'''

            index_array = np.arange(frame_index, frame_index + frames_num - self.time_steps)
            frame_index += frames_num

            # Append data
            self.train_features_list.append(feature)
            self.train_event_matrix_list.append(event_matrix)
            self.train_index_array_list.append(index_array)

        self.train_features = np.concatenate(self.train_features_list, axis=1)
        self.train_event_matrix = np.concatenate(self.train_event_matrix_list, axis=0)
        self.train_index_array = np.concatenate(self.train_index_array_list, axis=0)

        # Load validation feature and targets
        self.validate_features_list = []
        self.validate_event_matrix_list = []

        for feature_name in self.validate_feature_names:
            data = pickle.load(open(os.path.join(features_and_labels_dir, feature_name), 'rb'))
            feature = data['features']
            event_matrix = create_event_matrix(feature.shape[1], data['start_times'], data['end_times'])

            self.validate_features_list.append(feature)
            self.validate_event_matrix_list.append(event_matrix)

        self.random_state.shuffle(self.train_index_array)
        self.pointer = 0

    def generate_train(self):
        '''Generate mini-batch data for training.

        Returns:
          batch_data_dict: dict containing feature, event, elevation and azimuth
        '''

        while True:
            # Reset pointer
            if self.pointer >= len(self.train_index_array):
                self.pointer = 0
                self.random_state.shuffle(self.train_index_array)

            # Get batch indexes
            batch_indexes = self.train_index_array[self.pointer: self.pointer + self.batch_size]

            data_indexes = batch_indexes[:, None] + np.arange(self.time_steps)

            self.pointer += self.batch_size

            batch_feature = self.train_features[:, data_indexes]
            batch_event_matrix = self.train_event_matrix[data_indexes]

            # Transform data
            batch_feature = self.transform(batch_feature)

            yield torch.from_numpy(batch_feature), torch.from_numpy(batch_event_matrix)

    def generate_validate(self, data_type, max_validate_num=None):
        '''Generate feature and targets of a single audio file.

        Args:
          data_type: 'train' | 'validate'
          max_validate_num: None | int, maximum iteration to run to speed up
              evaluation

        Returns:
          batch_data_dict: dict containing feature, event, elevation and azimuth
        '''

        if data_type == 'train':
            feature_names = self.train_feature_names
            features_list = self.train_features_list
            event_matrix_list = self.train_event_matrix_list

        elif data_type == 'validate':
            feature_names = self.validate_feature_names
            features_list = self.validate_features_list
            event_matrix_list = self.validate_event_matrix_list

        else:
            raise Exception('Incorrect argument!')

        validate_num = len(feature_names)

        for n in range(validate_num):
            if n == max_validate_num:
                break

            name = os.path.splitext(feature_names[n])[0]
            feature = features_list[n]
            event_matrix = event_matrix_list[n]

            feature = self.transform(feature)

            features = feature[:, None, :, :]  # (channels_num, batch_size=1, frames_num, mel_bins)
            event_matrix = event_matrix[None, :, :]  # (batch_size=1, frames_num, mel_bins)
            '''The None above indicates using an entire audio recording as 
            input and batch_size=1 in inference'''

            yield torch.from_numpy(features), torch.from_numpy(event_matrix), name

    def transform(self, x):
        return (x - self.mean) / self.std


def get_film_clap_paths_and_labels(data_root):
    """
    Parses the Film_clap raw data and collect audio file paths , start_times and end_times of claps
    """
    time_margin = 0.5
    result = []
    num_claps = 0
    num_audio_files = 0
    for film_name in os.listdir(data_root):
        dirpath = os.path.join(data_root, film_name)
        csv_files = [os.path.join(dirpath, x) for x in os.listdir(dirpath) if x.endswith('.csv')]
        if film_name == "Meron" or len(csv_files) != 1:
            continue
        df = pd.read_csv(csv_files[0], sep=',')
        clip_names = np.unique(df['Clip Name'])
        done_names = set()
        for name in clip_names:
            if name not in done_names:
                done_names.add(name)
                soundfile_path = os.path.join(dirpath, name)
                if os.path.exists(soundfile_path):
                    relevant_df = df.loc[df['Clip Name'] == name]
                    start_times = [row[1] - time_margin for i, row in relevant_df.iterrows()]
                    end_times = [row[1] + time_margin for i, row in relevant_df.iterrows()]
                    result += [(soundfile_path,start_times, end_times)]
                    num_claps += len(start_times)
                    num_audio_files += 1
    print(f"Film clap dataset contains {num_audio_files} audio files with {num_claps} clap insidents")
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
        results += [(audio_path, start_times, end_times)]

    return results


def get_tau_sed_generator(data_dir, batch_size, train_or_eval='eval', force_preprocess=False):
    """
    Download, extract and preprocess the tau sed datset
    force_preprocess: Force the preprocess phase to repeate: usefull in case you change the preprocess parameters
    """
    ambisonic_2019_data_dir = f"{data_dir}/Tau_sound_events_2019"
    zipped_data_dir = os.path.join(ambisonic_2019_data_dir, 'zipped')
    extracted_data_dir= os.path.join(ambisonic_2019_data_dir, 'raw')
    processed_data_dir= os.path.join(ambisonic_2019_data_dir, f"processed_{cfg_descriptor}")
    audio_dir = f"{extracted_data_dir}/foa_{train_or_eval}"
    meda_data_dir = f"{extracted_data_dir}/metadata_{train_or_eval}"

    # Download and extact data
    if not os.path.exists(zipped_data_dir):
        print("Downloading zipped data")
        download_foa_data(zipped_data_dir, eval_only=train_or_eval == 'eval')
    if not os.path.exists(audio_dir):
        print("Extracting raw data")
        extract_foa_data(zipped_data_dir, extracted_data_dir, eval_only=train_or_eval == 'eval')
    else:
        print("Using existing raw data")

    # Preprocess data: create mel feautes and labels
    features_and_labels_dir = f"{processed_data_dir}/features_and_labels_{train_or_eval}"
    features_mean_std_file = f"{processed_data_dir}/mel_features_mean_std_{train_or_eval}.pkl"
    if not os.path.exists(features_and_labels_dir) or force_preprocess:
        print("preprocessing raw data")
        audio_paths_and_labels = get_tau_sed_paths_and_labels(audio_dir, meda_data_dir)
        preprocess_data(audio_paths_and_labels, output_dir=features_and_labels_dir, output_mean_std_file=features_mean_std_file)
    else:
        print("Using existing mel features")
    return DataGenerator(features_and_labels_dir, features_mean_std_file, batch_size)


def get_film_clap_generator(data_dir, batch_size, force_preprocess=False):
    """
    Preprocess and Creates a data generator for the film_clap dataset
    """
    if not os.path.exists(data_dir):
        raise Exception("You should get you own dataset...")
    features_and_labels_dir = f"{data_dir}/processed_{cfg_descriptor}/features_and_labels"
    features_mean_std_file = f"{data_dir}/processed_{cfg_descriptor}/mel_features_mean_std.pkl"
    if not os.path.exists(features_and_labels_dir) or force_preprocess:
        print("preprocessing raw data")
        audio_paths_and_labels = get_film_clap_paths_and_labels(os.path.join(data_dir, 'raw'))
        preprocess_data(audio_paths_and_labels, output_dir=features_and_labels_dir, output_mean_std_file=features_mean_std_file)
    else:
        print("Using existing mel features")
    return DataGenerator(features_and_labels_dir, features_mean_std_file, batch_size)