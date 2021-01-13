import numpy as np
import os
import pickle
from random import shuffle

import torch

import config as cfg
from dataset.common import get_film_clap_paths_and_labels, get_tau_sed_paths_and_labels
from dataset.download_tau_sed_2019 import ensure_tau_data
from dataset.preprocess import preprocess_data
from utils import human_format

cfg_descriptor = f"SaR-{human_format(cfg.working_sample_rate)}_FrS-{human_format(cfg.frame_size)}" \
                 f"_HoS-{human_format(cfg.hop_size)}_Mel-{cfg.mel_bins}_Ch-{cfg.audio_channels}"


class SpectogramGenerator(object):
    def __init__(self, features_and_labels_dir, mean_std_file, batch_size, val_descriptor, balance_classes=False, augment_data=False):
        self.batch_size = batch_size
        self.random_state = np.random.RandomState()
        self.augment_data = augment_data
        self.train_crop_size = cfg.train_crop_size

        self.train_features_list = []
        self.train_event_matrix_list = []
        self.train_index_with_event = []
        self.train_index_empty = []

        # Load data mean and std
        d = pickle.load(open(mean_std_file, 'rb'))
        self.mean = d['mean']
        self.std = d['std']

        # Split to train, test
        feature_names = os.listdir(features_and_labels_dir)
        if type(val_descriptor) == float:
            shuffle(feature_names)
            val_split = int(len(feature_names)*val_descriptor)
            self.train_feature_names = feature_names[val_split:]
            self.validate_feature_names = feature_names[:val_split]
        else:
           self.train_feature_names = []
           self.validate_feature_names = []
           for name in feature_names:
               if val_descriptor in name:
                   self.validate_feature_names.append(name)
               else:
                   self.train_feature_names.append(name)

        # Load training feature and targets
        frame_index = 0
        for feature_name in self.train_feature_names:
            data = pickle.load(open(os.path.join(features_and_labels_dir, feature_name), 'rb'))
            feature = data['features']
            event_matrix = create_event_matrix(feature.shape[1], data['start_times'], data['end_times'])

            frames_num = feature.shape[1]
            '''Number of frames of the log mel spectrogram of an audio 
            recording. May be different from file to file'''

            possible_start_indices = np.arange(frame_index, frame_index + frames_num - self.train_crop_size)
            frame_index += frames_num

            # Append data
            self.train_features_list.append(feature)
            self.train_event_matrix_list.append(event_matrix)

            # Slpit data to chunks which contain an event and such that are not
            indices_with_event = np.zeros(possible_start_indices.shape, dtype=bool)
            for i in np.where(event_matrix > 0)[0]:
                indices_with_event[i - self.train_crop_size: i] = True
            self.train_index_with_event += possible_start_indices[np.where(indices_with_event)[0]].tolist()
            self.train_index_empty += possible_start_indices[np.where(indices_with_event == False)[0]].tolist()

        self.train_features = np.concatenate(self.train_features_list, axis=1)
        self.train_event_matrix = np.concatenate(self.train_event_matrix_list, axis=0)

        # Balance classes in train data
        self.random_state.shuffle(self.train_index_with_event)
        self.random_state.shuffle(self.train_index_empty)
        if balance_classes:
            size = min(len(self.train_index_with_event), len(self.train_index_empty))
            self.train_index_with_event = self.train_index_with_event[:size]
            self.train_index_empty = self.train_index_empty[:size]
        self.train_start_indices = np.concatenate((self.train_index_empty, self.train_index_with_event))
        self.random_state.shuffle(self.train_start_indices)

        # Load validation feature and targets
        self.validate_features_list = []
        self.validate_event_matrix_list = []
        for feature_name in self.validate_feature_names:
            data = pickle.load(open(os.path.join(features_and_labels_dir, feature_name), 'rb'))
            feature = data['features']
            event_matrix = create_event_matrix(feature.shape[1], data['start_times'], data['end_times'])

            self.validate_features_list.append(feature)
            self.validate_event_matrix_list.append(event_matrix)

        self.pointer = 0

        print(f"Data generator initiated with {len(self.train_feature_names)} train samples "
              f"totaling {len(self.train_event_matrix) / cfg.frames_per_second:.1f} seconds "
              f"and {len(self.validate_feature_names)} val samples "
              f"totaling {len(np.concatenate(self.validate_event_matrix_list, axis=0)) / cfg.frames_per_second:.1f} seconds")

    def generate_train(self):
        '''
        Generate mini-batch data for training.
        Samples a start index and crops a self.train_crop_size long segment from the concatenated featues
        Returns:
          batch_data_dict: dict containing feature, event, elevation and azimuth
        '''
        while True:
            # Reset pointer
            if self.pointer > len(self.train_start_indices) - self.batch_size:
                self.pointer = 0
                self.random_state.shuffle(self.train_start_indices)

            # Get batch indexes
            batch_indexes = self.train_start_indices[self.pointer: self.pointer + self.batch_size]

            data_indexes = batch_indexes[:, None] + np.arange(self.train_crop_size)

            self.pointer += self.batch_size

            batch_feature = self.train_features[:, data_indexes]
            batch_event_matrix = self.train_event_matrix[data_indexes]

            if self.augment_data:
                # batch_feature, batch_event_matrix = self.augment_batch_mix_samples(batch_feature, batch_event_matrix)
                batch_feature, batch_event_matrix = self.augment_batch_add_noise(batch_feature, batch_event_matrix)

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

    def augment_batch_add_noise(self, batch_feature, batch_event_matrix):
        r = np.random.rand()
        if r > 0.5:
            noise_var = 0.001 + (r + 0.5) * (0.005 - 0.001)
            batch_feature += np.random.normal(0, noise_var, size=batch_feature.shape)
        return batch_feature, batch_event_matrix

    def augment_batch_mix_samples(self, batch_feature, batch_event_matrix):
        """
        Augment a samples by mixing its features and labesl with other train samples
        """
        number_of_augmentations = np.random.choice([0, 1, 2, 3], 1, p=[0.6, 0.25, 0.1, 0.05])[0]
        for i in range(number_of_augmentations):
            random_pointer = np.random.randint(len(self.train_start_indices) - self.batch_size + 1)
            new_batch_indexes = self.train_start_indices[random_pointer: random_pointer + self.batch_size]
            new_data_indexes = new_batch_indexes[:, None] + np.arange(self.train_crop_size)
            new_batch_feature = self.train_features[:, new_data_indexes]
            new_batch_event_matrix = self.train_event_matrix[new_data_indexes]
            new_batch_feature = self.transform(new_batch_feature)

            batch_feature += new_batch_feature
            batch_event_matrix = np.maximum(batch_event_matrix, new_batch_event_matrix)
        batch_feature /= (number_of_augmentations + 1)

        return batch_feature, batch_event_matrix


def create_event_matrix(frames_num, start_times, end_times):
    """
    Create a per-frame classification matrix whith 1 in times specified by start/end times and 0 elsewhere
    """
    # Researve space data
    event_matrix = np.zeros((frames_num, cfg.classes_num))

    for n in range(len(start_times)):
        start_frame = int(round(start_times[n] * cfg.frames_per_second))
        end_frame = int(round(end_times[n]* cfg.frames_per_second)) + 1

        event_matrix[start_frame: end_frame] = 1

    return event_matrix


def preprocess_tau_sed_data(data_dir, mode='eval', force_preprocess=False):
    """
    Download, extract and preprocess the tau sed datset
    force_preprocess: Force the preprocess phase to repeate: usefull in case you change the preprocess parameters
    """
    global cfg_descriptor
    cfg_descriptor = f"{cfg_descriptor}_C-{'-'.join(cfg.tau_sed_labels)}"

    ambisonic_2019_data_dir = f"{data_dir}/Tau_sound_events_2019"
    audio_dir, meta_data_dir = ensure_tau_data(ambisonic_2019_data_dir, mode=mode)

    processed_data_dir = os.path.join(ambisonic_2019_data_dir, f"processed_{cfg_descriptor}")
    features_and_labels_dir = f"{processed_data_dir}/features_and_labels_{mode}"
    features_mean_std_file = f"{processed_data_dir}/mel_features_mean_std_{mode}.pkl"
    if not os.path.exists(features_and_labels_dir) or force_preprocess:
        print("preprocessing raw data")
        audio_paths_and_labels = get_tau_sed_paths_and_labels(audio_dir, meta_data_dir)
        preprocess_data(audio_paths_and_labels, output_dir=features_and_labels_dir, output_mean_std_file=features_mean_std_file)
    else:
        print("Using existing mel features")
    return features_and_labels_dir, features_mean_std_file, "TAU"


def preprocess_film_clap_data(data_dir, force_preprocess=False):
    """
    Preprocess and Creates a data generator for the film_clap dataset
    """
    film_clap_dir = os.path.join(data_dir, 'Film_take_clap')
    audio_and_labels_dir = os.path.join(film_clap_dir, 'raw')
    global cfg_descriptor
    cfg_descriptor = f"{cfg_descriptor}_tm-{cfg.time_margin}"
    if not os.path.exists(film_clap_dir):
        raise Exception("You should get you own dataset...")
    features_and_labels_dir = f"{film_clap_dir}/processed_{cfg_descriptor}/features_and_labels"
    features_mean_std_file = f"{film_clap_dir}/processed_{cfg_descriptor}/mel_features_mean_std.pkl"
    if not os.path.exists(features_and_labels_dir) or force_preprocess:
        print("preprocessing raw data")
        audio_paths_and_labels = get_film_clap_paths_and_labels(audio_and_labels_dir, time_margin=cfg.time_margin)
        preprocess_data(audio_paths_and_labels, output_dir=features_and_labels_dir, output_mean_std_file=features_mean_std_file)
    else:
        print("Using existing mel features")
    return features_and_labels_dir, features_mean_std_file, "FlimClap"