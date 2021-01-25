import librosa
import numpy as np
import os
import pickle
from random import shuffle

import torch
from torch.utils.data import Dataset

import dataset.spectogram_features.spectogram_configs
import dataset.spectogram_features.spectogram_configs as cfg
from dataset.dataset_utils import get_film_clap_paths_and_labels, get_tau_sed_paths_and_labels
from dataset.download_tau_sed_2019 import ensure_tau_data
from dataset.spectogram_features.preprocess import preprocess_data, multichannel_complex_to_log_mel

class SpectogramDataset(Dataset):
    def __init__(self, features_and_labels_dir, mean_std_file, val_descriptor,
                 balance_classes=False, augment_data=False, preprocessed_mode='Complex'):
        """
        This dataset loads crops of the entire concatenated features of the data
        Args:
            features_and_labels_dir:
            mean_std_file: mean and std of the saved features # TODO: currently these are different for Complex histograms
            val_descriptor: How to split the data; float for percentage and string for specifing substring in desired files
            balance_classes: Limit the number of crops with no event to match the number of crops with events
            augment_data: 1. Add noise. 2. Mix STFT spectograms of multiple samples before converting to LogMel
            preprocessed_mode: defines whether if the preprocess phase included converting to LogMel or only STFT
        """
        assert preprocessed_mode in ['logMel', 'Complex'], "Spectogram type should be either logmel or complex"
        assert not (preprocessed_mode == 'logMel' and augment_data), "Can't perform augmentation in logMel spectograms"
        self.preprocessed_mode = preprocessed_mode
        self.random_state = np.random.RandomState()
        self.augment_data = augment_data
        self.train_crop_size = cfg.train_crop_size

        # Load data mean and std
        d = pickle.load(open(mean_std_file, 'rb'))
        self.mean = d['mean']
        self.std = d['std']

        train_feature_paths, self.val_feature_paths = _split_train_val(features_and_labels_dir, val_descriptor)

        self.train_features, self.train_event_matrix, self.train_start_indices = _read_train_data_to_memory(train_feature_paths,
                                                                                                            cfg.train_crop_size,
                                                                                                            balance_classes)

        self.val_features_list, self.val_event_matrix_list = _read_validation_data_to_memory(self.val_feature_paths)

        print(f"Data generator initiated with {len(train_feature_paths)} train samples "
              f"totaling {len(self.train_event_matrix) / cfg.frames_per_second:.1f} seconds "
              f"and {len(self.val_feature_paths)} val samples "
              f"totaling {len(np.concatenate(self.val_event_matrix_list, axis=0)) / cfg.frames_per_second:.1f} seconds")

    def __len__(self):
        return len(self.train_event_matrix)

    def __getitem__(self, idx):
        '''
        Generate mini-batch data for training.
        Samples a start index and crops a self.train_crop_size long segment from the concatenated featues
        Returns:
          batch_data_dict: dict containing feature, event, elevation and azimuth
        '''

        data_indexes = np.arange(self.train_crop_size) + self.train_start_indices[idx]

        features = self.train_features[:, data_indexes]
        event_matrix = self.train_event_matrix[data_indexes]

        if self.augment_data:
            feature, event_matrix = self.augment_mix_samples(features, event_matrix)
            feature, event_matrix = self.augment_add_noise(feature, event_matrix)

        # Transform data
        features = self.transform(features)

        return torch.from_numpy(features), torch.from_numpy(event_matrix)

    def get_validation_sampler(self, max_validate_num=None):
        feature_names = self.val_feature_paths
        features_list = self.val_features_list
        event_matrix_list = self.val_event_matrix_list

        validate_num = len(feature_names)

        for n in range(validate_num):
            if n == max_validate_num:
                break

            name = os.path.basename(os.path.splitext(feature_names[n])[0])
            feature = features_list[n]
            event_matrix = event_matrix_list[n]

            feature = self.transform(feature)

            features = feature[None, :, :, :]  # ( batch_size=1, channels_num, frames_num, mel_bins)
            event_matrix = event_matrix[None, :, :]  # (batch_size=1, frames_num, mel_bins)
            '''The None above indicates using an entire audio recording as 
            input and batch_size=1 in inference'''

            yield torch.from_numpy(features), torch.from_numpy(event_matrix), name

    def transform(self, x):
        x = (x - self.mean) / self.std

        if self.preprocessed_mode == 'logMel':
            return x
        else:  # If the preprocessed spectograms are saved as raw complex spectograms transform them into logMel
            return multichannel_complex_to_log_mel(x)

    def augment_add_noise(self, batch_feature, batch_event_matrix):
        # TODO these number are fit to noise added to waveform and not spectogram
        r = np.random.rand()
        if r > 0.5:
            noise_var = 0.001 + (r + 0.5) * (0.005 - 0.001)
            batch_feature += np.random.normal(0, noise_var, size=batch_feature.shape)
        return batch_feature, batch_event_matrix

    def augment_mix_samples(self, feature, event_matrix):
        """
        Augment a samples by mixing its features and labesl with other train samples
        """
        number_of_augmentations = np.random.choice([0, 1, 2, 3], 1, p=[0.6, 0.25, 0.1, 0.05])[0]
        for i in range(number_of_augmentations):
            random_pointer = np.random.randint(len(self.train_start_indices) + 1)
            new_data_indexes = np.arange(self.train_crop_size) + self.train_start_indices[random_pointer]
            new_feature = self.train_features[:, new_data_indexes]
            new_event_matrix = self.train_event_matrix[new_data_indexes]

            feature += new_feature
            event_matrix = np.maximum(event_matrix, new_event_matrix)
        feature /= (number_of_augmentations + 1)

        return feature, event_matrix


def _split_train_val(data_dir, val_descriptor):
    # Split to train, test
    feature_names = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    if type(val_descriptor) == float:
        shuffle(feature_names)
        val_split = int(len(feature_names) * val_descriptor)
        train_feature_names = feature_names[val_split:]
        validate_feature_names = feature_names[:val_split]
    else:
        train_feature_names = []
        validate_feature_names = []
        for name in feature_names:
            if val_descriptor in name:
                validate_feature_names.append(name)
            else:
                train_feature_names.append(name)

    return train_feature_names, validate_feature_names


def _read_train_data_to_memory(train_feature_paths, crop_size, balance_classes=False):
    """
    Creates a list with all spectograms conatenated to each other so that one can sample random crops over them by choosing
    from a set of start indices.
    """
    # Load training feature and targets
    frame_index = 0

    train_features_list = []
    train_event_matrix_list = []
    train_index_with_event = []
    train_index_empty = []

    for feature_path in train_feature_paths:
        data = pickle.load(open(feature_path, 'rb'))
        feature = data['features']
        event_matrix = create_event_matrix(feature.shape[1], data['start_times'], data['end_times'])

        frames_num = feature.shape[1]
        '''Number of frames of the (log mel / complex) spectrogram of an audio 
        recording. May be different from file to file'''

        possible_start_indices = np.arange(frame_index, frame_index + frames_num - crop_size)
        frame_index += frames_num

        # Append data
        train_features_list.append(feature)
        train_event_matrix_list.append(event_matrix)

        # Slpit data to chunks which contain an event and such that are not
        indices_with_event = np.zeros(possible_start_indices.shape, dtype=bool)
        for i in np.where(event_matrix > 0)[0]:
            indices_with_event[i - crop_size: i] = True
        train_index_with_event += possible_start_indices[np.where(indices_with_event)[0]].tolist()
        train_index_empty += possible_start_indices[np.where(indices_with_event == False)[0]].tolist()

    train_features = np.concatenate(train_features_list, axis=1)
    train_event_matrix = np.concatenate(train_event_matrix_list, axis=0)

    # Balance classes in train data
    np.random.shuffle(train_index_with_event)
    np.random.shuffle(train_index_empty)
    if balance_classes:
        size = min(len(train_index_with_event), len(train_index_empty))
        train_index_with_event = train_index_with_event[:size]
        train_index_empty = train_index_empty[:size]
    train_start_indices = np.concatenate((train_index_empty, train_index_with_event))
    np.random.shuffle(train_start_indices)

    return train_features, train_event_matrix, train_start_indices


def _read_validation_data_to_memory(feature_paths):
    # Load validation feature and targets
    features_list = []
    event_matrix_list = []
    for feature_path in feature_paths:
        data = pickle.load(open(feature_path, 'rb'))
        feature = data['features']
        event_matrix = create_event_matrix(feature.shape[1], data['start_times'], data['end_times'])

        features_list.append(feature)
        event_matrix_list.append(event_matrix)

    return features_list, event_matrix_list


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


def preprocess_tau_sed_data(data_dir, preprocess_mode, force_preprocess=False, fold_name='eval'):
    """
    Download, extract and preprocess the tau sed datset
    force_preprocess: Force the preprocess phase to repeate: usefull in case you change the preprocess parameters
    """
    cfg.cfg_descriptor += f"_C-{'-'.join(cfg.tau_sed_labels)}"

    ambisonic_2019_data_dir = f"{data_dir}/Tau_sound_events_2019"
    audio_dir, meta_data_dir = ensure_tau_data(ambisonic_2019_data_dir, fold_name=fold_name)

    processed_data_dir = os.path.join(ambisonic_2019_data_dir, f"processed_{dataset.spectogram_features.spectogram_configs.cfg_descriptor}")
    features_and_labels_dir = f"{processed_data_dir}/{preprocess_mode}-features_and_labels_{fold_name}"
    features_mean_std_file = f"{processed_data_dir}/{preprocess_mode}-features_mean_std_{fold_name}.pkl"
    if not os.path.exists(features_and_labels_dir) or force_preprocess:
        audio_paths_and_labels = get_tau_sed_paths_and_labels(audio_dir, meta_data_dir)
        preprocess_data(audio_paths_and_labels, output_dir=features_and_labels_dir,
                        output_mean_std_file=features_mean_std_file, preprocess_mode=preprocess_mode)
    else:
        print("Using existing mel features")
    return features_and_labels_dir, features_mean_std_file, "TAU"


def preprocess_film_clap_data(data_dir, preprocessed_mode, force_preprocess=False):
    """
    Preprocess and Creates a data generator for the film_clap dataset
    """
    film_clap_dir = os.path.join(data_dir, 'FilmClap')
    audio_and_labels_dir = os.path.join(film_clap_dir, 'raw')
    cfg.cfg_descriptor += f"_tm-{cfg.time_margin}"
    if not os.path.exists(film_clap_dir):
        raise Exception("You should get you own dataset...")
    features_and_labels_dir = f"{film_clap_dir}/processed/{dataset.spectogram_features.spectogram_configs.cfg_descriptor}/{preprocessed_mode}-features_and_labels"
    features_mean_std_file = f"{film_clap_dir}/processed/{dataset.spectogram_features.spectogram_configs.cfg_descriptor}/{preprocessed_mode}-features_mean_std.pkl"
    if not os.path.exists(features_and_labels_dir) or force_preprocess:
        print("preprocessing raw data")
        audio_paths_and_labels = get_film_clap_paths_and_labels(audio_and_labels_dir, time_margin=cfg.time_margin)
        preprocess_data(audio_paths_and_labels, output_dir=features_and_labels_dir,
                        output_mean_std_file=features_mean_std_file, preprocess_mode=preprocessed_mode)
    else:
        print("Using existing mel features")
    return features_and_labels_dir, features_mean_std_file, "FlimClap"