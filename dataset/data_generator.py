import numpy as np
import os
import pickle
from random import shuffle, choice

import pandas as pd
import torch

import config as cfg
from dataset.download_tau_sed_2019 import download_foa_data, extract_foa_data
from dataset.preprocess import preprocess_data
from utils import human_format

cfg_descriptor = f"SaR-{human_format(cfg.working_sample_rate)}_FrS-{human_format(cfg.frame_size)}" \
                 f"_HoS-{human_format(cfg.hop_size)}_Mel-{cfg.mel_bins}"


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


class DataGenerator(object):
    def __init__(self, features_and_labels_dir, mean_std_file, batch_size, val_descriptor, balance_classes=False, augment_data=False, seed=1234):
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
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
            indices_with_event = np.zeros(possible_start_indices.shape, dtype=bool)
            for i in np.where(event_matrix > 0)[0]:
                indices_with_event[i - self.train_crop_size: i] = True
            self.train_index_with_event += np.where(indices_with_event)[0].tolist()
            self.train_index_empty += np.where(indices_with_event == False)[0].tolist()

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
                batch_feature, batch_event_matrix = self.augment_batch_mix_samples(batch_feature, batch_event_matrix)
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
        if r > 0.2:
            batch_feature += np.random.normal(0, 0.01, size=batch_feature.shape)
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


def get_tau_sed_generator(data_dir, train_or_eval='eval', force_preprocess=False):
    """
    Download, extract and preprocess the tau sed datset
    force_preprocess: Force the preprocess phase to repeate: usefull in case you change the preprocess parameters
    """
    global cfg_descriptor
    cfg_descriptor = f"{cfg_descriptor}_C-{'-'.join(cfg.tau_sed_labels)}"
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
    return features_and_labels_dir, features_mean_std_file, "TAU"


def get_film_clap_generator(data_dir, force_preprocess=False):
    """
    Preprocess and Creates a data generator for the film_clap dataset
    """
    global cfg_descriptor
    cfg_descriptor = f"{cfg_descriptor}_tm-{cfg.time_margin}"
    if not os.path.exists(data_dir):
        raise Exception("You should get you own dataset...")
    features_and_labels_dir = f"{data_dir}/processed_{cfg_descriptor}/features_and_labels"
    features_mean_std_file = f"{data_dir}/processed_{cfg_descriptor}/mel_features_mean_std.pkl"
    if not os.path.exists(features_and_labels_dir) or force_preprocess:
        print("preprocessing raw data")
        audio_paths_and_labels = get_film_clap_paths_and_labels(os.path.join(data_dir, 'raw'), time_margin=cfg.time_margin)
        preprocess_data(audio_paths_and_labels, output_dir=features_and_labels_dir, output_mean_std_file=features_mean_std_file)
    else:
        print("Using existing mel features")
    return features_and_labels_dir, features_mean_std_file, "FlimClap"