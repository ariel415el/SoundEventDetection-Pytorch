import librosa
import numpy as np
import os
from tqdm import tqdm
import soundfile
import pandas as pd
import pickle
import subprocess
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import config as cfg

def download_foa_data(data_dir, eval_only=False):
    urls = [
            'https://zenodo.org/record/2599196/files/foa_dev.z01?download=1',
            'https://zenodo.org/record/2599196/files/foa_dev.z02?download=1',
            'https://zenodo.org/record/2599196/files/foa_dev.zip?download=1',
            'https://zenodo.org/record/2599196/files/metadata_dev.zip?download=1',
            'https://zenodo.org/record/3377088/files/foa_eval.zip?download=1',
            'https://zenodo.org/record/3377088/files/metadata_eval.zip?download=1',
    ]
    md5s = [
            'bd5b18a47a3ed96e80069baa6b221a5a',
            '5194ebf43ae095190ed78691ec9889b1',
            '2154ad0d9e1e45bfc933b39591b49206',
            'c2e5c8b0ab430dfd76c497325171245d',
            '4a8ca8bfb69d7c154a56a672e3b635d5',
            'a0ec7640284ade0744dfe299f7ba107b'
    ]
    names = [
        'foa_dev.z01',
        'foa_dev.z02',
        'foa_dev.zip',
        'metadata_dev.zip',
        'foa_eval.zip',
        'metadata_eval.zip'
    ]

    if eval_only:
        urls, md5s, names = urls[-2:], md5s[-2:], names[-2:]

    os.makedirs(data_dir, exist_ok=True)
    for url, md5, name in zip(urls, md5s, names):
        download_url(url, data_dir, md5=md5, filename=name)


def extract_foa_data(data_dir, eval_only=False):
    subprocess.call(["unzip", os.path.join(data_dir,'metadata_eval.zip'), "-d", data_dir])
    subprocess.call(["unzip", os.path.join(data_dir,'foa_eval.zip'), "-d", data_dir])
    if not eval_only:
        subprocess.call(["unzip", os.path.join(data_dir,'metadata_dev.zip'), "-d", data_dir])
        subprocess.call(f"zip -s 0 {os.path.join(data_dir,'foa_dev.zip')} --out {os.path.join(data_dir,'unsplit_foa_dev.zip')}".split(" "))
        subprocess.call(f"unzip {os.path.join(data_dir, 'unsplit_foa_dev.zip')} -d {data_dir}".split(" "))
        subprocess.call(f"cp -R {data_dir}/proj/asignal/DCASE2019/dataset/foa_eval -d {data_dir}/foa_eval".split(" "))
        #Todo remove {data_dir}/proj


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

        feature = np.array([self.transform_singlechannel(
            multichannel_audio[:, m]) for m in range(channels_num)])

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
    (multichannel_audio, fs) = soundfile.read(audio_path)
    '''(samples, channels_num)'''

    if target_fs is not None and fs != target_fs:
        (samples, channels_num) = multichannel_audio.shape

        multichannel_audio = np.array(
            [librosa.resample(multichannel_audio[:, i], orig_sr=fs, target_sr=target_fs) for i in range(channels_num)]
                                        ).T
        '''(samples, channels_num)'''

    return multichannel_audio, fs

def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def preprocess_data(audio_dir, labels_data_dir, output_dir, output_mean_std_file):
    os.makedirs(output_dir, exist_ok=True)

    feature_extractor = LogMelExtractor(
        sample_rate=cfg.sample_rate,
        window_size=cfg.window_size,
        hop_size=cfg.hop_size,
        mel_bins=cfg.mel_bins,
        fmin=cfg.fmin,
        fmax=cfg.fmax)

    all_features = []

    for audio_fname in tqdm(os.listdir(audio_dir)[:5]):
        bare_name = os.path.splitext(audio_fname)[0]

        audio_path = os.path.join(audio_dir, audio_fname)
        multichannel_audio, _ = read_multichannel_audio(audio_path=audio_path, target_fs=cfg.sample_rate)
        feature = feature_extractor.transform_multichannel(multichannel_audio)
        all_features.append(feature)

        labels_path = os.path.join(labels_data_dir, bare_name + ".csv")
        df = pd.read_csv(labels_path, sep=',')

        output_path = os.path.join(output_dir, bare_name + "_mel_features_and_labels.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({'features': feature, 'classes': df['sound_event_recording'].values,
                         'start_times': df['start_time'].values, 'end_times': df['end_time'].values},
                         f, protocol=pickle.HIGHEST_PROTOCOL)

    all_features = np.concatenate(all_features, axis=1)
    mean, std = calculate_scalar_of_tensor(all_features)
    with open(output_mean_std_file, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)

# class SED_dataset(Dataset):
#     def __init__(self, features_and_labels_dir):
#         self.data_paths = os.listdir(features_and_labels_dir)
#
#     def __len__(self):
#         return len(self.data_paths)
#
#     def __getitem__(self, idx):
#         with open(self.data_paths[idx], 'rb') as f:
#             return pickle.load(f)

def create_event_matrix(frames_num, classes, start_times, end_times):
    # Researve space data
    event_matrix = np.zeros((frames_num, cfg.classes_num))

    for n in range(len(classes)):
        class_id = cfg.lb_to_idx[classes[n]]
        start_frame = int(round(start_times[n] * cfg.frames_per_second))
        end_frame = int(round(end_times[n] * cfg.frames_per_second)) + 1

        event_matrix[start_frame: end_frame, class_id] = 1

    return event_matrix


class DataGenerator(object):
    def __init__(self, features_and_labels_dir, mean_std_file, batch_size, seed=1234):
        d = pickle.load(open(mean_std_file, 'rb'))
        self.mean = d['mean']
        self.std = d['std']
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)

        self.frames_per_second = cfg.frames_per_second
        self.classes_num = cfg.classes_num
        self.lb_to_idx = cfg.lb_to_idx
        self.time_steps = cfg.time_steps

        feature_names = sorted(os.listdir(features_and_labels_dir))

        val_split = int(len(feature_names)*0.9)
        self.train_feature_names = feature_names[:val_split]

        self.validate_feature_names = feature_names[val_split:]

        self.train_features_list = []
        self.train_event_matrix_list = []
        self.train_index_array_list = []
        frame_index = 0

        # Load training feature and targets
        for feature_name in self.train_feature_names:
            data = pickle.load(open(os.path.join(features_and_labels_dir, feature_name), 'rb'))
            feature = data['features']
            event_matrix = create_event_matrix(feature.shape[1], data['classes'], data['start_times'], data['end_times'])

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
            event_matrix = create_event_matrix(feature.shape[1], data['classes'], data['start_times'], data['end_times'])

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

            batch_data_dict = {
                'name': name,
                'feature': feature[:, None, :, :],  # (channels_num, batch_size=1, frames_num, mel_bins)
                'event': event_matrix[None, :, :],  # (batch_size=1, frames_num, mel_bins)
            }
            '''The None above indicates using an entire audio recording as 
            input and batch_size=1 in inference'''

            yield batch_data_dict

    def transform(self, x):
        return (x - self.mean) / self.std

def get_batch_generator(data_dir, batch_size, train_or_eval='eval'):
    ambisonic_2019_data_dir = f"{data_dir}/Tau_spatial_sound_events_2019"
    # audio_dir = f"{ambisonic_2019_data_dir}/raw/foa_{train_or_eval}"
    # meda_data_dir = f"{ambisonic_2019_data_dir}/raw/metadata_{train_or_eval}"
    audio_dir = f"{ambisonic_2019_data_dir}/foa_{train_or_eval}"
    meda_data_dir = f"{ambisonic_2019_data_dir}/metadata_{train_or_eval}"

    if not os.path.exists(audio_dir):
        print("Downloading raw data")
        download_foa_data(ambisonic_2019_data_dir, eval_only=train_or_eval == 'eval')
        extract_foa_data(ambisonic_2019_data_dir, eval_only=train_or_eval == 'eval')
    else:
        print("Using existing raw data")

    # features_and_labels_dir = f"{ambisonic_2019_data_dir}/processed/features_and_labels_{train_or_eval}"
    # features_mean_std_file = f"{ambisonic_2019_data_dir}/processed/mel_features_mean_std_{train_or_eval}.pkl"
    features_and_labels_dir = f"{ambisonic_2019_data_dir}/features_and_labels_{train_or_eval}"
    features_mean_std_file = f"{ambisonic_2019_data_dir}/mel_features_mean_std_{train_or_eval}.pkl"
    if not os.path.exists(features_and_labels_dir):
        print("preprocessing raw data")
        preprocess_data(audio_dir, meda_data_dir, output_dir=features_and_labels_dir, output_mean_std_file=features_mean_std_file)
    else:
        print("Using existing mel features"
              )
    return DataGenerator(features_and_labels_dir, features_mean_std_file, batch_size)

