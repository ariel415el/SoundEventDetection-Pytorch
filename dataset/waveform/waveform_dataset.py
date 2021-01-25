import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.spectogram_features.preprocess import read_multichannel_audio
from dataset.dataset_utils import get_film_clap_paths_and_labels

time_margin= 0.1
working_sample_rate = 48000
frame_size = int(working_sample_rate * time_margin * 2)
hop_size = frame_size // 2


class WaveformDataset(Dataset):
    def __init__(self, data_dir, val_descriptor=0.15, balance_classes=False, augment_data=False, epochs=10):
        audio_paths_labels_and_names = get_film_clap_paths_and_labels(os.path.join(data_dir, 'raw'), time_margin)
        self.balance_classes = balance_classes
        self.augment_data = augment_data
        self.epochs = epochs

        self.frames = []
        self.frame_labels = []
        self.val_samples_sets = []
        self.val_label_sets = []
        self.val_file_names = []
        print("WaveformDataset:")
        print("\t- Loading samples into memory... ")

        np.random.shuffle(audio_paths_labels_and_names)
        val_perc = int(len(audio_paths_labels_and_names) * val_descriptor)

        num_frames = 0
        num_event_frames = 0

        for i, (audio_path, start_times, end_times, audio_name) in tqdm(enumerate(audio_paths_labels_and_names)):
            waveform = read_multichannel_audio(audio_path, target_fs=working_sample_rate)
            waveform = waveform.T # -> (channels, samples)

            frames = []
            labels = []
            for center in np.arange(hop_size, waveform.shape[1] - hop_size + 1, step=hop_size):
                frame = waveform[: ,center - hop_size: center + hop_size]
                label = np.any([t[0] * working_sample_rate - hop_size < center < t[1] * working_sample_rate + hop_size for t in zip(start_times, end_times) ])
                frames.append(frame)
                labels.append(label)
                if label:
                    num_event_frames += 1
                num_frames += 1
            if i < val_perc:
                self.val_samples_sets.append(frames)
                self.val_label_sets.append(labels)
                self.val_file_names.append(audio_name)
            else:
                self.frames += frames
                self.frame_labels += labels
        print(f"\t- got {num_frames} fames. {num_event_frames} tagged as event")

    def get_validation_sampler(self, max_validate_num):
        for frames, labels, file_names in zip(self.val_samples_sets, self.val_label_sets, self.val_file_names):
            yield torch.tensor(frames), torch.tensor(labels), file_names

    def __len__(self):
        return len(self.frames) * self.epochs

    def __getitem__(self, idx):
        real_idx = idx % len(self.frames)
        waveform, label = self.frames[real_idx], self.frame_labels[real_idx]

        if self.augment_data:
            number_of_augmentations = np.random.choice([0, 1, 2, 3], 1, p=[0.6, 0.25, 0.1, 0.05])[0]
            for i in range(number_of_augmentations):
                random_idx = np.random.randint(len(self.frames) + 1)

                waveform += self.frames[random_idx]
                label = max(label, self.frame_labels[random_idx])
            waveform /= (number_of_augmentations + 1)

        return waveform, label


if __name__ == '__main__':
    from dataset.dataset_utils import get_film_clap_paths_and_labels
    data = get_film_clap_paths_and_labels('/home/ariel/projects/sound/data/Film_take_clap/raw', time_margin=0.1)
    dataset = WaveformDataset(data)
    print('done')
    x =1