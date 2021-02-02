import os

import numpy as np
import torch

from dataset.spectogram_features.preprocess import read_multichannel_audio
from dataset.waveform import waveform_configs as cfg


def split_to_frames_with_half_hop_size(waveform, start_times, end_times):
    """
    Splits the waveform to overlapping frames and taggs each frame if its covered up to some degree by an event
    """
    assert cfg.frame_size // 2 == cfg.hop_size
    frames = []
    labels = []
    for center in np.arange(cfg.hop_size, waveform.shape[1] - cfg.hop_size + 1, step=cfg.hop_size):
        frame = waveform[:, center - cfg.hop_size: center + cfg.hop_size]
        label = np.any([t[0] * cfg.working_sample_rate - cfg.hop_size < center < t[1] * cfg.working_sample_rate + cfg.hop_size for t in
                        zip(start_times, end_times)])
        # label = num_event_samples_in_frame / cfg.frame_size >= cfg.min_event_percentage_in_positive_frame
        frames.append(frame)
        labels.append(label)
    return frames, labels


def get_start_indices_labesl(waveform_length, start_times, end_times):
    """
    Returns: a waveform_length size boolean array where the ith entry says wheter or not a frame starting from the ith
    sample is covered by an event
    """
    label = np.zeros(waveform_length)
    for start, end in zip(start_times, end_times):
        event_first_start_index = int(start * cfg.working_sample_rate - cfg.frame_size * (1 - cfg.min_event_percentage_in_positive_frame))
        event_last_start_index = int(end * cfg.working_sample_rate - cfg.frame_size * cfg.min_event_percentage_in_positive_frame)
        label[event_first_start_index: event_last_start_index] = 1
    return label


class WaveformDataset:
    """
    This dataset allows training a detector on raw waveforms.
    It splits all waveforms to frames of a defined size with some overlap and tags gives them a tag of one of the classes
    or zero for no-event.
    """
    def __init__(self, audio_paths_labels_and_names, val_descriptor=0.15, balance_classes=False, augment_data=False):
        self.balance_classes = balance_classes
        self.augment_data = augment_data

        self.val_samples_sets = []
        self.val_label_sets = []
        self.val_file_names = []
        print("WaveformDataset:")
        print("\t- Loading samples into memory... ")
        # audio_paths_labels_and_names = audio_paths_labels_and_names[:50]
        np.random.shuffle(audio_paths_labels_and_names)
        val_perc = int(len(audio_paths_labels_and_names) * val_descriptor)

        # self.long_waveform = np.zeros((cfg.audio_channels, 0))
        # self.all_start_indices_labels = np.zeros(0, dtype=bool)
        # self.possible_start_indices = np.zeros(0, dtype=np.uint32)
        self.long_waveform = []
        self.all_start_indices_labels = []
        self.possible_start_indices = []
        frame_index = 0
        for i, (audio_path, start_times, end_times, audio_name) in enumerate(audio_paths_labels_and_names):
            waveform = read_multichannel_audio(audio_path, target_fs=cfg.working_sample_rate)
            waveform = waveform.T # -> (channels, samples)
            if i < val_perc:
                # Split wave form to overlapping frames and create labels for each
                frames, labels = split_to_frames_with_half_hop_size(waveform, start_times, end_times)
                self.val_samples_sets.append(frames)
                self.val_label_sets.append(labels)
                self.val_file_names.append(audio_name)
            else:
                # Store entire waveform so that frames can be later randomly cropped from it.
                # self.long_waveform = np.concatenate((self.long_waveform, waveform), axis=1)
                self.long_waveform.append(waveform)

                # Store the correct label for each starting sample index of a frame
                label_per_start_index = get_start_indices_labesl(waveform.shape[1], start_times, end_times).astype(bool)
                # self.all_start_indices_labels = np.concatenate((self.all_start_indices_labels, label_per_start_index))
                self.all_start_indices_labels.append(label_per_start_index)
                # restrict the starting indices so that random crop are not taken over two different waveforms
                possible_start_indices = np.arange(frame_index, frame_index + waveform.shape[1] - cfg.frame_size, dtype=np.uint32)
                # self.possible_start_indices = np.concatenate((self.possible_start_indices, possible_start_indices))
                self.possible_start_indices.append(possible_start_indices)

        self.long_waveform = np.concatenate(self.long_waveform, axis=1)
        self.all_start_indices_labels = np.concatenate(self.all_start_indices_labels)
        self.possible_start_indices = np.concatenate(self.possible_start_indices)

        np.random.shuffle(self.possible_start_indices)

        print(f"\t- Train split: {len(self.possible_start_indices)} overlapping fames. ~{100*np.sum(self.all_start_indices_labels==1)/len(self.possible_start_indices):.1f}% tagged as event")
        print(f"\t- Val split: {np.sum([ len(x) for x in self.val_label_sets])} frames. {np.sum([ np.sum(x) for x in self.val_label_sets])} tagged as event")

    def get_validation_sampler(self, max_validate_num):
        for i, (frames, labels, file_names) in enumerate(zip(self.val_samples_sets, self.val_label_sets, self.val_file_names)):
            if i > max_validate_num:
                break
            yield torch.tensor(frames), torch.tensor(labels), file_names

    def __len__(self):
        return len(self.possible_start_indices)

    def __getitem__(self, idx):
        start_index = self.possible_start_indices[idx]

        waveform = self.long_waveform[:, start_index + np.arange(cfg.frame_size)]
        label = self.all_start_indices_labels[start_index]

        if self.augment_data:
            waveform, label = self.augment_mix_samples(waveform, label)
            waveform, label = self.augment_add_noise(waveform, label)

        return waveform, label

    def augment_mix_samples(self, waveform, label):
        number_of_augmentations = np.random.choice([0, 1, 2, 3], 1, p=[0.5, 0.3, 0.15, 0.05])[0]
        for i in range(number_of_augmentations):
            random_start_idx = np.random.choice(self.possible_start_indices)
            waveform += self.long_waveform[:, random_start_idx + np.arange(cfg.frame_size)]
            label = max(label, self.all_start_indices_labels[random_start_idx])
        waveform /= (number_of_augmentations + 1)
        return waveform, label

    def augment_add_noise(self, waveform, label):
        # TODO these number are fit to noise added to waveform and not spectogram
        r = np.random.rand()
        if r > 0.5:
            noise_var = 0.001 + (r + 0.5) * (0.005 - 0.001)
            waveform += np.random.normal(0, noise_var, size=waveform.shape)
        return waveform, label


if __name__ == '__main__':
    from dataset.dataset_utils import get_film_clap_paths_and_labels
    dataset = WaveformDataset('/home/ariel/projects/sound/data/FilmClap')
    x = dataset[100]
    y = dataset[10000]
    z = dataset[10000000]
    import matplotlib.pyplot as plt
    w = 0
    z = 0
    for i in range(len(dataset)):
        frame, label = dataset[i]
        if label and z < 5:
            print(frame.shape)
            plt.plot(range(len(frame[0])), frame[0], c='r')
            plt.savefig(f"event{z}.png")
            plt.clf()
            z += 1
        elif not label and w < 5:
            print(frame.shape)
            plt.plot(range(len(frame[0])), frame[0], c='r')
            plt.savefig(f"no-event{w}.png")
            plt.clf()
            w += 1
        if z > 5 and w >5:
            break