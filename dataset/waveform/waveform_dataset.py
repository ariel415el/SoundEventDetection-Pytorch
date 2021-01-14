import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.spectogram_features.preprocess import read_multichannel_audio

time_margin= 0.1
working_sample_rate = 48000
frame_size = int(working_sample_rate * time_margin * 2)
hop_size = frame_size // 2


class waveform_dataset(Dataset):
    def __init__(self, audio_paths_labels_and_names):
        self.frames = []
        self.frame_labels = []
        for (audio_path, start_times, end_times, audio_name) in tqdm(audio_paths_labels_and_names):
            waveform = read_multichannel_audio(audio_path, target_fs=working_sample_rate)
            for center in np.arange(hop_size, len(waveform) - hop_size + 1, step=hop_size):
                frame = waveform[center - hop_size: center + hop_size]
                label = np.any([ t[0] < center < t[1] for t in zip(start_times, end_times) ])
                self.frames.append(frame)
                self.frame_labels.append(label)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.frame_labels[idx]


if __name__ == '__main__':
    from dataset.dataset_utils import get_film_clap_paths_and_labels
    data = get_film_clap_paths_and_labels('/home/ariel/projects/sound/data/Film_take_clap/raw', time_margin=0.1)
    dataset = waveform_dataset(data)
    print('done')
    x =1