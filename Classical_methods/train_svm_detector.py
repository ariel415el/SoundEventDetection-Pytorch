import pickle
import numpy as np
from sklearn import svm
import os
import matplotlib
import matplotlib.pyplot as plt
import random

from dataset.dataset_utils import get_film_clap_paths_and_labels, read_multichannel_audio
from dataset.spectogram.preprocess import multichannel_complex_to_log_mel
from dataset.waveform.waveform_dataset import split_to_frames_with_hop_size
from utils.metric_utils import calculate_metrics, f_score
from utils.plot_utils import plot_sample_features

import dataset.waveform.waveform_configs as cfg
matplotlib.use('TkAgg')


class SVM_detector:
    def __init__(self, soft_svm, recall_priority):
        self.soft_svm = soft_svm
        self.svm = svm.SVC(C=1, kernel="rbf", probability=soft_svm)
        self.recall_priority = recall_priority
    def learn(self, spectograms, event_matrices):
        data = np.concatenate(spectograms, axis=0)
        labels = np.concatenate(event_matrices, axis=0)
        sample_weights = labels * self.recall_priority + (1 - labels)
        print(f"Svm training on {len(data)} samples... ", end='')
        self.svm.fit(data, labels, sample_weight=sample_weights)
        print("Done")

    def predict(self, spectogram):
        result = np.zeros(spectogram.shape[0])
        for i in range(spectogram.shape[0]):
            if self.soft_svm:
                result[i] = self.svm.predict_proba([spectogram[i]])[0,1]
            else:
                result[i] = self.svm.predict([spectogram[i]])

        return result

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.svm, file)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                self.svm = pickle.load(file)

def get_raw_data():
    NFFT = 2**int(np.ceil(np.log2(cfg.frame_size)))

    audio_paths_labels_and_names = get_film_clap_paths_and_labels("../../data/FilmClap", time_margin=cfg.time_margin)

    features = []
    label_sets = []
    file_names = []
    for i, (audio_path, start_times, end_times, audio_name) in enumerate(audio_paths_labels_and_names):
        assert "_".join(audio_name.split("_")[1:]) in audio_path
        waveform = read_multichannel_audio(audio_path, target_fs=cfg.working_sample_rate)
        waveform = waveform.T  # -> (channels, samples)
        # Split wave form to overlapping frames and create labels for each
        frames, labels = split_to_frames_with_hop_size(waveform, start_times, end_times)
        frames = np.concatenate(frames, axis=0)
        frames *= np.hanning(frames.shape[1])
        complex_spectogram = np.fft.rfft(frames, NFFT)
        mel_features = multichannel_complex_to_log_mel(complex_spectogram)

        features.append(mel_features)
        label_sets.append(np.array(labels))
        file_names.append(audio_name)

    data = list(zip(features, label_sets, file_names))
    return data

def split_train_val(all_data):
    random.shuffle(all_data)
    features_list, event_matrix_list, file_names = zip(*all_data)
    features_list, event_matrix_list, file_names = list(features_list), list(event_matrix_list), list(file_names)

    # Split to train val
    val_amount = len(features_list) // 5
    train_features_list = features_list[val_amount:]
    train_event_matrix_list = event_matrix_list[val_amount:]
    train_file_names = file_names[val_amount:]

    val_features_list = features_list[:val_amount]
    val_event_matrix_list = event_matrix_list[:val_amount]
    val_file_names = file_names[:val_amount]

    return train_features_list, train_event_matrix_list, val_features_list, val_event_matrix_list, val_file_names

def evaluate_model(model, eval_data):
    # Evaluate model
    recal_sets, precision_sets, APs, accs = [], [], [], []
    for feature, event_mat, name in eval_data:
        pred = model.predict(feature)
        acc = np.mean(pred == event_mat)

        recals, precisions, AP = calculate_metrics(pred, event_mat)
        f1s = [f_score(r,p,1) for r,p in zip(recals, precisions)]
        print(f"{name} max f1 score: {np.max(f1s)}")
        recal_sets.append(recals)
        precision_sets.append(precisions)
        APs.append(AP)
        accs.append(acc)

        plot_sample_features(np.array([feature]),
                             mode='spectogram',
                             output=pred.reshape(-1,1),
                             target=event_mat.reshape(-1,1),
                             file_name=f"Acc:{acc:.2f}, AP: {AP:.2f}, f1: {np.max(f1s):.2f}",
                             plot_path=f"plots/{name}-f1: {np.max(f1s):.2f}.png")

    recal_vals = np.mean(recal_sets, axis=0)
    precision_vals = np.mean(precision_sets, axis=0)
    MAP = np.sum(recal_vals[:-1] * (recal_vals[:-1] - recal_vals[1:]))

    plt.plot(recal_vals, precision_vals)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.title(f"Validation AVG ROC"
              f"\nAP: {MAP:.2f}")
    plt.xlabel("Avg Recall")
    plt.ylabel("Avg Precision")
    plt.savefig("svm-classification.png")
    plt.clf()

if __name__ == '__main__':
    # Load data
    all_data = get_raw_data()
    train_features_list, train_event_matrix_list, val_features_list, val_event_matrix_list, val_file_names = split_train_val(all_data)

    # Train model
    model = SVM_detector(soft_svm=True, recall_priority=10)
    model.learn(train_features_list, train_event_matrix_list)
    model.save("last_pickled_model.pkl")

    # Evaluate model
    evaluate_model(model, zip(val_features_list, val_event_matrix_list, val_file_names))

