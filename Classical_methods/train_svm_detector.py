import pickle

import numpy as np
from sklearn import svm
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from dataset.spectogram.spectograms_dataset import create_event_matrix, preprocess_film_clap_data
from utils.metric_utils import calculate_metrics, f_score
from utils.plot_utils import plot_sample_features

matplotlib.use('TkAgg')


class SVM_detector:
    def __init__(self):
        self.svm = svm.SVC(C=1, kernel="rbf", probability=True)

    def learn(self, spectograms, event_matrices):
        data = np.concatenate(spectograms, axis=0)
        labels = np.concatenate(event_matrices, axis=0)
        sample_weights = labels * 5 + (1 - labels)
        print(f"Svm training on {len(data)} samples... ", end='')
        self.svm.fit(data, labels, sample_weight=sample_weights)
        print("Done")

    def predict(self, spectogram):
        result = np.zeros(spectogram.shape[0])
        for i in range(spectogram.shape[0]):
            # result[i] = self.svm.predict([spectogram[i]])
            result[i] = self.svm.predict_proba([spectogram[i]])[0,1]

        return result


def get_raw_data():
    features_and_labels_dir, features_mean_std_file = preprocess_film_clap_data('../../data',
                                                                                preprocessed_mode="logMel",
                                                                                force_preprocess=False)
    d = pickle.load(open(features_mean_std_file, 'rb'))
    mean = d['mean']
    std = d['std']

    all_paths = [os.path.join(features_and_labels_dir, x) for x in os.listdir(features_and_labels_dir)]

    file_names = []
    features_list = []
    event_matrix_list = []
    for feature_path in all_paths:
        data = pickle.load(open(feature_path, 'rb'))
        feature = (data['features'][0] - mean) / std
        features_list.append(feature)
        event_matrix = create_event_matrix(feature.shape[0], data['start_times'], data['end_times'])[:,0]
        event_matrix_list.append(event_matrix)
        file_names.append(os.path.basename(os.path.splitext(feature_path)[0]))

    data = list(zip(features_list, event_matrix_list, file_names))
    return data


if __name__ == '__main__':
    all_data = get_raw_data()
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


    model = SVM_detector()
    model.learn(train_features_list, train_event_matrix_list)

    recal_sets, precision_sets, APs, accs = [], [], [], []
    for feature, event_mat, name in zip(val_features_list, val_event_matrix_list, val_file_names):
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
        z = 1

    recal_vals = np.mean(recal_sets, axis=0)
    precision_vals = np.mean(precision_sets, axis=0)

    plt.plot(recal_vals, precision_vals)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    MAP = np.sum(recal_vals[:-1] * (recal_vals[:-1] - recal_vals[1:]))
    plt.title(f"Validation AVG ROC"
              f"\nAP: {MAP:.2f}")
    plt.xlabel("Avg Recall")
    plt.ylabel("Avg Precision")
    plt.savefig("svm-classification.png")
    plt.clf()