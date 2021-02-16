from dataset.spectogram.spectograms_dataset import preprocess_film_clap_data, SpectogramDataset
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import TSNE
from sklearn import svm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

if __name__ == '__main__':


    features_and_labels_dir, features_mean_std_file = preprocess_film_clap_data('../../data',
                                                                                preprocessed_mode="logMel",
                                                                                force_preprocess=False)

    dataset = SpectogramDataset(features_and_labels_dir, features_mean_std_file,
                                augment_data=False,
                                balance_classes=False,
                                val_descriptor=0.2,
                                preprocessed_mode="logMel")

    pos_frames = []
    neg_frames = []
    for idx in dataset.train_start_indices:
        features = dataset.train_features[0, idx]
        label = dataset.train_event_matrix[idx, 0]
        if label:
            pos_frames.append(features)
        else:
            neg_frames.append(features)
    pos_frames = np.array(pos_frames)
    neg_frames = np.array(neg_frames)
    neg_frames = neg_frames[np.random.randint(len(neg_frames), size=len(pos_frames)).tolist()]

    # pos_frames = pos_frames[np.random.randint(len(pos_frames), size=3000).tolist()]
    # neg_frames = neg_frames[np.random.randint(len(neg_frames), size=3000).tolist()]

    # neg_frames = random.sample(neg_frames, len(pos_frames))
    labels = np.zeros(len(pos_frames) + len(neg_frames))
    labels[:len(pos_frames)] = 1
    data = np.concatenate((pos_frames, neg_frames), axis=0)

    # pca = PCA(data.shape[1], 2)
    # pca.learn_encoder_decoder(data)
    # data_2d = pca.encode(data)

    # pca = skPCA(n_components=2)
    # pca.fit(data)
    # data_2d = pca.transform(data)

    # data_2d = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(data)

    # plt.scatter(data_2d[:len(pos_frames),0], data_2d[:len(pos_frames),1], color='r', label='pos', alpha=0.5)
    # plt.scatter(data_2d[len(pos_frames):,0], data_2d[len(pos_frames):,1], color='b', label='neg', alpha=0.5)

    print("Classifying")
    classifier = svm.SVC(C=1, kernel="rbf")
    classifier.fit(data[:-100], labels[:-100])
    predictions = classifier.predict(data[-100:])

    accuracy = np.mean(predictions == labels[-100:])
    print(accuracy)


