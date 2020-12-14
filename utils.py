import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from collections import defaultdict
import os

import config as cfg

eps = np.finfo(np.float).eps


def calculate_metrics(output, target, ths):
    N = min(output.shape[1], target.shape[1])
    T = target[:, 0: N, :]
    O = output[:, 0: N, :]
    recals = []
    precisions = []
    f1_scores = []
    for th in ths:
        O_discrete = np.where(O > th, 1, 0)
        recall, prec = compute_recall_precision(O_discrete, T)
        f1_score = 2 * prec * recall / (prec + recall + eps)
        recals.append(recall)
        precisions.append(prec)
        f1_scores.append(f1_score)
    return f1_scores, recals, precisions


def compute_recall_precision(O, T):
    TP = ((2 * T - O) == 1).sum()

    recall = float(TP) / float(T.sum() + eps)
    prec = float(TP) / float(O.sum() + eps)

    return recall, prec


def binary_crossentropy(output, target):
    '''Binary crossentropy between output and target.

    Args:
      output: (batch_size, frames_num, classes_num)
      target: (batch_size, frames_num, classes_num)
    '''

    # Number of frames differ due to pooling on eve/odd number of frames
    N = min(output.shape[1], target.shape[1])

    return F.binary_cross_entropy(
        output[:, 0: N, :],
        target[:, 0: N, :])


class loss_tracker:
    def __init__(self, plot_dir):
        self.train_buffer = []
        self.train_avgs = []
        self.val_buffer = []
        self.val_avgs = []
        self.metrics_buffer = defaultdict(lambda: list())
        self.metrics_avgs = defaultdict(lambda: list())
        self.plot_dir = plot_dir

    def report_train_loss(self, loss):
        self.train_buffer.append(loss)

    def report_val_losses(self, losses):
        self.val_buffer += losses

    def report_val_metrics(self, metrics):
        for metric_name, values in metrics.items():
            self.metrics_buffer[metric_name] += values

    def report_val_roc(self, recal_sets, precision_sets):
        self.recal_vals = np.mean(recal_sets, axis=0)
        self.precision_vals = np.mean(precision_sets, axis=0)
        self.recal_vals, self.precision_vals = zip(*sorted(zip(self.recal_vals, self.precision_vals)))

    def plot(self):
        self.train_avgs += [np.mean(self.train_buffer)]
        self.val_avgs += [np.mean(self.val_buffer)]
        self.train_buffer = []
        self.val_buffer = []

        plt.plot(np.arange(len(self.train_avgs)), self.train_avgs, label='train', color='blue')
        plt.plot(np.arange(len(self.val_avgs)), self.val_avgs, label='validation', color='orange')
        plt.xlabel("train step")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, 'Training_loss.png'))
        plt.clf()


        plt.plot(self.recal_vals, self.precision_vals)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, 'ROC.png'))
        plt.clf()

        colors = ['r', 'g', 'b', 'y']
        for i, (metric_name, values) in enumerate(self.metrics_buffer.items()):
            self.metrics_avgs[metric_name] += [np.mean(values)]

            plt.plot(np.arange(len(self.metrics_avgs[metric_name])), self.metrics_avgs[metric_name], label=metric_name, color=colors[i])

        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, 'Metrics.png'))
        plt.clf()

        self.metrics_buffer = defaultdict(lambda: list())


def plot_debug_image(mel_features, output, target=None, plot_path=None):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    num_plots = 2 if target is None else 3
    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 10))

    frames_num = output.shape[1]
    length_in_second = frames_num / float(cfg.frames_per_second)

    axs[0].matshow(mel_features.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(output[0].T, origin='lower', aspect='auto', cmap='jet')

    axs[0].set_title('Log mel spectrogram', color='r')
    axs[1].set_title("Predicted sound events)", color='b')
    if target is not None:
        axs[2].matshow(target[0].T, origin='lower', aspect='auto', cmap='jet')
        axs[2].set_title('Reference sound events', color='r')

    for i in range(2):
        axs[i].set_xticks([0, frames_num])
        axs[i].set_xticklabels(['0', '{:.1f} s'.format(length_in_second)])
        axs[i].xaxis.set_ticks_position('bottom')
        axs[i].set_yticks(np.arange(cfg.classes_num))
        axs[i].set_yticklabels(cfg.tau_sed_labels)
        axs[i].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)

    axs[0].set_ylabel('Mel bins')
    axs[0].set_yticks([0, cfg.mel_bins])
    axs[0].set_yticklabels([0, cfg.mel_bins])

    # fig.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)