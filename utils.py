import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from collections import defaultdict
import os

from config import tau_sed_labels, frames_per_second, classes_num, mel_bins

eps = np.finfo(np.float).eps


def calculate_metrics(output, target):
    ths =np.arange(0.05, 1, 0.05)
    N = min(output.shape[1], target.shape[1])
    T = target[:, 0: N, :]
    O = output[:, 0: N, :]
    recals = []
    precisions = []
    f1_scores = []
    for th in ths:
        O_discrete = np.where(O > th, 1, 0)
        recall, prec = compute_recall_precision(O_discrete, T)
        recals.append(recall)
        precisions.append(prec)
    return recals, precisions


def compute_recall_precision(O, T):
    TP = ((2 * T - O) == 1).sum()

    num_gt = T.sum()
    num_positives = O.sum()

    recall = float(TP) / float(num_gt) if num_gt > 0 else 0
    prec = (float(TP) / float(num_positives)) if num_positives > 0 else 1

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


class ProgressPlotter:
    def __init__(self):
        self.train_buffer = []
        self.train_avgs = []
        self.val_avgs = []
        self.fh_score_avgs = []
        self.f1_score_avgs = []
        self.f5_score_avgs = []
        self.f10_score_avgs = []
        self.iterations = []

    def report_train_loss(self, loss):
        self.train_buffer.append(loss)

    def report_validation_metrics(self, val_losses, recal_sets, precision_sets):
        self.val_avgs.append(np.mean(val_losses, axis=0))
        self.last_recal_vals = np.mean(recal_sets, axis=0)
        self.last_precision_vals = np.mean(precision_sets, axis=0)
        self.fh_score_avgs.append((self.last_recal_vals[0] + self.last_precision_vals[0]) / 2)
        f1_scores = (2 * self.last_recal_vals * self.last_precision_vals) / (self.last_precision_vals + self.last_recal_vals)
        f5_scores = ((1+5**2) * self.last_recal_vals * self.last_precision_vals) / (5**2 * self.last_precision_vals + self.last_recal_vals)
        f10_scores = ((1+10**2) * self.last_recal_vals * self.last_precision_vals) / (10**2 * self.last_precision_vals + self.last_recal_vals)
        self.f1_score_avgs.append(np.max(f1_scores))
        self.f5_score_avgs.append(np.max(f5_scores))
        self.f10_score_avgs.append(np.max(f10_scores))


    def plot(self, outputs_dir, iteration):
        self.iterations.append(iteration)
        self.plot_train_eval_losses(os.path.join(outputs_dir, 'Training_loss.png'))
        self.plot_max_fscores(os.path.join(outputs_dir, 'f1_scores.png'))
        self.plot_roc(os.path.join(outputs_dir, 'ROC_plots', f"Roc-iteration-{iteration}.png"))

    def plot_train_eval_losses(self, plot_path):
        self.train_avgs += [np.mean(self.train_buffer)]
        self.train_buffer = []

        plt.plot(np.arange(len(self.train_avgs)), self.train_avgs, label='train', color='blue')
        plt.plot(np.arange(len(self.val_avgs)), self.val_avgs, label='validation', color='orange')
        plt.xticks(range(len(self.iterations)), self.iterations)
        plt.xlabel("train step")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    def plot_max_fscores(self, plot_path):
        plt.plot(np.arange(len(self.f1_score_avgs)), self.f1_score_avgs, color='blue', label='f1 scroe')
        plt.plot(np.arange(len(self.f5_score_avgs)), self.f5_score_avgs, color='green', label='f5 scroe')
        plt.plot(np.arange(len(self.f10_score_avgs)), self.f10_score_avgs, color='orange', label='f10 scroe')
        plt.plot(np.arange(len(self.fh_score_avgs)), self.fh_score_avgs, color='red', label='Fscore in highest recall')
        plt.title("F scores")
        plt.xticks(range(len(self.iterations)), self.iterations)
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    def plot_roc(self, plot_path):
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.plot(self.last_recal_vals, self.last_precision_vals)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.title(f"Max recall {self.last_recal_vals[0]:.1f} with precision: {self.last_precision_vals[0]:.1f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(plot_path)
        plt.clf()


def plot_debug_image(mel_features, output=None, target=None, plot_path=None):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    num_plots = 1
    if output is not None:
        num_plots += 1
    if target is not None:
        num_plots += 1

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 10))

    frames_num = mel_features.shape[0]

    axs[0].matshow(mel_features.T, origin='lower', aspect='auto', cmap='jet')
    axs[0].set_title('Log mel spectrogram', color='r')

    axs[0].set_ylabel('Mel bins')

    axs[0].set_yticks([0, mel_bins])
    axs[0].set_yticklabels([0, mel_bins])

    if output is not None:
        axs[1].matshow(output.T, origin='lower', aspect='auto', cmap='jet')
        axs[1].set_title("Predicted sound events", color='b')
    if target is not None:
        idx = 1 if output is None else 2
        axs[idx].matshow(target.T, origin='lower', aspect='auto', cmap='jet')
        axs[idx].set_title(f"Reference sound events, marked frames: {int(target.sum())}", color='r')

    tick_hop = frames_num // 8
    xticks = np.concatenate((np.arange(0, frames_num - tick_hop, tick_hop), [frames_num]))
    xlabels = [f"frame {x}\n{x//frames_per_second:.1f}s" for x in xticks]
    for i in range(num_plots):
        # axs[i].set_xlabel('frame/second')
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(xlabels)
        axs[i].xaxis.set_ticks_position('bottom')
        if i > 0:
            axs[i].set_yticks(np.arange(classes_num))
            axs[i].set_yticklabels(tau_sed_labels)
            axs[i].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)

    # fig.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

def human_format(num):
    """
    :param num: A number to print in a nice readable way.
    :return: A string representing this number in a readable way (e.g. 1000 --> 1K).
    """
    magnitude = 0

    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])  # add more suffices if you need them