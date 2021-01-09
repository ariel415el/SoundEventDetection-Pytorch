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
    for th in ths:
        O_discrete = np.where(O > th, 1, 0)
        recall, prec = compute_recall_precision(O_discrete, T)
        recals.append(recall)
        precisions.append(prec)

    recals, precisions = np.array(recals), np.array(precisions)
    # from sklearn.metrics import average_precision_score
    # AP = average_precision_score(T.reshape(-1).astype(int), O.reshape(-1))
    AP = np.sum(precisions[:-1] * (recals[:-1] - recals[1:]))
    return recals, precisions, AP



def compute_recall_precision(O, T):
    TP = ((2 * T - O) == 1).sum()

    num_gt = T.sum()
    num_positives = O.sum()

    recall = float(TP) / float(num_gt) if num_gt > 0 else 0
    prec = (float(TP) / float(num_positives)) if num_positives > 0 else 1

    return recall, prec


def binary_crossentropy(output, target, p_ones=0.7):
    '''Binary crossentropy between output and target.

    Args:
      output: (batch_size, frames_num, classes_num)
      target: (batch_size, frames_num, classes_num)
    '''

    # Number of frames differ due to pooling on eve/odd number of frames
    N = min(output.shape[1], target.shape[1])
    clipped_output = output[:, 0: N, :]
    clipped_target = target[:, 0: N, :]
    import torch
    weight = torch.tensor([1 - p_ones, p_ones])
    weight_ = weight[clipped_target.data.view(-1).long()].view_as(clipped_target)
    return F.binary_cross_entropy(clipped_output, clipped_target, weight=weight_.to(target.device))


def f_score(recll, precision, precision_importance_factor=1):
    return (1+precision_importance_factor**2) * recll * precision / (precision_importance_factor**2 * recll + precision + 1e-9)


class ProgressPlotter:
    def __init__(self):
        self.train_buffer = []
        self.train_avgs = []
        self.val_avgs = []
        self.fh_score_avgs = []
        self.f1_score_avgs = []
        self.f5_score_avgs = []
        self.AP_avgs = []
        self.iterations = []

    def report_train_loss(self, loss):
        self.train_buffer.append(loss)

    def report_validation_metrics(self, val_losses, recal_sets, precision_sets, APs, iteration):
        self.iterations.append(iteration)

        self.val_avgs.append(np.mean(val_losses))
        self.AP_avgs.append(np.mean(APs))
        self.last_recal_vals = np.mean(recal_sets, axis=0)
        self.last_precision_vals = np.mean(precision_sets, axis=0)
        self.fh_score_avgs.append(self.last_precision_vals[0])
        f1_scores = f_score(self.last_precision_vals, self.last_recal_vals, precision_importance_factor=1)
        f5_scores = f_score(self.last_precision_vals, self.last_recal_vals, precision_importance_factor=5)
        self.f1_score_avgs.append(np.max(f1_scores))
        self.f5_score_avgs.append(np.max(f5_scores))

    def plot(self, outputs_dir):
        self.plot_train_eval_losses(os.path.join(outputs_dir, 'Training_loss.png'))
        self.plot_metrics(os.path.join(outputs_dir, 'Metrics.png'))
        self.plot_roc(os.path.join(outputs_dir, 'ROC_plots', f"Roc-iteration-{self.iterations[-1]}.png"))

    def plot_train_eval_losses(self, plot_path):
        self.train_avgs += [np.mean(self.train_buffer)]
        self.train_buffer = []

        plt.plot(np.arange(len(self.train_avgs)), self.train_avgs, label='train', color='blue')
        plt.plot(np.arange(len(self.val_avgs)), self.val_avgs, label='validation', color='orange')
        x_indices = np.arange(0, len(self.iterations), max(len(self.iterations) // 5, 1))
        plt.xticks(x_indices, np.array(self.iterations)[x_indices])
        plt.xlabel("train step")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    def plot_metrics(self, plot_path):
        plt.plot(np.arange(len(self.f1_score_avgs)), self.f1_score_avgs, color='blue', label='Max f1 scroe')
        plt.plot(np.arange(len(self.f5_score_avgs)), self.f5_score_avgs, color='green', label='Max f5 scroe')
        plt.plot(np.arange(len(self.fh_score_avgs)), self.fh_score_avgs, color='red', label='Precision in highest recall')
        plt.plot(np.arange(len(self.AP_avgs)), self.AP_avgs, color='orange', label='Average precision')
        plt.title("Metrics")
        x_indices = np.arange(0, len(self.iterations), max(len(self.iterations) // 5, 1))
        plt.xticks(x_indices, np.array(self.iterations)[x_indices])
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    def plot_roc(self, plot_path):
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.plot(self.last_recal_vals, self.last_precision_vals)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        MAP = np.sum(self.last_precision_vals[:-1] * (self.last_recal_vals[:-1] - self.last_recal_vals[1:]))
        plt.title(f"Validation AVG ROC"
                  f"\nAP: {MAP:.2f}")
        plt.xlabel("Avg Recall")
        plt.ylabel("Avg Precision")
        plt.savefig(plot_path)
        plt.clf()


def plot_debug_image(mel_features, output=None, target=None, file_name=None, plot_path=None):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    num_plots = 1
    if output is not None:
        num_plots += 1
    if target is not None:
        num_plots += 1

    fig, axs = plt.subplots(num_plots, 1, figsize=(20, 15))
    plt.subplots_adjust(hspace=1)
    frames_num = mel_features.shape[0]
    if file_name:
        fig.suptitle(f"Sample name: {file_name}")
    im = axs[0].matshow(mel_features.T, origin='lower', aspect='auto', cmap='jet')
    fig.colorbar(im, ax=axs[0])

    axs[0].set_title('Log mel spectrogram', color='r')

    axs[0].set_ylabel('Mel bins')

    axs[0].set_yticks([0, mel_bins])
    axs[0].set_yticklabels([0, mel_bins])

    if output is not None:
        im = axs[1].matshow(output.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        fig.colorbar(im, ax=axs[1])
        axs[1].set_title("Predicted sound events", color='b')
    if target is not None:
        idx = 1 if output is None else 2
        im = axs[idx].matshow(target.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        fig.colorbar(im, ax=axs[idx])
        axs[idx].set_title(f"Reference sound events, marked frames: {int(target.sum())}", color='r')

    tick_hop = frames_num // 8
    xticks = np.concatenate((np.arange(0, frames_num - tick_hop, tick_hop), [frames_num]))
    xlabels = [f"frame {x}\n{x/frames_per_second:.1f}s" for x in xticks]
    for i in range(num_plots):
        # axs[i].set_xlabel('frame/second')
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(xlabels)
        axs[i].xaxis.set_ticks_position('bottom')
        if i > 0:
            axs[i].set_yticks(np.arange(classes_num))
            axs[i].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)

    fig.tight_layout()
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)