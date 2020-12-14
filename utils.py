import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from collections import defaultdict
import os

eps = np.finfo(np.float).eps


def calculate_metrics(output, target, th):
    N = min(output.shape[1], target.shape[1])
    O = output[:, 0: N, :]
    T = target[:, 0: N, :]
    T = np.where(T > th, 1, 0)
    return compute_f1_score(O, T), compute_err(O, T)


def compute_f1_score(O, T):
    TP = ((2 * T - O) == 1).sum()

    prec = float(TP) / float(O.sum() + eps)
    recall = float(TP) / float(T.sum() + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def compute_err(O, T):
    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER



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

        colors = ['r', 'g', 'b', 'y']
        for i, (metric_name, values) in enumerate(self.metrics_buffer.items()):
            self.metrics_avgs[metric_name] += [np.mean(values)]

            plt.plot(np.arange(len(self.metrics_avgs[metric_name])), self.metrics_avgs[metric_name], label=metric_name, color=colors[i])

        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, 'Metrics.png'))
        plt.clf()

        self.metrics_buffer = defaultdict(lambda: list())