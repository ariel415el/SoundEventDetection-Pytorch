import os
import numpy as np
from matplotlib import pyplot as plt

from torch import tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from utils.metric_utils import f_score


class WeightedBCE:
    def __init__(self, recall_factor, multi_frame):
        self.recall_factor = tensor([recall_factor])
        self.multi_frame = multi_frame

    def __call__(self, output, target):
        if self.multi_frame:
            # expected shape (batch_size, frames_num, classes_num)
            # Number of frames differ due to pooling on eve/odd number of frames
            N = min(output.shape[1], target.shape[1])
            _output = output[:, :N]
            _target = target[:, :N]

        else:
            # expected shape (batch_size, classes_num)
            _output = output.reshape(-1)
            _target = target

        return binary_cross_entropy_with_logits(_output, _target,
                                         pos_weight=self.recall_factor.to(_output.device))


class ProgressPlotter:
    def __init__(self):
        self.train_buffer = []
        self.train_avgs = []
        self.val_avgs = []
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
