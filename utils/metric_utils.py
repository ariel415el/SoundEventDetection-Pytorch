import numpy as np


def calculate_metrics(output, target):
    ths =np.arange(0.00, 1.05, 0.05)
    N = min(output.shape[0], target.shape[0])
    T = target[:N]
    O = output[:N]
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

    recall = float(TP) / float(num_gt) if num_gt > 0 else 1
    prec = float(TP) / float(num_positives) if num_positives > 0 else 1

    return recall, prec


def f_score(recll, precision, precision_importance_factor=1):
    return (1+precision_importance_factor**2) * recll * precision / (precision_importance_factor**2 * recll + precision + 1e-9)