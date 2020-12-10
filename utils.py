from torch.nn import functional as F


def binary_crossentropy(output, target):
    '''Binary crossentropy between output and target.

    Args:
      output: (batch_size, frames_num, classes_num)
      target: (batch_size, frames_num, classes_num)
    '''

    # To let output and target to have the same time steps. The mismatching
    # size is caused by pooling in CNNs.
    N = min(output.shape[1], target.shape[1])

    return F.binary_cross_entropy(
        output[:, 0: N, :],
        target[:, 0: N, :])