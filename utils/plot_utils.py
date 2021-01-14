import os

import numpy as np
from matplotlib import pyplot as plt

from dataset.spectogram_features.spectogram_configs import mel_bins, frames_per_second, classes_num


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