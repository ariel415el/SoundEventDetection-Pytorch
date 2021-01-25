import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_waveform(ax, waveform, sample_rate):
    # subsample for faster plotting
    ax.set_facecolor('k')
    new_sample_rate = sample_rate / 10
    new_waveform = waveform[::10]
    ax.plot(range(len(new_waveform)), new_waveform, c='r')
    ax.margins(x=0)
    ax.set_title('Time', color='r')
    ax.set_ylabel('Amplitudes')

    xticks = np.arange(0, len(new_waveform), len(new_waveform) // 8)
    xlabels = [f"Sec {x / new_sample_rate:.2f}s" for x in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.xaxis.set_ticks_position('bottom')


def plot_spectogram(ax, spectpgram, frames_per_second):
    frames_num, mel_bins = spectpgram.shape
    colorbar = ax.matshow(spectpgram.T, origin='lower', aspect='auto', cmap='jet')
    ax.set_title('Log mel spectrogram', color='r')
    ax.set_ylabel('Mel bins')
    ax.set_yticks([0, mel_bins])
    ax.set_yticklabels([0, mel_bins])

    tick_hop = frames_num // 8
    xticks = np.concatenate((np.arange(0, frames_num - tick_hop, tick_hop), [frames_num]))
    xlabels = [f"frame {x}\n{x / frames_per_second:.1f}s" for x in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.xaxis.set_ticks_position('bottom')

    return colorbar


def plot_classification_matrix(ax, mat, frames_per_second):
    frames_num = mat.shape[0]
    colorbar = ax.matshow(mat.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    tick_hop = frames_num // 8
    xticks = np.concatenate((np.arange(0, frames_num - tick_hop, tick_hop), [frames_num]))
    xlabels = [f"frame {x}\n{x / frames_per_second:.1f}s" for x in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.xaxis.set_ticks_position('bottom')

    return colorbar


def add_colorbar_to_axs(fig, ax, colorbar):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=0.01)
    fig.colorbar(colorbar, cax=cax, orientation='vertical')


def plot_mel_features(input, mode, output=None, target=None, file_name=None, plot_path=None):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    num_plots = 1
    if output is not None:
        num_plots += 1
    if target is not None:
        num_plots += 1

    fig, axs = plt.subplots(num_plots, 1, figsize=(20, 20))
    plt.subplots_adjust(hspace=1)
    if file_name:
        fig.suptitle(f"Sample name: {file_name}")

    input = input.mean(0)
    if mode == 'Spectogram':
        from dataset.spectogram_features.spectogram_configs import mel_bins, frames_per_second, classes_num
        colorbar = plot_spectogram(axs[0], input, frames_per_second)
        add_colorbar_to_axs(fig, axs[0], colorbar)

    else: # mode == 'Waveform
        from dataset.waveform.waveform_configs import working_sample_rate, hop_size
        frames_per_second = working_sample_rate // hop_size
        waveform = input[:,:hop_size].flatten()
        plot_waveform(axs[0], waveform, working_sample_rate)

    if output is not None:
        colorbar = plot_classification_matrix(axs[1], output, frames_per_second)
        axs[1].set_title("Predicted sound events", color='b')
        add_colorbar_to_axs(fig, axs[1], colorbar)

    if target is not None:
        idx = 1 if output is None else 2
        colorbar = plot_classification_matrix(axs[idx], target, frames_per_second)
        axs[idx].set_title(f"Reference sound events, marked frames: {int(target.sum())}", color='r')

        add_colorbar_to_axs(fig, axs[idx], colorbar)

    fig.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)