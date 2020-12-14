import argparse
from models import *
import config as cfg
from dataset.preprocess import LogMelExtractor, read_multichannel_audio
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # Train
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--outputs_dir', type=str, default='inference_outputs', help='Directory of your workspace.')
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

    model = Cnn_AvgPooling(cfg.classes_num).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'])

    print("Preprocessing audio file..")
    feature_extractor = LogMelExtractor(
        sample_rate=cfg.sample_rate,
        window_size=cfg.window_size,
        hop_size=cfg.hop_size,
        mel_bins=cfg.mel_bins,
        fmin=cfg.fmin,
        fmax=cfg.fmax)

    multichannel_audio, _ = read_multichannel_audio(audio_path=args.audio_file, target_fs=cfg.sample_rate)

    print("Inference..")
    mel_features = feature_extractor.transform_multichannel(multichannel_audio)

    with torch.no_grad():
        output_event = model(torch.from_numpy(mel_features).to(device).float().unsqueeze(1))
    output_event = output_event.cpu()

    logmel = mel_features[0] # takes the firs chennel only

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(output_event[0].T, origin='lower', aspect='auto', cmap='jet')

    axs[0].set_title('Log mel spectrogram', color='r')
    axs[1].set_title(f"Predicted sound events ", color='b')

    frames_num = output_event.shape[1]
    length_in_second = frames_num / float(cfg.frames_per_second)
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

    fig.tight_layout()
    os.makedirs(args.outputs_dir, exist_ok=True)
    plt.savefig(os.path.join(args.outputs_dir, f"{os.path.splitext(os.path.basename(args.audio_file))[0]}.png"))
    plt.close(fig)
