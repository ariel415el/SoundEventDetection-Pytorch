import argparse
import os
from models import *
import config as cfg
from dataset.preprocess import LogMelExtractor, read_multichannel_audio
from utils import plot_debug_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # Train
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, default='inference_outputs', help='Directory of your workspace.')
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

    model = Cnn_AvgPooling(cfg.classes_num).to(device)
    # checkpoint = torch.load(args.ckpt, map_location=device)
    # model.load_state_dict(checkpoint['model'])

    print("Preprocessing audio file..")
    feature_extractor = LogMelExtractor(
        sample_rate=cfg.working_sample_rate,
        nfft=cfg.NFFT,
        window_size=cfg.frame_size,
        hop_size=cfg.hop_size,
        mel_bins=cfg.mel_bins,
        fmin=cfg.mel_min_freq,
        fmax=cfg.mel_max_freq)

    multichannel_audio, _ = read_multichannel_audio(audio_path=args.audio_file, target_fs=cfg.working_sample_rate)

    print("Inference..")
    mel_features = feature_extractor.transform_multichannel(multichannel_audio)

    with torch.no_grad():
        output_event = model(torch.from_numpy(mel_features).to(device).float().unsqueeze(1))
    output_event = output_event.cpu()
    os.makedirs(args.outputs_dir, exist_ok=True)

    plot_debug_image(mel_features[0], output=output_event[0], plot_path=os.path.join(args.outputs_dir, f"{os.path.splitext(os.path.basename(args.audio_file))[0]}.png"))

