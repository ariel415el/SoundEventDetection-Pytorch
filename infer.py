import argparse
import os
from models import *
from dataset.spectogram_features import spectogram_configs as cfg
from dataset.spectogram_features.preprocess import multichannel_stft, multichannel_complex_to_log_mel
from dataset.dataset_utils import read_multichannel_audio
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

    multichannel_audio = read_multichannel_audio(audio_path=args.audio_file, target_fs=cfg.working_sample_rate)

    log_mel_features = multichannel_complex_to_log_mel(multichannel_stft(multichannel_audio))[0]

    print("Inference..")
    with torch.no_grad():
        output_event = model(torch.from_numpy(log_mel_features).to(device).float().unsqueeze(1))
    output_event = output_event.cpu()
    os.makedirs(args.outputs_dir, exist_ok=True)

    plot_debug_image(log_mel_features, output=output_event[0], plot_path=os.path.join(args.outputs_dir, f"{os.path.splitext(os.path.basename(args.audio_file))[0]}.png"))