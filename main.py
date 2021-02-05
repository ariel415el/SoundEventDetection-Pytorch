import argparse
import os

import torch
from torch.utils.data import DataLoader
from train import train
from utils.common import WeightedBCE


def get_spectogram_dataset_model_and_criterion(args):
    from dataset.spectogram_features.spectograms_dataset import preprocess_film_clap_data, SpectogramDataset, preprocess_tau_sed_data
    from dataset.spectogram_features import spectogram_configs as cfg
    from models.spectogram_models import Cnn_AvgPooling

    # Define the dataset
    if args.dataset_name.lower() == "tau":
        features_and_labels_dir, features_mean_std_file = preprocess_tau_sed_data(args.dataset_dir,
                                                                                  fold_name='eval',
                                                                                  preprocess_mode=args.preprocess_mode,
                                                                                  force_preprocess=args.force_preprocess)
    elif args.dataset_name.lower() == "filmclap":
        features_and_labels_dir, features_mean_std_file = preprocess_film_clap_data(args.dataset_dir,
                                                                                    preprocessed_mode=args.preprocess_mode,
                                                                                    force_preprocess=args.force_preprocess)
    else:
        raise ValueError(f"Only tau and filmclap datasets are supported, '{args.dataset_name}' given")

    dataset = SpectogramDataset(features_and_labels_dir, features_mean_std_file,
                                       augment_data=args.augment_data,
                                       balance_classes=args.balance_classes,
                                       val_descriptor=args.val_descriptor,
                                       preprocessed_mode=args.preprocess_mode)

    # Define the model
    model = Cnn_AvgPooling(cfg.classes_num, model_config=[(32,2), (64,2), (128,2), (128,1)])
    # model = MobileNetV1(cfg.classes_num)
    if args.ckpt != '':
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # define the crieterion
    criterion = WeightedBCE(recall_factor=args.recall_priority, multi_frame=True)

    return dataset, model, criterion, cfg.cfg_descriptor


def get_waveform_dataset_and_model(args):
    from dataset.waveform.waveform_dataset import WaveformDataset
    from dataset.waveform.waveform_configs import cfg_descriptor, time_margin
    from models.waveform_models import M5
    from dataset.dataset_utils import get_film_clap_paths_and_labels, get_tau_sed_paths_and_labels
    from dataset.download_tau_sed_2019 import ensure_tau_data

    if args.dataset_name.lower() == "tau":
        audio_dir, meta_data_dir = ensure_tau_data(f"{args.dataset_dir}/Tau_sound_events_2019", fold_name='eval')
        audio_paths_labels_and_names = get_tau_sed_paths_and_labels(audio_dir, meta_data_dir)
    elif args.dataset_name.lower() == "filmclap":
        audio_paths_labels_and_names = get_film_clap_paths_and_labels(os.path.join(args.dataset_dir, 'FilmClap', 'raw'), time_margin)
    else:
        raise ValueError(f"Only tau and filmclap datasets are supported, '{args.dataset_name}' given")

    dataset = WaveformDataset(audio_paths_labels_and_names,
                              augment_data=args.augment_data,
                              balance_classes=args.balance_classes,
                              val_descriptor=args.val_descriptor
                              )
    model = M5(1)

    criterion = WeightedBCE(recall_factor=args.recall_priority, multi_frame=False)


    return dataset, model, criterion, cfg_descriptor

def get_dataset_and_model(args):
    if args.train_features.lower() == "spectogram":
        return get_spectogram_dataset_model_and_criterion(args)
    elif args.train_features.lower() == "waveform":
        return get_waveform_dataset_and_model(args)
    else:
        raise ValueError(f"training features can be raw waveform or spectogram only, '{args.train_features}' given")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # Traininng
    parser.add_argument('--dataset_name', type=str, default='FilmClap', help='FilmClap or TAU')
    parser.add_argument('--train_features', type=str, default='Waveform', help='Spectogram or Waveform')
    parser.add_argument('--preprocess_mode', type=str, default='Complex', help='logMel or Complex; relevant only for Spectogram features')
    parser.add_argument('--force_preprocess', action='store_true', default=False, help='relevant only for Spectogram features')

    # Train
    parser.add_argument('--dataset_dir', type=str, default='../data', help='Directory of dataset.')
    parser.add_argument('--outputs_root', type=str, default='training_dir')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--val_descriptor', default=0.15, help='float for percentage string for specifying fold substring')
    parser.add_argument('--train_tag', type=str, default='')

    # Training tricks
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--balance_classes', action='store_true', default=False,
                        help='Whether to make sure there is same number of samples with and without events')
    parser.add_argument('--recall_priority', type=float, default=10, help='priority factor for the bce loss')

    # Hyper parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0000005)
    parser.add_argument('--num_train_steps', type=int, default=1e+6)
    parser.add_argument('--log_freq', type=int, default=1000)

    # Infrastructure
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_workers', default=12, type=int)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

    dataset, model, criterion, cfg_descriptor = get_dataset_and_model(args)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = model.to(device)
    model.model_description()

    train_name = f"{args.dataset_name}_cfg({cfg_descriptor}_b{args.batch_size}_lr{args.lr}_{args.train_tag}"
    if args.balance_classes:
        train_name += "_BC"
    if args.augment_data:
        train_name += "_AD"

    train(model, dataloader, criterion,
          num_steps=args.num_train_steps,
          outputs_dir=os.path.join(args.outputs_root, train_name),
          device=device,
          lr=args.lr,
          log_freq=args.log_freq)
