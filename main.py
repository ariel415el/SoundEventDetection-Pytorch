import argparse
import os

from torch.utils.data import DataLoader
from dataset.spectogram_features import spectogram_configs as cfg
from dataset.spectogram_features.spectograms_dataset import preprocess_film_clap_data, SpectogramDataset
from dataset.spectogram_features.spectogram_configs import cfg_descriptor
from models.spectogram_models import *
from train import train


def get_spectogram_dataset_and_model(args):
    model = Cnn_AvgPooling(cfg.classes_num, model_config=[(32,2), (64,2), (128,2), (128,1)]).to(device)
    # model = MobileNetV1(cfg.classes_num).to(device)
    if args.ckpt != '':
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # features_and_labels_dir, features_mean_std_file, dataset_name = preprocess_tau_sed_data(args.dataset_dir, mode='eval', force_preprocess=args.force_preprocess)
    features_and_labels_dir, features_mean_std_file, dataset_name = preprocess_film_clap_data(args.dataset_dir,
                                                                                              preprocessed_mode=args.preprocess_mode,
                                                                                              force_preprocess=args.force_preprocess)

    dataset = SpectogramDataset(features_and_labels_dir, features_mean_std_file,
                                       augment_data=args.augment_data,
                                       balance_classes=args.balance_classes,
                                       val_descriptor=args.val_descriptor,
                                       preprocessed_mode=args.preprocess_mode)

    return dataset, model, dataset_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # Train
    parser.add_argument('--dataset_dir', type=str, default='../data', help='Directory of dataset.')
    parser.add_argument('--outputs_root', type=str, default='training_dir')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--val_descriptor', default=0.15, help='float for percentage string for specifying fold substring')
    parser.add_argument('--train_tag', type=str, default='')
    parser.add_argument('--force_preprocess', action='store_true', default=False)

    parser.add_argument('--preprocess_mode', type=str, default='Complex', help='logMel or Complex')
    parser.add_argument('--augment_data', action='store_true', default=False, help='Only on logMel preprocess mode')
    parser.add_argument('--balance_classes', action='store_true', default=False,
                        help='Whether to make sure there is same number of samples with and without events')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.000003)
    parser.add_argument('--num_train_steps', type=int, default=30000)
    parser.add_argument('--log_freq', type=int, default=100)

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_workers', default=12, type=int)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")
    print(device)

    dataset, model, dataset_name = get_spectogram_dataset_and_model(args)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    train_name = f"{dataset_name}_cfg({cfg_descriptor})_b{args.batch_size}_lr{args.lr}_{args.train_tag}"
    if args.balance_classes:
        train_name += "_BC"
    if args.augment_data:
        train_name += "_AD"

    model.model_description()

    train(model, dataloader,
          num_steps=args.num_train_steps,
          outputs_dir=os.path.join(args.outputs_root, train_name),
          device=device,
          lr=args.lr,
          log_freq=args.log_freq)
