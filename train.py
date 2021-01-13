import os
import argparse
from tqdm import tqdm
from torch import optim
from utils import clip_and_aply_criterion, ProgressPlotter, calculate_metrics, plot_debug_image, f_score
from models import *
import config as cfg
from dataset.spectograms_dataset import preprocess_film_clap_data, preprocess_tau_sed_data, SpectogramGenerator, cfg_descriptor
from time import time
import numpy as np


def eval(model, data_generator, outputs_dir, iteration, device, limit_val_samples=None):
    losses = []
    recal_sets, precision_sets, APs = [], [], []
    debug_outputs = []
    debug_targets = []
    debug_inputs = []
    debug_file_names = []
    for idx, (mel_features, target, file_name) in enumerate(
            data_generator.generate_validate('validate', max_validate_num=limit_val_samples)):
        # for idx, (mel_features, target) in enumerate(data_generator.generate_train()):
        #     file_name = "NA"
        model.eval()
        with torch.no_grad():
            model.eval()
            output = model(mel_features.to(device).float()).cpu()

        loss = clip_and_aply_criterion(output, target.float())

        output_logits = torch.sigmoid(output).numpy()
        target = target.numpy()

        recal_vals, precision_vals, AP = calculate_metrics(output_logits, target)

        losses.append(loss.item())
        recal_sets.append(recal_vals)
        precision_sets.append(precision_vals)
        APs.append(AP)

        debug_outputs.append(output_logits)
        debug_targets.append(target)
        debug_inputs.append(mel_features)
        debug_file_names.append(file_name)

    # plot input, outputs and targets of worst and best samples by each metric
    for (metric_name, values, named_indices) in [
                                        ("loss", losses, [('worst', -1), ('2-worst', -2), ('3-worst', -3), ('best', 0)]),
                                        ('AP', APs, [('worst', 0), ('best', -1)])]:
        indices = np.argsort(values)
        for (name, idx) in named_indices:
            val_sample_idx = indices[idx]
            unormelized_mel = debug_inputs[val_sample_idx][0][0] * data_generator.std + data_generator.mean
            plot_debug_image(unormelized_mel, output=debug_outputs[val_sample_idx][0],
                             target=debug_targets[val_sample_idx][0],
                             file_name=debug_file_names[
                                           val_sample_idx] + f" {metric_name} {values[val_sample_idx]:.2f}",
                             plot_path=os.path.join(outputs_dir, 'images', f"Iter-{iteration}",
                                                    f"{metric_name}-{name}.png"))

    return losses, recal_sets, precision_sets, APs


def train(model, data_generator, num_steps, lr, log_freq, outputs_dir, device):
    lr_decay_freq = 200
    plotter = ProgressPlotter()
    os.makedirs(os.path.join(outputs_dir, 'checkpoints'), exist_ok=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    iterations = 0
    print("Training")
    tqdm_bar = tqdm(data_generator.generate_train())
    training_start_time = time()
    for (batch_features, event_labels) in tqdm_bar:
        # forward
        model.train()
        batch_outputs = model(batch_features.to(device).float())
        loss = clip_and_aply_criterion(batch_outputs, event_labels.to(device).float())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plotter.report_train_loss(loss.item())
        iterations += 1

        if iterations % lr_decay_freq == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.99

        if iterations % log_freq == 0:
            im_sec = iterations * data_generator.batch_size / (time() - training_start_time)
            tqdm_bar.set_description(
                f"step: {iterations}, loss: {loss.item():.2f}, im/sec: {im_sec:.1f}, lr: {optimizer.param_groups[0]['lr']:.8f}")

            val_losses, recal_sets, precision_sets, APs = eval(model, data_generator, outputs_dir, iteration=iterations,
                                                          device=device)

            plotter.report_validation_metrics(val_losses, recal_sets, precision_sets, APs, iterations)
            plotter.plot(outputs_dir)

            checkpoint = {
                'iterations': iterations,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, os.path.join(outputs_dir, 'checkpoints', f"iteration_{iterations}.pth"))

        if iterations == num_steps:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # Train
    parser.add_argument('--dataset_dir', type=str, default='../data', help='Directory of dataset.')
    parser.add_argument('--outputs_root', type=str, default='training_dir')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.000003)
    parser.add_argument('--val_perc', type=float, default=0.15)
    parser.add_argument('--num_train_steps', type=int, default=30000)
    parser.add_argument('--log_freq', type=int, default=500)
    parser.add_argument('--train_tag', type=str, default='')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--force_preprocess', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--balance_classes', action='store_true', default=False,
                        help='Whether to make sure there is same number of samples with and without events')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")
    print(device)
    model = Cnn_AvgPooling(cfg.classes_num, model_config=[(32,2), (64,2), (128,2), (128,1)]).to(device)
    # model = MobileNetV1(cfg.classes_num).to(device)
    if args.ckpt != '':
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])

    features_and_labels_dir, features_mean_std_file, dataset_name = preprocess_tau_sed_data(args.dataset_dir, mode='eval', force_preprocess=args.force_preprocess)
    # features_and_labels_dir, features_mean_std_file, dataset_name = preprocess_film_clap_data(args.dataset_dir, force_preprocess=args.force_preprocess)

    data_generator = SpectogramGenerator(features_and_labels_dir, features_mean_std_file,
                                         batch_size=args.batch_size,
                                         augment_data=args.augment_data,
                                         balance_classes=args.balance_classes,
                                         val_descriptor='DyingWithYou')

    train_name = f"{dataset_name}_cfg({cfg_descriptor})_b{args.batch_size}_lr{args.lr}_{args.train_tag}"
    if args.balance_classes:
        train_name += "_BC"
    if args.augment_data:
        train_name += "_AD"

    model.model_description()

    train(model, data_generator,
          num_steps=args.num_train_steps,
          outputs_dir=os.path.join(args.outputs_root, train_name),
          device=device,
          lr=args.lr,
          log_freq=args.log_freq)
