import os
import argparse
from tqdm import tqdm
from torch import optim
from utils import binary_crossentropy, ProgressPlotter, calculate_metrics, plot_debug_image
from models import *
import config as cfg
from dataset.data_generator import get_film_clap_generator, get_tau_sed_generator, DataGenerator
from time import time
import numpy as np

def eval(model, data_generator, outputs_dir, iteration, device, limit_val_samples=32):
    losses = []
    recal_sets, precision_sets, max_f1_vals = [], [], []
    outputs = []
    targets = []
    inputs = []
    for idx, (mel_features, target, file_name) in enumerate(data_generator.generate_validate('validate', max_validate_num=limit_val_samples)):
    # for idx, (mel_features, target) in enumerate(data_generator.generate_train()):
        model.eval()
        with torch.no_grad():
            model.eval()
            output = model(mel_features.to(device).float()).cpu()

        loss = binary_crossentropy(output, target.float())

        output = output.numpy()
        target = target.numpy()

        recal_vals, precision_vals = calculate_metrics(output, target)

        losses.append(loss.item())
        recal_sets.append(recal_vals)
        precision_sets.append(precision_vals)

        outputs.append(output)
        targets.append(target)
        inputs.append(mel_features)

    for (name, idx) in [("best", np.argmin(losses)), ("worst", np.argmax(losses))]:
        mel_features, output, target = inputs[idx], outputs[idx], targets[idx]
        unormelized_mel = mel_features[0][0] * data_generator.std + data_generator.mean
        plot_debug_image(unormelized_mel, output=output[0], target=target[0], file_name=file_name + f" loss {losses[idx]:.2f}",
                         plot_path=os.path.join(outputs_dir, 'images', f"Iter-{iteration}_{name}.png"))

    return losses, recal_sets, precision_sets


def train(model, data_generator, num_steps, lr, log_freq, outputs_dir, device):
    lr_decay_freq = 100
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
        loss = binary_crossentropy(batch_outputs, event_labels.to(device).float())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plotter.report_train_loss(loss.item())
        iterations+=1

        if iterations % lr_decay_freq == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95

        if iterations % log_freq == 0:
            im_sec = iterations * data_generator.batch_size / (time() - training_start_time)
            tqdm_bar.set_description(f"step: {iterations}, loss: {loss.item():.2f}, im/sec: {im_sec:.1f}")

            val_losses, recal_sets, precision_sets = eval(model, data_generator, outputs_dir, iteration=iterations, device=device)

            plotter.report_validation_metrics(val_losses, recal_sets, precision_sets, iterations)
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--val_perc', type=float, default=0.15)
    parser.add_argument('--num_train_steps', type=int, default=5000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--force_preprocess', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--balance_classes', action='store_true', default=False,
                        help='Whether to make sure there is same number of samples with and without events')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")
    print(device)
    model = Cnn_AvgPooling(cfg.classes_num, model_config=[(64,2), (128,2), (256,2), (512,1)]).to(device)
    model.model_description()
    if args.ckpt != '':
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # features_and_labels_dir, features_mean_std_file, dataset_name = get_tau_sed_generator(args.dataset_dir, train_or_eval='eval', force_preprocess=args.force_preprocess)
    features_and_labels_dir, features_mean_std_file, dataset_name = get_film_clap_generator("../data/Film_take_clap", force_preprocess=args.force_preprocess)

    data_generator = DataGenerator(features_and_labels_dir, features_mean_std_file,
                                   batch_size=args.batch_size,
                                   augment_data=args.augment_data,
                                   balance_classes=args.balance_classes,
                                   val_descriptor=args.val_perc)

    train_name = f"{dataset_name}_b{args.batch_size}_lr{args.lr}"
    if args.balance_classes:
        train_name += "_BC"
    if args.augment_data:
        train_name += "_AD"

    train(model, data_generator,
          num_steps=args.num_train_steps,
          outputs_dir=os.path.join(args.outputs_root, train_name),
          device=device,
          lr=args.lr,
          log_freq=args.log_freq)
