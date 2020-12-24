import os
import argparse
from tqdm import tqdm
from torch import optim
import numpy as np
from utils import binary_crossentropy, loss_tracker, calculate_metrics, plot_debug_image
from models import *
import config as cfg
from dataset.data_generator import get_film_clap_generator, get_tau_sed_generator


def eval(model, data_generator, outputs_dir, iteration, device, limit_val_samples=4):
    losses = []
    recal_sets, precision_sets, max_f1_vals = [], [], []
    for idx, (mel_features, target, file_name) in enumerate(data_generator.generate_validate('validate', max_validate_num=limit_val_samples)):
        model.eval()
        with torch.no_grad():
            model.eval()
            output = model(mel_features.to(device).float()).cpu()

        loss = binary_crossentropy(output, target.float())

        output = output.numpy()
        target = target.numpy()

        f1_vals, recal_vals, precision_vals = calculate_metrics(output, target)

        losses.append(loss.item())
        recal_sets.append(recal_vals)
        precision_sets.append(precision_vals)
        max_f1_vals.append(np.max(f1_vals))

        unormelized_mel = mel_features[0][0] * data_generator.std + data_generator.mean
        plot_debug_image(unormelized_mel, output=output[0], target=target[0], plot_path=os.path.join(outputs_dir, f"Iter-{iteration}_img-{idx}.png"))

    return losses, max_f1_vals, recal_sets, precision_sets


def train(model, data_generator, num_steps, log_freq, outputs_dir, device):
    lr_decay_freq = 100
    plotter = loss_tracker(outputs_dir)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    os.makedirs(os.path.join(outputs_dir, 'checkpoints'), exist_ok=True)

    iterations = 0
    print("Training")
    for (batch_features, event_labels) in tqdm(data_generator.generate_train()):
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
            print(f"step: {iterations}, loss: {loss.item():.2f}")
            losses, max_f1_vals, recal_sets, precision_sets = eval(model, data_generator,
                                                                   outputs_dir=os.path.join(outputs_dir, 'images'),
                                                                   iteration=iterations,
                                                                   device=device)

            plotter.report_val_losses(losses)
            plotter.report_val_metrics({'max_f1_score':max_f1_vals})
            plotter.report_val_roc(recal_sets, precision_sets)

            plotter.plot()

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
    parser.add_argument('--outputs_root', type=str, default='outputs')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train_steps', type=int, default=5000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--force_preprocess', action='store_true', default=False)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

    model = Cnn_AvgPooling(cfg.classes_num, model_config=[(64,2), (128,2), (256,2), (512,1)]).to(device)
    model.model_description()
    if args.ckpt != '':
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
    # data_generator = get_tau_sed_generator(args.dataset_dir, args.batch_size, train_or_eval='dev', force_preprocess=args.force_preprocess)
    data_generator = get_film_clap_generator("../data/Film_take_clap", args.batch_size, force_preprocess=args.force_preprocess)

    train(model, data_generator, num_steps=args.num_train_steps, outputs_dir=args.outputs_root, device=device, log_freq=args.log_freq)
