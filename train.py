import os
from tqdm import tqdm
import torch
from torch import optim
from utils.common import clip_and_aply_criterion, ProgressPlotter
from utils.metric_utils import calculate_metrics
from utils.plot_utils import plot_debug_image
from time import time
import numpy as np


def eval(model, dataloader, outputs_dir, iteration, device, limit_val_samples=None):
    losses = []
    recal_sets, precision_sets, APs = [], [], []
    debug_outputs = []
    debug_targets = []
    debug_inputs = []
    debug_file_names = []
    val_sampler = dataloader.dataset.get_validation_sampler(max_validate_num=limit_val_samples)
    for idx, (mel_features, target, file_name) in enumerate(val_sampler):

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
            unormelized_mel = debug_inputs[val_sample_idx][0][0]#  * data_generator.std + data_generator.mean
            plot_debug_image(unormelized_mel, output=debug_outputs[val_sample_idx][0],
                             target=debug_targets[val_sample_idx][0],
                             file_name=debug_file_names[val_sample_idx] + f" {metric_name} {values[val_sample_idx]:.2f}",
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
    tqdm_bar = tqdm(data_generator)
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
