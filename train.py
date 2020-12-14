import os
import argparse
from tqdm import tqdm
import numpy as np
from torch import optim

from utils import binary_crossentropy
from models import *
import config as cfg
from dataset.data_generator import get_tau_sed_generator, get_film_clap_generator
import matplotlib.pyplot as plt


class loss_tracker:
    def __init__(self, plot_path):
        self.train_buffer = []
        self.val_buffer = []
        self.train_avgs = []
        self.val_avgs = []
        self.plot_path = plot_path

    def plot(self):
        self.train_avgs += [np.mean(self.train_buffer)]
        self.val_avgs += [np.mean(self.val_buffer)]
        self.train_buffer = []
        self.val_buffer = []

        plt.plot(np.arange(len(self.train_avgs)), self.train_avgs, label='train', color='blue')
        plt.plot(np.arange(len(self.val_avgs)), self.val_avgs, label='validation', color='orange')
        plt.xlabel("train step")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(self.plot_path)
        plt.clf()


def eval(model, data_generator, num_samples, outputs_dir, iteration, device):
    os.makedirs(outputs_dir, exist_ok=True)
    losses = []
    for idx, (mel_features, event_matrix, file_name) in enumerate(data_generator.generate_validate('validate', max_validate_num=num_samples)):
        model.eval()
        with torch.no_grad():
            model.eval()
            output_event = model(mel_features.to(device).float())

        loss = binary_crossentropy(output_event.cpu().float(), event_matrix.float())
        losses.append(loss.item())
        output_event = output_event.cpu().numpy()
        event_matrix = event_matrix.cpu().numpy()

        # Plot outputs
        frames_num = event_matrix.shape[1]
        length_in_second = frames_num / float(cfg.frames_per_second)

        fig, axs = plt.subplots(3, 1, figsize=(15, 10))

        logmel = mel_features[0][0] * data_generator.std + data_generator.mean

        axs[0].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(event_matrix[0].T, origin='lower', aspect='auto', cmap='jet')
        axs[2].matshow(output_event[0].T, origin='lower', aspect='auto', cmap='jet')

        axs[0].set_title('Log mel spectrogram', color='r')
        axs[1].set_title('Reference sound events', color='r')
        axs[2].set_title(f"Predicted sound events (loss: {loss:.2f})", color='b')

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

        # fig.tight_layout()
        plt.savefig(os.path.join(outputs_dir, f"Iter-{iteration}_img-{idx}.png"))
        plt.close(fig)

        return losses


def train(model, data_generator, num_steps, outputs_dir, device):
    plotter = loss_tracker(os.path.join(outputs_dir, "training_plot.png"))
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0., amsgrad=True)
    os.makedirs(os.path.join(outputs_dir, 'checkpoints'), exist_ok=True)
    iterations = 0
    print("Training")
    train_losses = []
    for (mel_features, event_labels) in tqdm(data_generator.generate_train()):
        batch_features = mel_features.to(device).float()
        event_labels = event_labels.to(device).float()

        model.train()
        batch_outputs = model(batch_features)
        loss = binary_crossentropy(batch_outputs, event_labels)
        plotter.train_buffer.append(loss.item())
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterations+=1

        if iterations % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95

        if iterations % 1000 == 0:
            print(f"step: {iterations}, loss: {loss.item():.2f}")

            val_losses = eval(model, data_generator, 10, outputs_dir=os.path.join(outputs_dir, 'images'), iteration=iterations, device=device)
            plotter.val_buffer += val_losses
            plotter.plot()

            checkpoint = {
                'iterations': iterations,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, os.path.join(outputs_dir, 'checkpoints', f"iterations_{iterations}.pth"))

        if iterations == num_steps:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # Train
    parser.add_argument('--dataset_dir', type=str, default='data', help='Directory of dataset.')
    parser.add_argument('--outputs_root', type=str, default='outputs', help='Directory of your workspace.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

    model = Cnn_9layers_AvgPooling(cfg.classes_num).to(device)

    data_generator = get_tau_sed_generator(args.dataset_dir, args.batch_size, train_or_eval='dev')
    # data_generator = get_film_clap_generator("../data/Film_take_clap", args.batch_size)

    train(model, data_generator, num_steps=5000, outputs_dir=args.outputs_root, device=device)
