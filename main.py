import os
import argparse
from tqdm import tqdm
from torch import optim

from utils import binary_crossentropy
from models import *
import config as cfg
from data import get_batch_generator


def train(model, data_generator, num_steps, outputs_dir, device):
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0., amsgrad=True)

    iterations = 0
    print("Training")
    for (mel_features, event_labels) in tqdm(data_generator.generate_train()):

        batch_features = mel_features.to(device).float()
        event_labels = event_labels.to(device).float()

        model.train()
        batch_outputs = model(batch_features)
        loss = binary_crossentropy(batch_outputs, event_labels)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterations+=1
        if iterations % 100 == 0:
            print(f"step: {iterations}, loss: {loss.item()}")

            checkpoint = {
                'iterations': iterations,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, os.path.join(outputs_dir, 'checkpoints', '{}_iterations.pth'.format(iterations)))

        if iterations == num_steps:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # Train
    parser.add_argument('--dataset_dir', type=str, default='../data', help='Directory of dataset.')
    parser.add_argument('--outputs_root', type=str, default='outputs', help='Directory of your workspace.')
    parser.add_argument('--audio_type', default='foa', type=str, choices=['foa', 'mic'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

    model = Cnn_9layers_AvgPooling(cfg.classes_num).to(device)

    data_generator = get_batch_generator(args.dataset_dir, args.batch_size, train_or_eval='eval')

    train(model, data_generator, num_steps=300, outputs_dir=args.outputs_root, device=device)
    # eval()