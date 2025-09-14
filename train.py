import os
import argparse

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from safetensors.torch import save_file

from models import ConvAE, ConvVAE
from utils import VAELoss, Plotter, Config

def add_arguments(parser):
    parser.add_argument('--_model_weights_dir',
                        type=str,
                        required=True,
                        help='Directory to save the model')
    parser.add_argument('--dataset',
                        type=str,
                        default='MNIST',
                        help='Dataset name')
    parser.add_argument('--img_wh',
                        type=int,
                        default=32,
                        help='Image width and height')
    parser.add_argument('--latent_space_dim',
                        type=int,
                        default=4,
                        help='Dimension of latent space')
    parser.add_argument('--in_channels',
                        type=int,
                        default=1,
                        help='Number of input channels')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--iterations',
                        type=int,
                        default=25000,
                        help='Number of training iterations')
    parser.add_argument('--eval_intervals',
                        type=int,
                        default=250,
                        help='Evaluation interval')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of data loading workers')
    parser.add_argument('--kl_divergence_weight',
                        type=float,
                        default=1.0,
                        help='Weight for KL divergence loss')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
                        help='Learning rate')
    parser.add_argument('--variational',
                        action=argparse.BooleanOptionalAction,
                        help='Enable live plotting')

    return parser

def setup_data_pipeline(config):
    data_folder_prefix = 'data'
    transform = transforms.Compose([transforms.Resize((config.img_wh, config.img_wh)),
                                    transforms.ToTensor()])
    if config.dataset == 'MNIST':
        train_dataset = MNIST(os.path.join(data_folder_prefix, 'mnist'),
                              train=True, transform=transform, download=True)
        test_dataset = MNIST(os.path.join(data_folder_prefix, 'mnist'),
                             train=False, transform=transform, download=True)
    else:
        raise Exception('Wrong dataset specified')

    trainloader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers)
    testloader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    return trainloader, testloader


def train(model, variational, trainloader, testloader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    train_losses, test_losses = [], []
    train_loss, test_loss = [], []
    enc_val_latents = []
    all_enc_val_latents = []
    steps_done = 0

    train = True
    pbar = tqdm(range(0, config.iterations))

    while train:
        for data, _ in trainloader:
            data = data.to(device)
            if variational:
                z, mu, lnvar, sigma, reconstruction = model(data)
                loss = VAELoss(data, reconstruction, mu, lnvar, alpha=1, beta=1)
            else:
                enc, dec = model(data)
                loss = torch.mean((dec-data) ** 2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())

            if steps_done % config.eval_intervals == 0:
                model.eval()

                for data, labels in testloader:
                    data = data.to(device)
                    if variational:
                        with torch.no_grad():
                            enc, mu, lnvar, sigma, reconstruction = model(data)
                        loss = VAELoss(data, reconstruction, mu, lnvar,
                                       alpha=config.kl_divergence_weight,
                                       beta=1-config.kl_divergence_weight)
                    else:
                        with torch.no_grad():
                            enc, dec = model(data)
                        loss = torch.mean((data-dec)**2)
                    test_loss.append(loss.item())

                    enc, labels = enc.cpu().flatten(1), labels.unsqueeze(1)
                    enc_val_latents.append(torch.cat([enc, labels], axis=-1))

                mean_train_loss = np.mean(train_loss)
                mean_test_loss = np.mean(test_loss)

                print(
                    f'Iteration: {steps_done}/{config.iterations}: Train Loss: {mean_train_loss:.4f} | Test Loss: {mean_test_loss:.4f}')

                train_losses.append(mean_train_loss)
                test_losses.append(mean_test_loss)

                train_loss, test_los = [], []

                all_enc_val_latents.append(torch.concatenate(enc_val_latents).detach())
                enc_val_latents = []

                model.train()

            steps_done += 1
            pbar.update(1)
            if steps_done >= config.iterations:
                train = False
                break

    all_enc_val_latents = [np.array(enc_eval) for enc_eval in all_enc_val_latents]
    log = {
        'train_loss': train_losses,
        'test_loss': test_losses
    }

    return model, log, all_enc_val_latents


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    config = Config(**{k:v for k,v in args.__dict__.items() if not k.startswith('_')})
    trainloader, testloader = setup_data_pipeline(config)
    model = ConvVAE(config) if config._variational else ConvAE(config)
    model, log, latent_space_representation = train(models=model,
                                                     variational=config._variational,
                                                     trainloader=trainloader,
                                                     testloader=testloader,
                                                     config=config,
                                                     device=device)

    os.makedirs(config._model_weights_dir, exist_ok=True)
    state_dict = model.state_dict()
    save_file(state_dict,
              os.path.join(config._model_weights_dir, 'vae' if config._variational else 'ae')+'.safetensors')

if __name__ == '__main__':
    main()
