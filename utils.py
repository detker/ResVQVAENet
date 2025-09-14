from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch


@dataclass
class Config:
    dataset: str = 'MNIST'
    img_wh: int = 32
    latent_space_dim: int = 4
    in_channels: int = 1
    batch_size: int = 64
    iterations: int = 25000
    eval_intervals: int = 250
    num_workers: int = 8
    kl_divergence_weight: float = 1.0
    lr: float = 0.0005


class Plotter:
    @staticmethod
    def plot_latent_space_tsne(latent_space_representation):
        conv_ae_latent_space = latent_space_representation[-1]

        conv_ae_features = conv_ae_latent_space[:, :-1]
        conv_ae_lbls = conv_ae_latent_space[:, -1:]
        tsne = TSNE(2, n_jobs=-1)
        conv_ae_2d_compresses = tsne.fit_transform(X=conv_ae_features)

        conv_ae_encoding = np.hstack((conv_ae_2d_compresses, conv_ae_lbls))
        conv_ae_encoding = pd.DataFrame(conv_ae_encoding, columns=["x", "y", "class"])
        conv_ae_encoding = conv_ae_encoding.sort_values(by="class")
        conv_ae_encoding["class"] = conv_ae_encoding["class"].astype(int).astype(str)

        for grouper, group in conv_ae_encoding.groupby('class'):
            plt.scatter(x=group['x'], y=group['y'], label=grouper, alpha=0.8, s=5)


def VAELoss(x, x_hat, mu, logvar, alpha=1, beta=1):
    # alpha, beta - KL divergence, MSE weights respectively
    pixel_diff = (x - x_hat) * (x - x_hat)
    pixel_diff = pixel_diff.flatten(1)
    mse = pixel_diff.sum(axis=-1).mean()

    kl_divergence = (1 + logvar - mu * mu - torch.exp(logvar)).flatten(1)
    kl_per_sample = -0.5 * kl_divergence.sum(dim=-1)
    kl_loss = kl_per_sample.mean()

    return alpha * kl_loss + beta * mse
