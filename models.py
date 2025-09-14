import torch
import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.latent_dim = config.latent_space_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=self.latent_dim, kernel_size=3, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_dim, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def enc_forward(self, x):
        return self.encoder(x)

    def dec_forward(self, x):
        return self.decoder(x)

    def forward(self, x):
        enc = self.enc_forward(x)
        dec = self.dec_forward(enc)

        return enc, dec


class ConvVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.latent_dim = config.latent_space_dim
        self.kl_divergence_weight = config.kl_divergence_weight

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.gaussian_params = nn.Conv2d(in_channels=32, out_channels=2*self.latent_dim, kernel_size=3, stride=1, padding="same")

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_dim, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def enc_forward(self, x):
        x = self.encoder(x) # (B, 2latent_dim=8, 4, 4)
        mu, lnvar = self.gaussian_params(x).chunk(2, dim=1) #(B, 4, 4, 4) x2
        sigma = torch.exp(0.5 * lnvar)
        # Reparametrization trick
        eps = torch.randn_like(sigma) # eps ~ N(0,1) r
        z = mu + sigma*eps # (B, 4, 4, 4) z ~ N(mu, sigma)

        return z, mu, lnvar, sigma

    def dec_forward(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def forward(self, x):
        z, mu, lnvar, sigma = self.enc_forward(x)
        reconstructed = self.dec_forward(z)
        return z, mu, lnvar, sigma, reconstructed
