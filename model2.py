import torch
import torch.nn as nn
import torch.nn.functional as F


class EncResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, downsample=None, middle_conv_stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=middle_conv_stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4 * planes, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity

        return x


class DecResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, upsample=None, middle_conv_stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(planes * 4)
        self.conv1 = nn.Conv2d(planes * 4, planes, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) if middle_conv_stride == 1 else \
            nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, in_planes, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()

        self.upsample = upsample

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        x = x + identity

        return x

# Simple residual block (like in VQGAN)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.conv1(F.relu(self.norm1(x)))
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.skip(x)

# Decoder with progressive upsampling
class VQGANDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.block1 = ResBlock(latent_dim, 256)  # [B, 512, H, W] -> [B, 256, H, W]
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(256, 256, 3, padding=1))

        self.block2 = ResBlock(256, 128)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(128, 128, 3, padding=1))

        self.block3 = ResBlock(128, 64)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(64, 64, 3, padding=1))

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, padding=1)
        )

        self.final_block = ResBlock(64, 64)
        self.out = nn.Conv2d(64, 3, 1)  # RGB output
        self.gate = nn.Sigmoid()

    def forward(self, z):
        # z: [B, latent_dim, H, W]  (e.g., [B, 512, 16, 16])
        x = self.block1(z)
        x = self.up1(x)

        x = self.block2(x)
        x = self.up2(x)

        x = self.block3(x)
        x = self.up3(x)

        x = self.up4(x)

        x = self.final_block(x)
        x = self.out(x)

        return self.gate(x) # keep outputs in [0,1]

class VectorQuantizer(nn.Module):
    def __init__(self, latent_dim=2, codebook_size=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(self, x):
        # x: (B, E)
        # embd: (L, E)
        # we want to compute distances between x and embd

        # (we want (x-embd)**2 = x**2 - 2*x*embd + embd**2)
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)  # (B, 1)
        x_embd = x @ self.embedding.weight.t()  # (B, L)
        embd2 = torch.sum(self.embedding.weight ** 2, dim=-1)  # (L,)
        distances = x2 - 2 * x_embd + embd2  # (B, L)

        indicies = torch.argmin(distances, dim=-1)

        quantized_latents = self.embedding(indicies)

        return quantized_latents, indicies  # (B, E)


class ResEncoder(nn.Module):
    def __init__(self, layer_counts, in_channels=3):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn0 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()

        self.in_planes = 64

        self.layer0 = self._make_layers(layer_count=layer_counts[0], planes=64, stride=1)
        self.layer1 = self._make_layers(layer_count=layer_counts[1], planes=128, stride=2)
        self.layer2 = self._make_layers(layer_count=layer_counts[2], planes=256, stride=2)
        self.layer3 = self._make_layers(layer_count=layer_counts[3], planes=512, stride=2)

    def _make_layers(self, layer_count, planes, stride):
        downsample = None
        layers = nn.ModuleList()

        if stride != 1 or self.in_planes != planes * 4:
            # downsample = nn.Sequential(
            #     nn.BatchNorm2d(self.in_planes),
            #     nn.ReLU(),
            #     nn.Conv2d(self.in_planes, planes * 4, kernel_size=1, stride=stride)
            # )
            downsample = nn.Conv2d(self.in_planes, planes * 4, kernel_size=1, stride=stride)

        # self.in_planes -> planes -> planes*4
        # 64 -> 64 -> 256
        # 256 -> 128 -> 512
        # 512 -> 256 -> 1024
        # 1024 -> 512 -> 2048
        layers.append(EncResidualBlock(
            in_planes=self.in_planes,
            planes=planes,
            downsample=downsample,
            middle_conv_stride=stride
        ))

        self.in_planes = 4 * planes
        # 256
        # 512
        # 1024
        # 2048

        # 256 -> 64 -> 256
        # 512 -> 128 -> 512
        # 1024 -> 256 -> 1024
        # 2048 -> 512 -> 2048
        for _ in range(1, layer_count):
            layers.append(EncResidualBlock(
                in_planes=self.in_planes,
                planes=planes
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        # x = self.bn0(x)
        # x = self.relu(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (B, latent_dim, H', W')


class ResDecoder(nn.Module):
    def __init__(self, layer_counts, out_channels=3):
        super().__init__()

        self.in_planes = 2048

        self.layer0 = self._make_layers(layer_count=layer_counts[0], planes=512, stride=2)
        self.layer1 = self._make_layers(layer_count=layer_counts[1], planes=256, stride=2)
        self.layer2 = self._make_layers(layer_count=layer_counts[2], planes=128, stride=2)
        self.layer3 = self._make_layers(layer_count=layer_counts[3], planes=64, stride=1)

        # self.bn = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=7, stride=2,
                                                 padding=3, output_padding=1, bias=False)
        self.gate = nn.Sigmoid()

    def _make_layers(self, layer_count, planes, stride):
        upsample = None
        layers = nn.ModuleList()

        # planes*4 -> planes -> in_planes
        # 2048 -> 512 -> 2048
        # 1024 -> 256 -> 1024
        # 512 -> 128 -> 512
        # 256 -> 64 -> 256
        for _ in range(0, layer_count - 1):
            layers.append(DecResidualBlock(
                in_planes=self.in_planes,
                planes=planes
            ))

        assert planes % 2 == 0
        self.in_planes //= 2
        if stride == 1: self.in_planes //= 2
        # in_planes: 1024
        # 512
        # 256
        # 64

        if stride != 1 or planes * 4 != self.in_planes:
            # upsample = nn.Sequential(
            #     nn.BatchNorm2d(planes * 4),
            #     nn.ReLU(),
            #     nn.ConvTranspose2d(planes * 4, self.in_planes, kernel_size=1, stride=2, output_padding=1) if stride != 1 \
            #     else nn.Conv2d(planes*4, self.in_planes, kernel_size=1, stride=1)
            # )
            upsample = nn.ConvTranspose2d(planes * 4, self.in_planes, kernel_size=1, stride=2, output_padding=1) if stride != 1 \
                else nn.Conv2d(planes * 4, self.in_planes, kernel_size=1, stride=1)

        # planes*4 -> planes -> in_planes
        # 2048 -> 512 -> 1024
        # 1024 -> 256 -> 512
        # 512 -> 128 -> 256
        # 256 -> 64 -> 64
        layers.append(DecResidualBlock(
            in_planes=self.in_planes,
            planes=planes,
            upsample=upsample,
            middle_conv_stride=stride
        ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.conv_transpose(x)
        x = self.gate(x)
        return x  # (B, C, H, W)


class ConvResidualVQVAE(nn.Module):
    def __init__(self, layer_counts, in_channels=3, latent_dim=2048, codebook_size=128, n_codebooks=4):
        super().__init__()

        self.layer_counts = layer_counts
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks

        self.encoder = ResEncoder(layer_counts=layer_counts, in_channels=in_channels)

        self.conv_proj = nn.Conv2d(latent_dim, 512, kernel_size=1, stride=1)

        self.vqs = nn.ModuleList([
            VectorQuantizer(latent_dim=512, codebook_size=256)
            for _ in range(n_codebooks)
        ])

        # self.decoder = ResDecoder(layer_counts=layer_counts[::-1], out_channels=in_channels)
        self.decoder = VQGANDecoder(latent_dim=512)

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def enc_forward(self, x):
        # x: (B, in_channels, H, W)
        z = self.encoder(x)
        # print(z.shape)
        return z  # (B, latent_dim, H', W')

    def quantize(self, z):
        commitment_losses = 0
        codebook_losses = 0
        codebooks_usage_ratio = []
        codebooks_perplexities = []

        accum_quantized_latent = torch.zeros_like(z, device=z.device)

        # z: (B*H'*W', latent_dim)
        for idx, quantizer in enumerate(self.vqs):
            codes, indices = quantizer(z)  # codes: (B'*H'*W', latent_dim)

            codebook_losses += torch.mean((codes - z.detach()) ** 2)
            commitment_losses += torch.mean((z - codes.detach()) ** 2)

            codes = z + (codes - z).detach()

            accum_quantized_latent += codes

            if not self.training:
                usage_ratio = indices.unique().numel() / self.codebook_size
                usage_ratio = torch.tensor(usage_ratio, device=z.device, dtype=z.dtype, requires_grad=False)
                codebooks_usage_ratio.append(usage_ratio)
                counts = torch.bincount(indices, minlength=self.codebook_size).float().requires_grad_(False).to(z.device)
                probs = counts / counts.sum()
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                perplexity = torch.exp(entropy)
                codebooks_perplexities.append(perplexity)

            z = z - codes.detach()

        codebooks_usage_ratio = torch.tensor(codebooks_usage_ratio, device=z.device)
        codebooks_perplexities = torch.tensor(codebooks_perplexities, device=z.device)

        return accum_quantized_latent, codebook_losses, commitment_losses, codebooks_usage_ratio, codebooks_perplexities

    def dec_forward(self, z):
        # z: (B, latent_dim, H', W')
        batch_size, latent_dim, h, w = z.shape
        z_prim = z.permute(0, 2, 3, 1).contiguous().reshape(-1, latent_dim)  # (B*H'*W', latent_dim)
        quantized_z, codebook_loss, commitment_loss, codebooks_usage_ratios, codebooks_perplexities = self.quantize(z_prim)
        quantized_z = quantized_z.reshape(batch_size, h, w, latent_dim).permute(0, 3, 1, 2).contiguous()  # (B, latent_dim, H', W')
        # print(quantized_z.shape)
        # print(z.shape)
        x = quantized_z + 0.05*z
        dec = self.decoder(x)  # (B, C, H, W)

        return dec, quantized_z, codebook_loss, commitment_loss, codebooks_usage_ratios, codebooks_perplexities

    def forward(self, x):
        # x: (B, C, H, W)
        latent = self.enc_forward(x)  # (B, latent_dim, H', W')
        latent_proj = self.conv_proj(latent) # (B, proj_latent_dim, H', W')
        # print(latent_proj.shape)
        dec, quantized_latent, codebook_loss, commitment_loss, codebooks_usage_ratios, codebooks_perplexities = self.dec_forward(latent_proj)
        # print(dec.shape)
        # dec: (B, C, H, W); quantized_latent: (B, latent_dim, H', W')
        return latent, quantized_latent, dec, codebook_loss, commitment_loss, codebooks_usage_ratios, codebooks_perplexities

def ConvResidualVQVAE_ResNet50Backbone(in_channels=3):
    return ConvResidualVQVAE(layer_counts=[3,4,6,3], in_channels=in_channels)

def ConvResidualVQVAE_ResNet101Backbone(in_channels=3):
    return ConvResidualVQVAE(layer_counts=[3,4,23,3], in_channels=in_channels)

def ConvResidualVQVAE_ResNet150Backbone(in_channels=3):
    return ConvResidualVQVAE(layer_counts=[3,8,36,3], in_channels=in_channels)

if __name__ == '__main__':
    rvq = ConvResidualVQVAE(layer_counts=[3, 4, 6, 3], in_channels=3)
    x = torch.rand(size=(2, 3, 224, 224))
    rvq(x)
    # print([y for y in rvq(x)])
    # print([name for name,p in rvq.named_parameters()])
    #
    # params = 0
    # for p in rvq.parameters():
    #     params += p.numel()
    # print(params)

    # latent, quantized_latent, dec, codebook_loss, commitment_loss = rvq(x)
    # print(quantized_latent.unique().numel() / 1024)
    # print(quantized_latent)
