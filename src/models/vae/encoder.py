import torch
from torch import nn
from torch.nn import functional as F
from src.models.vae.decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, H, W) -> (Batch_size, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (Batch_size, 128, H, W) -> (Batch_size, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, H, W) -> (Batch_size, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, H, W) -> (Batch_size, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (Batch_size, 128, H/2, W/2) -> (Batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            # (Batch_size, 256, H/2, W/2) -> (Batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),
            # (Batch_size, 256, H/2, W/2) -> (Batch_size, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (Batch_size, 256, H/4, W/4) -> (Batch_size, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            # (Batch_size, 512, H/4, W/4) -> (Batch_size, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, H/4, W/4) -> (Batch_size, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            VAE_AttentionBlock(512),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            nn.GroupNorm(32, 512),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            nn.SiLU(),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (Batch_size, 8, H/8, W/8) -> (Batch_size, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch_size, Channel, H, W)
        noise: (Batch_size, Out_Channels, H/8, W/8)
        """

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # (Pad_left, Pad_right, Pad_top, Pad_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (Batch_size, 8, H/8, W/8) -> two tensors of shape (Batch_size, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (Batch_size, 4, H/8, W/8) -> (Batch_size, 4, H/8, W/8)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # z = N(0, 1) -> x = N(mean, variance)
        # x = mean + stdev * z
        x = mean + stdev * noise

        x *= 0.18215
        return x
