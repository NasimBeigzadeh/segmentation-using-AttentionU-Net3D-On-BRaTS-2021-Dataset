
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3D convolution blocks.
    Each block uses GroupNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    3D Attention Gate for U-Net skip connections.

    x: Encoder feature map
    g: Decoder gating signal
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.GroupNorm(8, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.GroupNorm(8, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(
                F_int,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        attention = self.W_x(x) + self.W_g(g)
        attention = self.relu(attention)
        attention = self.psi(attention)

        return x * attention


class UNet3D(nn.Module):
    """
    Four-level 3D Attention U-Net.

    Input:
        [B, 4, D, H, W]

    Output:
        [B, 4, D, H, W]

    Classes:
        0 = Background
        1 = NCR/NET
        2 = ED
        3 = ET
    """

    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)

        self.pool = nn.MaxPool3d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up4 = nn.ConvTranspose3d(
            256, 128, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose3d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose3d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose3d(
            32, 16, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(32, 16)

        # Attention Gates
        self.att4 = AttentionGate(
            F_g=128, F_l=128, F_int=64
        )

        self.att3 = AttentionGate(
            F_g=64, F_l=64, F_int=32
        )

        self.att2 = AttentionGate(
            F_g=32, F_l=32, F_int=16
        )

        self.att1 = AttentionGate(
            F_g=16, F_l=16, F_int=8
        )

        # Output layer
        self.out = nn.Conv3d(
            16,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + Attention
        d4 = self.up4(b)
        a4 = self.att4(e4, d4)
        d4 = torch.cat([d4, a4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        a3 = self.att3(e3, d3)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        a2 = self.att2(e2, d2)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        a1 = self.att1(e1, d1)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)
