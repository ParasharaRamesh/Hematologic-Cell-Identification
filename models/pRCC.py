import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
        ])

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder = nn.ModuleList([
            self.upconv_block(1024, 512),
            self.upconv_block(512, 256),
            self.upconv_block(256, 128),
            self.upconv_block(128, 64),
        ])

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        skip_connections = []

        # Encode
        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(2)(x)

        # Bottleneck
        x = self.bottleneck(x)

        #TODO.x
        latent = x.copy()

        # Decode
        for i, block in enumerate(self.decoder):
            x = block(x)
            x = torch.cat([x, skip_connections[-(i + 1)]], dim=1)

        # Output
        reconstruction = self.outconv(x)

        return latent, reconstruction


if __name__ == '__main__':
    # Example usage:
    autoencoder = UNet(in_channels=3, out_channels=3)
    input_tensor = torch.randn(1, 3, 2000, 2000)
    latent, output = autoencoder(input_tensor)
    print(latent.shape)  # Should match the input shape (1, 3, 2000, 2000)
    print(output.shape)  # Should match the input shape (1, 3, 2000, 2000)
