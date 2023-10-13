import torch
from torchinfo import summary
import torch.nn as nn
import config.params as config

class pRCCUnetAutoencoder(nn.Module):
    def __init__(self, latent_dim_size=config.pRCC_latent_dim):
        super().__init__()

        # constants
        self.latent_dim_size = latent_dim_size

        # Encoding path.
        self.encode_conv_1 = self.encoding_step(3, 32)
        self.encode_conv_2 = self.encoding_step(32, 64)
        self.encode_conv_3 = self.encoding_step(64, 128)
        self.encode_conv_4 = self.encoding_step(128, 256)
        self.encode_conv_5 = self.encoding_step(256, 256)  # Reduced to 256 channels

        # Linear layers for latent encoding
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)  # Reduce dimensions

    def forward(self, x):
        # Encoding
        encode_5 = self.encode_conv_1(x)  # Shape: (batchsize, 32, 512, 512)
        encode_4 = self.encode_conv_2(encode_5)  # Shape: (batchsize, 64, 256, 256)
        encode_3 = self.encode_conv_3(encode_4)  # Shape: (batchsize, 128, 128, 128)
        encode_2 = self.encode_conv_4(encode_3)  # Shape: (batchsize, 256, 64, 64)
        encode_1 = self.encode_conv_5(encode_2)  # Shape: (batchsize, 256, 32, 32)

        # Decoding
        decode_1 = self.decoding_step(encode_1, 256,
                                      256)  # Output is now in decode_1 with shape (batchsize, 256, 64, 64)
        decode_2 = self.decoding_step(decode_1 + encode_2, 256, 128)  # Shape: (batchsize, 128, 128, 128)
        decode_3 = self.decoding_step(decode_2 + encode_3, 128, 64)  # Shape: (batchsize, 64, 256, 256)
        decode_4 = self.decoding_step(decode_3 + encode_4, 64, 32)  # Shape: (batchsize, 32, 512, 512)
        decode_5 = self.decoding_step(decode_4 + encode_5, 32, 3)  # Shape: (batchsize, 3, 1024, 1024)

        # Latent Encoding
        latent_encoding = self.linear_latent(encode_1)  # Shape: (batchsize, 2048)

        return latent_encoding, decode_5

    def encoding_step(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # BatchNorm added
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # BatchNorm added
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoding_step(self, x, in_channels, out_channels, ct_padding=0, conv_padding=1):
        # need to explicitly move the convTranspose to correct device!
        x = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2,
                               padding=ct_padding).to(config.device)(x)  # Upsample
        x = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=conv_padding) \
            .to(config.device)(x)  # Convolution
        return x

    def linear_latent(self, x):
        # Apply pooling to reduce dimensions
        x = nn.MaxPool2d(kernel_size=2).to(config.device)(x)  # Shape: (batchsize, 256, 16, 16)
        x = nn.MaxPool2d(kernel_size=2).to(config.device)(x)  # Shape: (batchsize, 256, 8, 8)
        x = nn.AdaptiveMaxPool2d((4, 4)).to(config.device)(x)  # Shape: (batchsize, 256, 4, 4)

        # Flatten the tensor
        x = nn.Flatten().to(config.device)(x)  # Shape: (batchsize, 256 * 4 * 4)

        # Apply linear layers to further reduce dimensions
        x = nn.Linear(256 * 4 * 4, 4096).to(config.device)(x)  # Shape: (batchsize, 4096)
        x = nn.Linear(4096, self.latent_dim_size).to(config.device)(x)  # Shape: (batchsize, 2048)
        return x

# if __name__ == '__main__':
#     # Example usage:
#     autoencoder = pRCCUnetAutoencoder()
#     autoencoder.to("cpu")
#     # summary(autoencoder, input_size=(3, 512, 512), device="cpu", batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
#     input_tensor = torch.randn(1, 3, 512, 512)
#     latent, output = autoencoder(input_tensor)
