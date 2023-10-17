import torch
from torchinfo import summary
import torch.nn as nn
import config.params as config


# Initial approach with UNet had issues training it in a stable manner
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

    def forward(self, x):
        # Input is of dimension (batchsize, 3, 512,512)

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
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoding_step(self, x, in_channels, out_channels, ct_padding=0, conv_padding=1):
        # need to explicitly move the convTranspose to correct device!
        x = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2,
                               padding=ct_padding).to(config.device)(x)  # Upsample
        # x = nn.ReLU().to(config.device)(x)
        x = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=conv_padding).to(
            config.device)(x)  # Convolution
        # x =
        return x

    def linear_latent(self, x):
        # Apply convolution and max-pooling to reduce dimensions
        x = nn.Conv2d(256, 128, kernel_size=3, padding=1).to(config.device)(x)  # Shape: (batchsize, 128, 32, 32)
        x = nn.BatchNorm2d(128).to(config.device)(x)
        x = nn.MaxPool2d(kernel_size=2).to(config.device)(x)  # Shape: (batchsize, 128, 16, 16)

        x = nn.Conv2d(128, 64, kernel_size=3, padding=1).to(config.device)(x)  # Shape: (batchsize, 64, 16, 16)
        x = nn.BatchNorm2d(64).to(config.device)(x)
        x = nn.MaxPool2d(kernel_size=2).to(config.device)(x)  # Shape: (batchsize, 64, 8, 8)

        # Flatten the tensor
        x = nn.Flatten().to(config.device)(x)  # Shape: (batchsize, 32 * 4 * 4)

        # Apply linear layers to further reduce dimensions
        x = nn.Linear(1024, 1024).to(config.device)(x)  # Shape: (batchsize, 1024)
        x = nn.ReLU().to(config.device)(x)
        x = nn.Linear(1024, self.latent_dim_size).to(config.device)(x)  # Shape: (batchsize, 1024)
        x = nn.Sigmoid().to(config.device)(x)

        return x

#Regular Autoencoder
class pRCCAutoencoder(nn.Module):
    def __init__(self, latent_dim=config.pRCC_latent_dim):
        super().__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1).to(config.device)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1).to(config.device)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1).to(config.device)
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1).to(config.device)
        self.encoder_conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1).to(config.device)

        # Decoder
        self.decoder_conv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1).to(config.device)
        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1).to(config.device)
        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1).to(config.device)
        self.decoder_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1).to(config.device)
        self.decoder_conv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1).to(config.device)

        #Batch norms
        self.bn32 = nn.BatchNorm2d(32).to(config.device)
        self.bn64 = nn.BatchNorm2d(64).to(config.device)
        self.bn128 = nn.BatchNorm2d(128).to(config.device)
        self.bn256 = nn.BatchNorm2d(256).to(config.device)

        #Activation
        self.relu = nn.ReLU().to(config.device)

    def forward(self, x):
        # Encoding
        x1_enc = self.encoder_conv1(x)# Shape: (batch_size, 32, 256, 256)
        x1_enc = self.relu(self.bn32(x1_enc))

        x2_enc = self.encoder_conv2(x1_enc)  # Shape: (batch_size, 64, 128, 128)
        x2_enc = self.relu(self.bn64(x2_enc))

        x3_enc = self.encoder_conv3(x2_enc)  # Shape: (batch_size, 128, 64, 64)
        x3_enc = self.relu(self.bn128(x3_enc))

        x4_enc = self.encoder_conv4(x3_enc)  # Shape: (batch_size, 256, 32, 32)
        x4_enc = self.relu(self.bn256(x4_enc))

        x5_enc = self.encoder_conv5(x4_enc)  # Shape: (batch_size, 256, 16, 16)
        x5_enc = self.relu(self.bn256(x5_enc))

        # Decoding
        x5_dec = self.decoder_conv1(x5_enc)  # Shape: (batch_size, 256, 32, 32)
        x5_dec = self.relu(self.bn256(x5_dec))

        x4_dec = self.decoder_conv2(x5_dec)  # Shape: (batch_size, 128, 64, 64)
        x4_dec = self.relu(self.bn128(x4_dec))

        x3_dec = self.decoder_conv3(x4_dec)  # Shape: (batch_size, 64, 128, 128)
        x3_dec = self.relu(self.bn64(x3_dec))

        x2_dec = self.decoder_conv4(x3_dec)  # Shape: (batch_size, 32, 256, 256)
        x2_dec = self.relu(self.bn32(x2_dec))

        x1_dec = self.decoder_conv5(x2_dec)  # Shape: (batch_size, 3, 512, 512)

        return x5_enc, x1_dec




if __name__ == '__main__':
    # Example usage:
    # autoencoder = pRCCUnetAutoencoder().to(config.device)
    autoencoder = pRCCAutoencoder().to(config.device)
    # summary(autoencoder, input_size=(3, 512, 512), device=config.device, batch_dim=0,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
    input_tensor = torch.randn(1, 3, 512, 512).to(config.device)
    latent, output = autoencoder(input_tensor)
