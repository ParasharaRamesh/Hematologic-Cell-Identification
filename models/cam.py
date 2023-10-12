import torch
from torchinfo import summary
import torch.nn as nn
import config.params as config

'''
TODO.


'''


class CamelyonClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.num_classes = 2

        self.conv1 = self.conv_and_batch_norm_block(3, 64)
        self.conv2 = self.conv_and_batch_norm_block(64, 128, pool=True)
        self.res1 = self.conv_and_batch_norm_block(128, 128)
        self.res2 = self.conv_and_batch_norm_block(128, 128)

        self.conv3 = self.conv_and_batch_norm_block(128, 256, pool=True)
        self.conv4 = self.conv_and_batch_norm_block(256, 512, pool=True)

        self.res3 = self.conv_and_batch_norm_block(512, 512)
        self.res4 = self.conv_and_batch_norm_block(512, 512)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.num_classes)
        )

    def conv_and_batch_norm_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)  # shape (b,64,256,256)
        out1 = self.conv2(out)  # shape (b,128,128,128)
        out = self.res1(out1) + out1  # skip connections, shape (b,128,128,128)
        out = out1 + self.res2(out) + out  # multi skip connections, shape (b,128,128,128)
        out = self.conv3(out)  # shape is (b,256,64,64)
        out2 = self.conv4(out)  # shape is (b,512,32,32)
        out = self.res3(out2) + out2  # skip connections, shape is (b,512,32,32)
        out = out2 + self.res4(out) + out  # multi skip connections, shape is (b,512,32,32)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # Example usage:
    cam = CamelyonClassifier().to(config.device)
    summary(cam, input_size=(3, 256, 256), device=config.device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
    # input_tensor = torch.randn(1, 3, 256, 256).to(config.device)
    # output = cam(input_tensor)
