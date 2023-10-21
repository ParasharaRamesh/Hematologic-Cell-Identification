import torch
from torchinfo import summary
import torch.nn as nn
import config.params as config
import torchvision.models as models

class WBCClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = self.conv_and_batch_norm_block(3, 64, pool=True)
        self.conv2 = self.conv_and_batch_norm_block(64, 64, pool=True)
        self.res1 = self.conv_and_batch_norm_block(64, 64)
        self.res2 = self.conv_and_batch_norm_block(64, 64)

        self.conv3 = self.conv_and_batch_norm_block(64, 32, pool=True)
        self.conv4 = self.conv_and_batch_norm_block(32, 16, pool=True)
        self.res3 = self.conv_and_batch_norm_block(16, 16)
        self.res4 = self.conv_and_batch_norm_block(16, 16)

        # #new approach with only resnet18
        # # Replace the initial layers with a pre-trained ResNet-18 backbone
        # self.resnet18 = models.resnet18(pretrained=True).to(config.device)
        #
        # # set to non trainable
        # self.resnet18.eval()
        # for param in self.resnet18.parameters():
        #     param.requires_grad = False
        #
        # # remove the fully connected layer
        # self.resnet18.fc = nn.Identity().to(config.device)  # Remove the fully connected layer (classifier)

        self.classifier = nn.Sequential(
            #Input shape is (b, 16,16,16)
            # Convolutional layers to reduce spatial dimensions
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # Shape: (b, 8, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (b, 8, 8, 8)

            nn.Flatten(),  # Shape: (b, 64 * 4 * 4)

            # Linear layers with ReLU activation
            nn.Linear(8 * 8 * 8, 64),  # Shape: (b, 64)
            nn.ReLU(),  # Shape: (b, 64)

            nn.Linear(64, 8),  # Shape: (b, 32)
            nn.ReLU(),  # Shape: (b, 32)

            nn.Linear(8, self.num_classes)  # Final linear layer with output size 5 (for 5 classes)
        ).to(config.device)

    def conv_and_batch_norm_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers).to(config.device)

    # def forward_with_resenet18(self, x):
    #     # Pass input through the ResNet-18 backbone
    #     features = self.resnet18(x)
    #
    #     # Continue with the linear stack and predictor
    #     classified = self.classifier(features)
    #     return classified

    def forward(self, x):
        out = self.conv1(x)  # shape (b,64,128,128)
        out1 = self.conv2(out)  # shape (b,64,64,64)
        out = self.res1(out1) + out1  # skip connections, shape (b,64,64,64)
        out = out1 + self.res2(out) + out  # multi skip connections, shape (b,64,64,64)

        out = self.conv3(out)  # shape is (b,32,32,32)
        out2 = self.conv4(out)  # shape is (b,16,16,16)
        out = self.res3(out2) + out2  # skip connections, shape is (b,16,16,16)
        out = out2 + self.res4(out) + out  # multi skip connections, shape is (b,16,16,16)

        out = self.classifier(out)
        return out


if __name__ == '__main__':
    wbc = WBCClassifier().to(config.device)
    summary(wbc, input_size=(3, 256, 256), device=config.device, batch_dim=0,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
