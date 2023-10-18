import torch
from torchinfo import summary
import torch.nn as nn
import config.params as config
import torchvision.models as models

class WBCClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = self.conv_and_batch_norm_block(3, 64)
        self.conv2 = self.conv_and_batch_norm_block(64, 128, pool=True)
        self.res1 = self.conv_and_batch_norm_block(128, 128)
        self.res2 = self.conv_and_batch_norm_block(128, 128)

        self.conv3 = self.conv_and_batch_norm_block(128, 256, pool=True)
        self.conv4 = self.conv_and_batch_norm_block(256, 512, pool=True)

        self.res3 = self.conv_and_batch_norm_block(512, 512)
        self.res4 = self.conv_and_batch_norm_block(512, 512)

        #
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
            # Convolutional layers to reduce spatial dimensions
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # Shape: (b, 256, 32, 32)
            nn.ReLU(),  # Shape: (b, 256, 32, 32)
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (b, 256, 16, 16)

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Shape: (b, 128, 16, 16)
            nn.ReLU(),  # Shape: (b, 128, 16, 16)
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (b, 128, 8, 8)

            # Additional Convolutional layer to reduce channels
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Shape: (b, 64, 8, 8)
            nn.ReLU(),  # Shape: (b, 64, 8, 8)
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (b, 64, 4, 4)

            # Flatten the output to feed into linear layers
            nn.Flatten(),  # Shape: (b, 64 * 4 * 4)

            # Linear layers with ReLU activation
            nn.Linear(64 * 4 * 4, 64),  # Shape: (b, 64)
            nn.ReLU(),  # Shape: (b, 64)

            nn.Linear(64, 32),  # Shape: (b, 32)
            nn.ReLU(),  # Shape: (b, 32)

            nn.Linear(32, self.num_classes)  # Final linear layer with output size 5 (for 5 classes)
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
    cam = WBCClassifier().to(config.device)
    # summary(cam, input_size=(3, 256, 256), device=config.device, batch_dim=0,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
    input_tensor = torch.randn(1, 3, 256, 256).to(config.device)
    output = cam(input_tensor)
