import sys

import torch
from torchinfo import summary
import torch.nn as nn
import config.params as config
import torchvision.models as models

class CamelyonClassifier(nn.Module):
    def __init__(self, num_classes=2):
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

        self.linear_stack = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(32768, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 5),
            nn.ReLU()
        )

        # Replace the initial layers with a pre-trained ResNet-18 backbone
        self.resnet18 = models.resnet18(pretrained=True)

        #set to non trainable
        self.resnet18.eval()
        for param in self.resnet18.parameters():
            param.requires_grad = False

        #remove the fully connected layer
        self.resnet18.fc = nn.Identity()  # Remove the fully connected layer (classifier)

        self.resnet_linear_stack = nn.Sequential(
            nn.Linear(512, 128),  # Adjust input size based on the output of ResNet-18
            nn.ReLU(),
            nn.Linear(128, 64),  # Adjust input size based on the output of ResNet-18
            nn.ReLU(),
            nn.Linear(64, 32),  # Adjust input size based on the output of ResNet-18
            nn.ReLU(),
            nn.Linear(32, 32),  # Adjust input size based on the output of ResNet-18
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )

        self.predictor = nn.Sequential(
            nn.Linear(5, self.num_classes)
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

    def forward(self,x):
        # Pass input through the ResNet-18 backbone
        features = self.resnet18(x)

        # Continue with the linear stack and predictor
        linear_stack = self.resnet_linear_stack(features)
        predictor = self.predictor(linear_stack)
        return linear_stack, predictor

    def old_forward(self, x):
        '''
        This contains the old model architecture

        :param x:
        :return:
        '''
        #Assumes that x is of shape (b, 3, 256,256)
        out = self.conv1(x)  # shape (b,64,256,256)
        out1 = self.conv2(out)  # shape (b,128,128,128)
        out = self.res1(out1) + out1  # skip connections, shape (b,128,128,128)

        out = out1 + self.res2(out) + out  # multi skip connections, shape (b,128,128,128)
        out = self.conv3(out)  # shape is (b,256,64,64)
        out2 = self.conv4(out)  # shape is (b,512,32,32)

        out = self.res3(out2) + out2  # skip connections, shape is (b,512,32,32)
        out = out2 + self.res4(out) + out  # multi skip connections, shape is (b,512,32,32)

        linear_stack = self.linear_stack(out)
        predictor = self.predictor(linear_stack)
        return linear_stack, predictor

if __name__ == '__main__':
    cam = CamelyonClassifier().to(config.device)
    summary(cam, input_size=(3, 224, 224), device=config.device, batch_dim=0,col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
