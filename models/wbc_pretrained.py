import torch
import torch.nn as nn
from torchinfo import summary

from models.cam import CamelyonClassifier
from models.pRCC import pRCCUnetAutoencoder
from models.wbc import WBCClassifier
from config import params as config


class PretrainedWBCClassifier(nn.Module):
    def __init__(self,
                 pRCC_model,
                 Cam16_model,
                 WBC_model,
                 pRCC_weights_path=None,
                 cam_weights_path=None,
                 wbc_weights_path=None
                 ):
        super().__init__()

        self.pRCC_model = pRCC_model.to(config.device)
        self.Cam16_model = Cam16_model.to(config.device)
        self.WBC_model = WBC_model.to(config.device)

        if pRCC_weights_path:
            print(f"Loading the pRCC weights from {pRCC_weights_path}")
            pRCC_checkpoint = torch.load(pRCC_weights_path, map_location=config.device)
            print(f"Retrieved the pRCC Checkpoint file!")
            print(f"Keys present in this checkpoint are {pRCC_checkpoint.keys()}")
            # Load the best weights for the pRCC model and make it non-trainable
            self.pRCC_model.load_state_dict(pRCC_checkpoint['model_state_dict'])
            print(f"Finished loading the pRCC weights from {pRCC_weights_path}")
            self.pRCC_model.eval()
            for param in self.pRCC_model.parameters():
                param.requires_grad = False
            print(f"Setting the pRCC model to eval mode and non trainable!")

        if cam_weights_path:
            print(f"Loading the Camelyon weights from {cam_weights_path}")
            # Load the best weights for the Cam16 model and make it non-trainable
            cam_checkpoint = torch.load(cam_weights_path, map_location=config.device)
            print(f"Retrieved the Camelyon Checkpoint file!")
            print(f"Keys present in this checkpoint are {cam_checkpoint.keys()}")
            self.Cam16_model.load_state_dict(cam_checkpoint['model_state_dict'])
            print(f"Finished loading the Camelyon weights from {cam_weights_path}")
            self.Cam16_model.eval()
            for param in self.Cam16_model.parameters():
                param.requires_grad = False
            print(f"Setting the Camelyon model to eval mode and non trainable!")

        if wbc_weights_path:
            print(f"Loading the base WBC weights from {wbc_weights_path}")
            # Load the best weights for the WBC model (which remains trainable)
            wbc_checkpoint = torch.load(wbc_weights_path, map_location=config.device)
            print(f"Retrieved the WBC Checkpoint file!")
            print(f"Keys present in this checkpoint are {wbc_checkpoint.keys()}")
            self.WBC_model.load_state_dict(wbc_checkpoint)
            print(f"Finished loading the base WBC weights from {wbc_weights_path}")

        # Input (batch_size, 128, 64, 64)
        self.pRCC_latent_to_output = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 256, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output shape: (batch_size, 256, 32, 32)
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 128, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output shape: (batch_size, 128, 16, 16)
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 64, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output shape: (batch_size, 64, 8, 8)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 32, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output shape: (batch_size, 32, 4, 4)
            nn.Flatten(),  # Output shape: (batch_size, 32 * 4 * 4)
            nn.Linear(32 * 4 * 4, 5)  # Output shape: (batch_size, 5)
        ).to(config.device)

        # Additional linear layer to combine the outputs of all three models (5 + 5 + 5)
        self.combine_outputs = nn.Linear(5, 5).to(config.device)

    def forward(self, pRCC_input, Cam16_input, WBC_input):
        # Forward pass through pRCC model
        pRCC_latent, _ = self.pRCC_model(pRCC_input)  # b, 128,250,250
        pRCC_output = self.pRCC_latent_to_output(pRCC_latent)  # b,5

        # Forward pass through Cam16 model ( the second one is the prediction across 2 classes)
        Cam16_output, _ = self.Cam16_model(Cam16_input)  # b,5

        # Forward pass through WBC model
        WBC_output = self.WBC_model(WBC_input)  # b,5

        # Combine the outputs from each of the model
        combined_output = pRCC_output + Cam16_output + WBC_output  # b,5

        # Combine the outputs of all three models
        combined_predictor = self.combine_outputs(combined_output)  # b,5

        return combined_predictor


if __name__ == '__main__':
    # Create instances of pRCC, Cam16, and WBC models
    pRCC_model = pRCCUnetAutoencoder()  # input is (3,512,512)
    Cam16_model = CamelyonClassifier()  # input is (3,224,224)
    WBC_model = WBCClassifier()  # input is (3,256,256)

    # Create the modified combined model
    combined_model = PretrainedWBCClassifier(pRCC_model, Cam16_model, WBC_model).to(config.device)
    print(combined_model)

    #NOTE: not able to use torchinfo.summary for this complex model therefore I am just printing the basic model architecture which arises from printing the object!
    # summary(combined_model, input_size=[(1, 3, 512, 512), (1, 3, 224, 224), (1, 3, 256, 256)], device=config.device,
    #         batch_dim=0,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
