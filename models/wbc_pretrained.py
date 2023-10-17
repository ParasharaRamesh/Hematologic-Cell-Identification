import torch
import torch.nn as nn

from models.cam import CamelyonClassifier
from models.pRCC import pRCCUnetAutoencoder
from models.wbc import WBCClassifier


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

        if pRCC_weights_path:
            # Load the best weights for the pRCC model and make it non-trainable
            pRCC_model.load_state_dict(torch.load(pRCC_weights_path))
            pRCC_model.eval()
            for param in pRCC_model.parameters():
                param.requires_grad = False

        if cam_weights_path:
            # Load the best weights for the Cam16 model and make it non-trainable
            Cam16_model.load_state_dict(torch.load(cam_weights_path))
            Cam16_model.eval()
            for param in Cam16_model.parameters():
                param.requires_grad = False

        if wbc_weights_path:
            # Load the best weights for the WBC model (which remains trainable)
            WBC_model.load_state_dict(torch.load(wbc_weights_path))

        self.pRCC_model = pRCC_model
        self.Cam16_model = Cam16_model
        self.WBC_model = WBC_model

        #TODO.x make this a linear stack and dont just have a 1D thing... have a (b,256,16,16)
        # Modify the pRCC model to add a linear layer (2048 -> 5)
        self.pRCC_latent_to_output = nn.Linear(2048, 5)

        # Additional linear layer to combine the outputs of all three models (5 + 5 + 5)
        self.combine_outputs = nn.Linear(5, 5)

    def forward(self, pRCC_input, Cam16_input, WBC_input):
        # Forward pass through pRCC model
        pRCC_latent, _ = self.pRCC_model(pRCC_input)
        pRCC_output = self.pRCC_latent_to_output(pRCC_latent)

        # Forward pass through Cam16 model ( the second one is the prediction across 2 classes)
        Cam16_output, _ = self.Cam16_model(Cam16_input)

        # Forward pass through WBC model
        WBC_output = self.WBC_model(WBC_input)

        # Combine the outputs from each of the model
        combined_output = pRCC_output + Cam16_output + WBC_output

        # Combine the outputs of all three models
        combined_predictor = self.combine_outputs(combined_output)

        return combined_predictor


if __name__ == '__main__':
    # Create instances of pRCC, Cam16, and WBC models
    pRCC_model = pRCCUnetAutoencoder()
    Cam16_model = CamelyonClassifier()
    WBC_model = WBCClassifier()

    # Create the modified combined model
    combined_model = PretrainedWBCClassifier(pRCC_model, Cam16_model, WBC_model)
