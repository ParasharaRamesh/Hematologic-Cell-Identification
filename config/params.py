import torch
from torchvision import transforms

# 1. FLAGS
run_in_local = True

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. CONSTANTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_split = 0.2
validation_split = 0.3

learning_rate = 0.01
weight_decay = 1e-4
grad_clip = 0.1

pRCC_batch_size = 16
pRCC_img_resize_target = 512  # from 2000 -> 512 ( Too big to fit on machine!)
pRCC_latent_dim = 2048

cam_batch_size = 16
# cam_img_resize_target = 256 # from 384 -> 256
cam_img_resize_target = 224 # as we are passing it into resnet

wbc_batch_size = 16
wbc_img_resize_target = 224 # from 575 -> 224
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRANSFORMS

# constants
#computed from the pRCC dataset itself
# pRCC_stats = ((0.6843, 0.5012, 0.6436), (0.1962, 0.2372, 0.1771))
# cam_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# wbc_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
