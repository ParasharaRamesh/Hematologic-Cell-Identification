from torchvision import transforms

# 1. FLAGS
run_in_local = True

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. CONSTANTS

test_split = 0.2
validation_split = 0.4

learning_rate = 0.01
weight_decay = 1e-4
grad_clip = 0.1

pRCC_batch_size = 8
pRCC_img_resize_target = 1024  # from 2000 -> 1024

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRANSFORMS

# constants
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# basic transforms




# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
