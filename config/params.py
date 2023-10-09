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
pRCC_image_crop_pixels = 1024  # from 2000 -> 1024

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRANSFORMS

# constants
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# basic transforms
transforms_basic = transforms.Compose([
    transforms.Resize((pRCC_image_crop_pixels, pRCC_image_crop_pixels)),  # Resize images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# pRCC dataset transforms
transforms_pRCC_flips = transforms.Compose([
    transforms.Resize((pRCC_image_crop_pixels, pRCC_image_crop_pixels)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

transforms_pRCC_rotations = transforms.Compose([
    transforms.Resize((pRCC_image_crop_pixels, pRCC_image_crop_pixels)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

transforms_pRCC_flips_and_rotations = transforms.Compose([
    transforms.Resize((pRCC_image_crop_pixels, pRCC_image_crop_pixels)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
