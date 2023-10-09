from torchvision import transforms

#1. FLAGS
run_in_local = True

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2. CONSTANTS

test_split = 0.2
validation_split = 0.4

pRCC_batch_size = 8


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3. TRANSFORMS

#constants
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#basic transforms
transforms_basic = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

#pRCC dataset transforms
transforms_pRCC_flips = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

transforms_pRCC_rotations = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

transforms_pRCC_flips_and_rotations = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
