# from config.params import *
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import os


if __name__ == '__main__':

    transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # Resize images to a fixed size
        # transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        # transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])
    # dataset = pRCCDataset(data_dir=)
    path = os.path.abspath("../datasets/pRCC/")
    dataset = ImageFolder(root=path, transform=transform)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        images,_ = batch
        break
