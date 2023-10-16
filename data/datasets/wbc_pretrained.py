import torch

import config.params as config
from torch.utils.data import ConcatDataset, random_split, Subset, DataLoader, TensorDataset
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms

from data.debug.debug import LocalDebug
from data.move.device_data_loader import DeviceDataLoader
from collections import Counter
import random
from tqdm import tqdm


# This dataset is not balanced therefore we need to apply transformations appropriately
class PretrainedWBCDataset:
    def __init__(self,
                 train_path,
                 eval_path,
                 eval_size=200,
                 val_split=config.validation_split,
                 batch_size=config.wbc_batch_size,
                 wbc_resize_to=config.wbc_img_resize_target,
                 cam_resize_to=config.cam_img_resize_target,
                 pRCC_resize_to=config.pRCC_img_resize_target,
                 ):
        # constants
        self.train_path = train_path
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.eval_size = eval_size

        # image resize sizes
        self.wbc_resize_to = wbc_resize_to
        self.cam_resize_to = cam_resize_to
        self.pRCC_resize_to = pRCC_resize_to

        # paths
        self.train_path = os.path.join(self.train_path, "train", "data")
        self.eval_path = os.path.join(self.eval_path, "val", "data")

        # transformations
        self.wbc_resize_transform = transforms.Compose([
            transforms.Resize((self.wbc_resize_to, self.wbc_resize_to)),
            transforms.ToTensor()
        ])
        self.cam_resize_transform = self.resize_transformations(self.cam_resize_to)
        self.pRCC_resize_transform = self.resize_transformations(self.pRCC_resize_to)

        # create dataset
        self.test_dataset, self.validation_dataset = self.get_test_val_datasets()
        self.train_dataset = self.get_train_dataset()
        print("Datasets are initialized")

    def resize_transformations(self, resize_to):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor()
        ])

    def get_train_dataset(self):
        unbalanced_dataset = ImageFolder(root=self.train_path, transform=self.wbc_resize_transform)
        dataloader = DeviceDataLoader(unbalanced_dataset, 1)
        print("constructing train dataset with augmentation & balancing")
        return self.construct_dataset(dataloader)

    def get_test_val_datasets(self, take_subset=True):
        image_folder = ImageFolder(root=self.eval_path, transform=self.wbc_resize_transform)
        # Calculate the number of samples to use for validation

        num_total_samples = len(image_folder)

        # find the no of train samples
        num_validation_samples = int(num_total_samples * self.val_split)
        num_test_samples = num_total_samples - num_validation_samples

        test_dataset, validation_dataset = random_split(image_folder, [num_test_samples, num_validation_samples])

        if take_subset:
            # find the no of train samples
            num_val_eval_samples = int(self.eval_size * self.val_split)
            num_test_eval_samples = self.eval_size - num_val_eval_samples

            test_dataset = LocalDebug.create_mini_dataset(test_dataset, num_test_eval_samples)
            validation_dataset = LocalDebug.create_mini_dataset(validation_dataset, num_val_eval_samples)

        test_dataloader = DeviceDataLoader(test_dataset, 1)
        val_dataloader = DeviceDataLoader(validation_dataset, 1)

        print("constructing test & val dataset with augmentation")
        return self.construct_dataset(test_dataloader), self.construct_dataset(val_dataloader)

    def get_dataloaders(self):
        '''

        :return: the Train, Val and test dataloaders
        '''

        # Create DataLoaders for validation and test sets
        return DeviceDataLoader(self.train_dataset, self.batch_size), \
            DeviceDataLoader(self.test_dataset, self.batch_size), \
            DeviceDataLoader(self.validation_dataset, self.batch_size)

    def construct_dataset(self, dataloader):
        pRCC_image_tensors = []
        cam_image_tensors = []
        wbc_image_tensors = []
        image_target_tensors = []

        for data in tqdm(dataloader):
            wbc_img_tensor, target_tensor = data
            wbc_img_tensor = wbc_img_tensor.squeeze()

            # get the image first
            pRCC_img_tensor = self.pRCC_resize_transform(wbc_img_tensor)
            cam_img_tensor = self.cam_resize_transform(wbc_img_tensor)

            # add it
            wbc_image_tensors.append(wbc_img_tensor)
            pRCC_image_tensors.append(pRCC_img_tensor)
            cam_image_tensors.append(cam_img_tensor)
            image_target_tensors.append(target_tensor)

        return TensorDataset(
            torch.stack(pRCC_image_tensors),
            torch.stack(cam_image_tensors),
            torch.stack(wbc_image_tensors),
            torch.stack(image_target_tensors)
        )


if __name__ == '__main__':
    train_path = os.path.abspath("../../datasets/modified/WBC_1_balanced")
    eval_path = os.path.abspath("../../datasets/modified/WBC_test")
    wbc = PretrainedWBCDataset(train_path, eval_path)
    a, b, c = wbc.get_dataloaders()
    print("Done")
