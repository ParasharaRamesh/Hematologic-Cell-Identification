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
class WBCDataset:
    def __init__(self,
                 train_path,
                 eval_path,
                 eval_size=200,
                 val_split=config.validation_split,
                 batch_size=config.wbc_batch_size,
                 resize_to=config.wbc_img_resize_target):
        # constants
        self.train_path = train_path
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.resize_to = resize_to
        self.val_split = val_split
        self.eval_size = eval_size

        # paths
        self.train_path = os.path.join(self.train_path, "train", "data")
        self.eval_path = os.path.join(self.eval_path, "val", "data")

        # transformations
        self.transforms = transforms.Compose([
            transforms.Resize((self.resize_to, self.resize_to)),
            transforms.ToTensor()
        ])

        # create dataset
        self.train_dataset = self.get_train_dataset()
        self.test_dataset, self.validation_dataset = self.get_test_val_datasets()
        print("Datasets are initialized")

    def get_train_dataset(self):
        image_folder = ImageFolder(root=self.train_path, transform=self.transforms)

        # Uncomment for local testing
        # image_folder = LocalDebug.create_mini_dataset(image_folder, 5)

        return image_folder

    def get_test_val_datasets(self, take_subset=True):
        image_folder = ImageFolder(root=self.eval_path, transform=self.transforms)
        print("constructing test and val dataset with augmentation")

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

        return test_dataset, validation_dataset

    def get_dataloaders(self):
        '''

        :return: the Train, Val and test dataloaders
        '''

        # Create DataLoaders for validation and test sets
        return DeviceDataLoader(self.train_dataset, self.batch_size), \
            DeviceDataLoader(self.test_dataset, self.batch_size), \
            DeviceDataLoader(self.validation_dataset, self.batch_size)


if __name__ == '__main__':
    train_path = os.path.abspath("../../datasets/modified/WBC_1")
    eval_path = os.path.abspath("../../datasets/modified/WBC_test")
    # out_path = os.path.abspath("../../datasets/modified/WBC_1")
    wbc = WBCDataset(train_path, eval_path)
    a, b, c = wbc.get_dataloaders()
    print("Done")
