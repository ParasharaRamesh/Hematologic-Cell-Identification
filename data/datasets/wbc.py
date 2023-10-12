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
        self.eval_transforms = transforms.Compose([
            transforms.Resize((self.resize_to, self.resize_to)),
            transforms.ToTensor(),
            transforms.Normalize(*config.wbc_stats),
        ])

        self.train_transforms = [
            # normal
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.ToTensor(),
                transforms.Normalize(*config.wbc_stats)
            ]),
            # horizontal flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*config.wbc_stats)
            ]),
            # vertical flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*config.wbc_stats)
            ]),
            # transformation with flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*config.wbc_stats)
            ]),
            # transformation with rotation
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(*config.wbc_stats)
            ]),
            # transformation with rotation & flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomRotation(degrees=10),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*config.wbc_stats)
            ])
        ]

        # self.num_transforms_for_target will contain the number of times each target needs to be transformed
        self.find_num_of_transforms_needed_for_balancing()

        # create dataset
        self.test_dataset, self.validation_dataset = self.get_test_val_datasets()
        self.train_dataset = self.get_train_dataset()

    def get_train_dataset(self):
        unbalanced_dataset = ImageFolder(root=self.train_path, transform=transforms.ToTensor())
        dataloader = DeviceDataLoader(unbalanced_dataset, 1)

        augmented_image_tensors = []
        augmented_image_target_tensors = []

        for data in tqdm(dataloader):
            img_tensor, target_tensor = data
            img_tensor = img_tensor.squeeze()

            # add the remainining transforms to the list
            num_of_transformations = self.num_transforms_for_target[target_tensor.item()]
            for _ in range(num_of_transformations):
                random_transform = random.choice(self.train_transforms)
                random_augmentation_img_tensor = random_transform(img_tensor)

                augmented_image_tensors.append(random_augmentation_img_tensor)
                augmented_image_target_tensors.append(target_tensor)

        return TensorDataset(torch.stack(augmented_image_tensors), torch.stack(augmented_image_target_tensors))

    def get_test_val_datasets(self, take_subset=True):
        image_folder = ImageFolder(root=self.eval_path, transform=self.eval_transforms)
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

    def find_num_of_transforms_needed_for_balancing(self):
        unbalanced_train_dataset = ImageFolder(root=self.train_path, transform=self.eval_transforms)

        target_counts = Counter(unbalanced_train_dataset.targets)
        target_with_most_count, max_count = target_counts.most_common(1)[0]

        self.num_transforms_for_target = {target: max_count // counts for target, counts in target_counts.items()}

    def get_dataloaders(self):
        '''

        :return: the Train, Val and test dataloaders
        '''

        # Create DataLoaders for validation and test sets
        return DeviceDataLoader(self.train_dataset, self.batch_size), \
            DeviceDataLoader(self.test_dataset, self.batch_size), \
            DeviceDataLoader(self.validation_dataset, self.batch_size)


if __name__ == '__main__':
    train_path = os.path.abspath("../../datasets/modified/WBC_10")
    eval_path = os.path.abspath("../../datasets/modified/WBC_test")
    wbc = WBCDataset(train_path, eval_path)
    a, b, c = wbc.get_dataloaders()
    print("Done")
