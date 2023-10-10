from config.params import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
from torchvision.datasets import ImageFolder


class pRCCDataset:
    def __init__(self, path, batch_size=pRCC_batch_size, resize_to=pRCC_img_resize_target, test_split=test_split, validation_split=validation_split):
        # constants
        self.path = path
        self.test_split = test_split
        self.validation_split = validation_split
        self.resize_to = resize_to
        self.batch_size = batch_size

        # transformations
        self.transforms = [
            # basic transformation
            transforms.Compose([
                transforms.Resize((resize_to, resize_to)),  # Resize images to a fixed size
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ]),
            # transformation with flips
            transforms.Compose([
                transforms.Resize((resize_to, resize_to)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ]),
            # transformation with rotation
            transforms.Compose([
                transforms.Resize((resize_to, resize_to)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ]),
            # transformation with rotation & flips
            transforms.Compose([
                transforms.Resize((resize_to, resize_to)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ])
        ]

        # create dataset
        self.dataset = self.construct_dataset_with_augmentation()

    def get_dataset(self, transformation):
        '''
        Given a folder with sub folders containing images, get all the images along with applying transformations

        :param inp_path:
        :param transformations:
        :return:
        '''
        return ImageFolder(root=self.path, transform=transformation)

    def construct_dataset_with_augmentation(self):
        '''

        :return: Concatenated dataset with all augmentations
        '''
        augmentations = []
        for transformation in self.transforms:
            augmentations.append(self.get_dataset(transformation))
        return ConcatDataset(augmentations)

    def get_dataloaders(self):
        '''

        :return: the Train, Val and test dataloaders
        '''
        # Calculate the number of samples to use for validation
        num_total_samples = len(self.dataset)

        # find the no of train samples
        num_test_samples = int(num_total_samples * test_split)
        num_train_samples = num_total_samples - num_test_samples

        num_validation_samples = int(num_test_samples * validation_split)
        num_test_samples = num_test_samples - num_validation_samples

        # Split the full dataset into train and test sets
        train_dataset, test_dataset, validation_dataset = random_split(self.dataset, [num_train_samples, num_test_samples, num_validation_samples])

        # Create DataLoaders for validation and test sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, validation_loader


if __name__ == '__main__':
    path = os.path.abspath("../datasets/pRCC/")
    pRCC = pRCCDataset(path)
    pRCC.get_dataloaders()
