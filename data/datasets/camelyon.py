import config.params as config
from torch.utils.data import ConcatDataset
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data.move.device_data_loader import DeviceDataLoader

'''
This dataset is balanced so no need to do any fancy things
Here all there are only 2 classes normal:0 and tumor:1, we need to one hot encode it as well

'''
class CamelyonDataset:
    def __init__(self, path, batch_size=config.cam_batch_size, resize_to=config.cam_img_resize_target):
        # constants
        self.path = path
        self.batch_size = batch_size
        self.resize_to = resize_to

        # paths
        self.train_path = os.path.join(self.path, "train", "data")
        self.test_path = os.path.join(self.path, "test", "data")
        self.val_path = os.path.join(self.path, "val", "data")

        # transformations
        self.eval_transforms = [
            transforms.Compose([
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.ToTensor()
            ])
        ]

        self.train_transforms = [
            # basic transformation
            *self.eval_transforms,
            # transformation with flips
            transforms.Compose([
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]),
            # transformation with rotation
            transforms.Compose([
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor()
            ]),
            # transformation with rotation & flips
            transforms.Compose([
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        ]

        # create dataset
        self.validation_dataset = self.get_val_dataset()
        self.test_dataset = self.get_test_dataset()
        self.train_dataset = self.get_train_dataset_with_augmentation()

    def get_train_dataset_with_augmentation(self):
        '''

        :return: Concatenated dataset with all augmentations
        '''
        augmentations = []
        for transformation in self.train_transforms:
            augmentations.append(self.get_train_dataset(transformation))
        return ConcatDataset(augmentations)

    def get_train_dataset(self, transformation):
        image_folder = ImageFolder(root=self.train_path, transform=transformation)

        # Uncomment for local testing
        # image_folder = create_mini_dataset(image_folder, 5)

        return image_folder

    def get_test_dataset(self):
        image_folder = ImageFolder(root=self.test_path, transform=self.eval_transforms)

        # Uncomment for local testing
        # image_folder = create_mini_dataset(image_folder, 5)

        return image_folder

    def get_val_dataset(self):
        image_folder = ImageFolder(root=self.val_path, transform=self.eval_transforms)

        # Uncomment for local testing
        # image_folder = create_mini_dataset(image_folder, 5)

        return image_folder

    def get_dataloaders(self):
        '''

        :return: the Train, Val and test dataloaders
        '''

        # Create DataLoaders for validation and test sets
        return DeviceDataLoader(self.train_dataset, self.batch_size), DeviceDataLoader(self.test_dataset,
                                                                                       self.batch_size), DeviceDataLoader(
            self.validation_dataset, self.batch_size)


if __name__ == '__main__':
    path = os.path.abspath("../../datasets/modified/Camelyon")
    cam = CamelyonDataset(path)
    cam.get_dataloaders()
