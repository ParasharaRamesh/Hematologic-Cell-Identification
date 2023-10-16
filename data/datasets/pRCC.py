import config.params as config
from torch.utils.data import ConcatDataset, random_split, DataLoader
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data.move.device_data_loader import DeviceDataLoader
import matplotlib.pyplot as plt
from PIL import Image

class pRCCDataset:
    def __init__(self, path, batch_size=config.pRCC_batch_size, resize_to=config.pRCC_img_resize_target,
                 test_split=config.test_split, validation_split=config.validation_split):
        # constants
        self.path = path
        self.test_split = test_split
        self.validation_split = validation_split
        self.resize_to = resize_to
        self.batch_size = batch_size
        self.resize_to = resize_to

        # transformations
        self.transforms = [
            # basic transformation
            transforms.Compose([
                transforms.Resize((self.resize_to, self.resize_to)),  # Resize images to a fixed size
                transforms.ToTensor()
            ]),
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

        self.tensor_to_img_transform = transforms.ToPILImage()

        # create dataset
        self.dataset = self.construct_dataset_with_augmentation()

    def get_dataset(self, transformation):
        '''
        Given a folder with sub folders containing images, get all the images along with applying transformations

        :param inp_path:
        :param transformations:
        :return:
        '''
        image_folder = ImageFolder(root=self.path, transform=transformation)

        # Uncomment for local testing
        # image_folder = LocalDebug.create_mini_dataset(image_folder, 5)

        return image_folder

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
        test_dataset, train_dataset, validation_dataset = self.get_split_datasets()

        # Create DataLoaders for validation and test sets
        return DeviceDataLoader(train_dataset, self.batch_size), DeviceDataLoader(test_dataset,
                                                                                  self.batch_size), DeviceDataLoader(
            validation_dataset, self.batch_size)

    def get_split_datasets(self):
        num_total_samples = len(self.dataset)
        # find the no of train samples
        num_test_samples = int(num_total_samples * self.test_split)
        num_train_samples = num_total_samples - num_test_samples
        num_validation_samples = int(num_test_samples * self.validation_split)
        num_test_samples = num_test_samples - num_validation_samples
        # Split the full dataset into train and test sets
        train_dataset, test_dataset, validation_dataset = random_split(self.dataset,
                                                                       [num_train_samples, num_test_samples,
                                                                        num_validation_samples])
        return test_dataset, train_dataset, validation_dataset

    def show_sample_images(self):
        '''
        Show one image from each of the dataloaders

        :return:
        '''
        # Calculate the number of samples to use for validation
        test_dataset, train_dataset, validation_dataset = self.get_split_datasets()

        # create simple dataloaders of batchsize 1
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_loader = DataLoader(train_dataset, batch_size=1)
        validation_loader = DataLoader(validation_dataset, batch_size=1)

        print(f"Showing one image from train_loader")
        self.show_img_from_loader(train_loader)

        print(f"Showing one image from test_loader")
        self.show_img_from_loader(test_loader)

        print(f"Showing one image from validation_loader")
        self.show_img_from_loader(validation_loader)

    def show_img_from_loader(self, loader):
        for data in loader:
            img_tensor, _ = data
            img_tensor = img_tensor.squeeze()
            img = self.tensor_to_img_transform(img_tensor)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis labels and ticks
            plt.show()
            break


if __name__ == '__main__':
    path = os.path.abspath("../../datasets/modified/pRCC/")
    pRCC = pRCCDataset(path)
    # pRCC.get_dataloaders()
    pRCC.show_sample_images()
