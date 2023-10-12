import config.params as config
from torch.utils.data import ConcatDataset, random_split
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data.move.device_data_loader import DeviceDataLoader


# This dataset is balanced so no need to do any fancy things
class CamelyonDataset:
    def __init__(self, path, batch_size=config.cam_batch_size, test_split=config.test_split,
                 validation_split=config.validation_split):
        # constants
        self.path = path
        self.test_split = test_split
        self.validation_split = validation_split
        self.batch_size = batch_size

        # paths
        self.train_data_path = "train/data"
        self.train_mask_path = "train/mask"
        self.test_path = "test"
        self.val_path = "val"

        # transformations
        self.transforms = [
            # basic transformation
            transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(*config.stats),
            ])
            # ,
            # # transformation with flips
            # transforms.Compose([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomVerticalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(*config.stats)
            # ]),
            # # transformation with rotation
            # transforms.Compose([
            #     transforms.RandomRotation(degrees=15),
            #     transforms.ToTensor(),
            #     transforms.Normalize(*config.stats)
            # ]),
            # # transformation with rotation & flips
            # transforms.Compose([
            #     transforms.RandomRotation(degrees=15),
            #     transforms.RandomVerticalFlip(),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(*config.stats)
            # ])
        ]

        # create dataset
        # self.dataset = self.construct_dataset_with_augmentation()

    def get_datasets(self, train_transformation, test_transformation, val_transformation):
        return self.get_train_dataset(train_transformation), self.get_val_dataset(
            val_transformation), self.get_test_dataset(test_transformation)

    def get_train_dataset(self, transformation):
        '''
        Given a folder with sub folders containing images, get all the images along with applying transformations

        :param inp_path:
        :param transformations:
        :return:
        '''
        data_folder = ImageFolder(root=f"{self.path}/{self.train_data_path}", transform=transforms.ToTensor())
        mask_folder = ImageFolder(root=f"{self.path}/{self.train_mask_path}", transform=transforms.ToTensor())

        # Uncomment for local testing
        # image_folder = create_mini_dataset(image_folder, 5)

        '''
        NOTES:
        mask_folder.class_to_idx: normal->0, tumor->1
        mask_folder.imgs -> [(path,class_idx),..]
        
        can apply segmentation mask on top by finding the corresponding index of the data_folder ( just index in os.listdir())... apply it.
        
        '''

        #TODO.x need to find out a way to apply a segmentation mask on the data folder! and add it to the training data
        return data_folder, mask_folder

    def get_test_dataset(self, transformation):
        '''
        Given a folder with sub folders containing images, get all the images along with applying transformations

        :param inp_path:
        :param transformations:
        :return:
        '''
        # TODO.x getting only tensor
        image_folder = ImageFolder(root=f"{self.path}/{self.test_path}", transform=transforms.ToTensor())

        # Uncomment for local testing
        # image_folder = create_mini_dataset(image_folder, 5)

        return image_folder

    def get_val_dataset(self, transformation):
        '''
        Given a folder with sub folders containing images, get all the images along with applying transformations

        :param inp_path:
        :param transformations:
        :return:
        '''
        # TODO.x getting only tensor
        image_folder = ImageFolder(root=f"{self.path}/{self.val_path}", transform=transforms.ToTensor())

        # Uncomment for local testing
        # image_folder = create_mini_dataset(image_folder, 5)

        return image_folder

    def construct_dataset_with_augmentation(self):
        '''

        :return: Concatenated dataset with all augmentations
        '''
        augmentations = []
        for transformation in self.transforms:
            augmentations.append(self.get_datasets(transformation))
        return ConcatDataset(augmentations)

    def get_dataloaders(self):
        '''

        :return: the Train, Val and test dataloaders
        '''
        # Calculate the number of samples to use for validation
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

        # Create DataLoaders for validation and test sets
        return DeviceDataLoader(train_dataset, self.batch_size), DeviceDataLoader(test_dataset,
                                                                                  self.batch_size), DeviceDataLoader(
            validation_dataset, self.batch_size)


if __name__ == '__main__':
    path = os.path.abspath("../../datasets/CAM16_100cls_10mask")
    cam = CamelyonDataset(path)
    cam.get_train_dataset(None)
