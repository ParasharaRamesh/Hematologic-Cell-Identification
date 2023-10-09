from config.params import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os

from data.common import get_dataset

#TODO.x need to fix this
def get_Camelyon_dataset(inp_path):
    '''
    Given inp path to pRCC dataset return the full dataset with transforms

    :param inp_path:
    :return:
    '''
    cam_dataset = get_dataset(inp_path, transforms_basic)
    cam_dataset_with_augmentation_1 = get_dataset(inp_path, transforms_pRCC_flips)
    cam_dataset_with_augmentation_2 = get_dataset(inp_path, transforms_pRCC_rotations)
    cam_dataset_with_augmentation_3 = get_dataset(inp_path, transforms_pRCC_flips_and_rotations)
    return ConcatDataset([cam_dataset, cam_dataset_with_augmentation_1, cam_dataset_with_augmentation_2, cam_dataset_with_augmentation_3])


def get_pRCC_dataloaders(inp_path):
    '''
    Given the inp path to the pRCC dataset return the train, test and val dataloaders

    :param inp_path:
    :return:
    '''
    pRCC_dataset = get_Camelyon_dataset(inp_path)

    # Calculate the number of samples to use for validation
    num_total_samples = len(pRCC_dataset)

    # find the no of train samples
    num_test_samples = int(num_total_samples * test_split)
    num_train_samples = num_total_samples - num_test_samples

    num_validation_samples = int(num_test_samples * validation_split)
    num_test_samples = num_test_samples - num_validation_samples

    # Split the full dataset into train and test sets
    train_dataset, test_dataset, validation_dataset = random_split(pRCC_dataset, [num_train_samples, num_test_samples, num_validation_samples])

    # Create DataLoaders for validation and test sets
    train_loader = DataLoader(train_dataset, batch_size=pRCC_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=pRCC_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=pRCC_batch_size, shuffle=True)

    return train_loader, test_loader, validation_loader


if __name__ == '__main__':
    path = os.path.abspath("../datasets/CAM16_100cls_10mask/test")
    get_pRCC_dataloaders(path)
