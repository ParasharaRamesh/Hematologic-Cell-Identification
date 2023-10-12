# Functions needed for local debugging
import torch
from torch.utils.data import Subset


class LocalDebug:
    @staticmethod
    def create_mini_dataset(dataset, num_samples):
        '''
        Function to be used locally for checking if the model runs or not!

        :param dataset:
        :param num_samples:
        :return:
        '''
        subset_indices = torch.randperm(len(dataset))[:num_samples]
        subset_dataset = Subset(dataset, subset_indices)
        return subset_dataset

    @staticmethod
    def calculate_mean_and_std_of_dataset(dataset):
        '''
        Useful when determining what transforms to set

        :param dataset:
        :return:
        '''
        # dataset =  ImageFolder(root=self.path, transform=transforms.ToTensor())

        # Initialize variables to accumulate mean and standard deviation
        mean = torch.zeros(3)
        std = torch.zeros(3)

        # Loop through the dataset to compute mean and standard deviation
        for img, _ in dataset:
            mean += img.mean(1).mean(1)
            std += img.view(3, -1).std(1)

        # Calculate the mean and standard deviation
        mean /= len(dataset)
        std /= len(dataset)

        return (tuple(mean.tolist()), tuple(std.tolist()))
