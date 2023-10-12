import torch
from torch.utils.data import DataLoader, Subset
import config.params as config

'''
Wrapper on top of dataloader to move tensors to device
'''
class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, device=config.device):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            yield self._move_to_device(batch)

    def _move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._move_to_device(item) for item in batch]
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        else:
            return batch


#Functions needed for local debugging
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


#TODO.1 write code given a dataset, find the corresponding image, apply mask and save it in a folder

if __name__ == '__main__':
    pass