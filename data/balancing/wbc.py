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

'''
TODO.x

1. move this to a new place for creating the wbc balancing ( call that module balancing)..
2. in wbc and wbc_pretrained dont have any balancing stuff... just load from path

'''


# This dataset is not balanced therefore we need to apply transformations appropriately
class WBCDatasetBalancer:
    def __init__(self,
                 train_path,
                 out_path,
                 batch_size=config.wbc_batch_size,
                 resize_to=config.wbc_img_resize_target):
        # constants
        self.train_path = train_path
        self.out_path = out_path
        self.batch_size = batch_size
        self.resize_to = resize_to

        # paths
        self.train_path = os.path.join(self.train_path, "train", "data")
        self.out_path = os.path.join(self.out_path, "train", "data")

        # Save the image to the output directory
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # transformations
        self.eval_transforms = transforms.Compose([
            transforms.Resize((self.resize_to, self.resize_to)),
            transforms.ToTensor()
        ])

        self.train_transforms = [
            # normal
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.ToTensor()
            ]),
            # horizontal flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            # vertical flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]),
            # transformation with flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]),
            # transformation with rotation
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor()
            ]),
            # transformation with rotation & flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize_to, self.resize_to)),
                transforms.RandomRotation(degrees=10),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        ]

        # self.num_transforms_for_target will contain the number of times each target needs to be transformed
        self.find_num_of_transforms_needed_for_balancing()
        self.tensor_to_img_transform = transforms.ToPILImage()

        # create dataset
        self.balance_dataset()
        print("Datasets are initialized")

    def balance_dataset(self):
        unbalanced_dataset = ImageFolder(root=self.train_path, transform=transforms.ToTensor())
        dataloader = DeviceDataLoader(unbalanced_dataset, 1)

        print("creating directories for each target class")
        for target_class in unbalanced_dataset.classes:
            target_class_path = os.path.join(self.out_path, target_class)
            os.makedirs(target_class_path, exist_ok=True)

        print("constructing train dataset with augmentation & balancing")
        for i, data in enumerate(tqdm(dataloader)):
            img_tensor, target_tensor = data
            img_tensor = img_tensor.squeeze()

            target_class_idx = target_tensor.item()
            target_class = unbalanced_dataset.classes[target_class_idx]
            target_class_path = os.path.join(self.out_path, target_class)

            # add the remaining transforms to the list
            num_of_transformations = self.num_transforms_for_target[target_class_idx]

            for j in range(num_of_transformations):
                random_transform_idx = random.choice(range(len(self.train_transforms)))
                random_transform = self.train_transforms[random_transform_idx]
                random_augmentation_img_tensor = random_transform(img_tensor)
                random_augmentation_img = self.tensor_to_img_transform(random_augmentation_img_tensor)

                file_path = os.path.join(target_class_path, f"{i}_transform_{j}_rti_{random_transform_idx}.jpg")
                random_augmentation_img.save(file_path)


    def find_num_of_transforms_needed_for_balancing(self):
        unbalanced_train_dataset = ImageFolder(root=self.train_path, transform=self.eval_transforms)

        target_counts = Counter(unbalanced_train_dataset.targets)
        target_with_most_count, max_count = target_counts.most_common(1)[0]

        self.num_transforms_for_target = {target: max_count // counts for target, counts in target_counts.items()}



if __name__ == '__main__':
    # train_path_1 = os.path.abspath("../../datasets/modified/WBC_1")
    # out_path_1 = os.path.abspath("../../datasets/modified/WBC_1_balanced")
    # wbc_1 = WBCDatasetBalancer(train_path_1, out_path_1)
    # print("WBC 1 done")

    # train_path_10 = os.path.abspath("../../datasets/modified/WBC_10")
    # out_path_10 = os.path.abspath("../../datasets/modified/WBC_10_balanced")
    # wbc_10 = WBCDatasetBalancer(train_path_10, out_path_10)
    # print("WBC 10 Done")

    train_path_50 = os.path.abspath("../../datasets/modified/WBC_50")
    out_path_50 = os.path.abspath("../../datasets/modified/WBC_50_balanced")
    wbc_50 = WBCDatasetBalancer(train_path_50, out_path_50)
    print("WBC 50 Done")

    train_path_100 = os.path.abspath("../../datasets/modified/WBC_100")
    out_path_100 = os.path.abspath("../../datasets/modified/WBC_100_balanced")
    wbc_100 = WBCDatasetBalancer(train_path_100, out_path_100)
    print("WBC 100 Done")
