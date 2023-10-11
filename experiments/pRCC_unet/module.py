import os
import random
import torch
from pytorch_msssim import SSIM
from torch.utils.data import Subset
import numpy as np
import config.params as config
from data.common import DeviceDataLoader
from experiments.base.module import Module
from utils.loss import SSIMLoss
import matplotlib.pyplot as plt
from torch import nn


class pRCCModule(Module):
    def __init__(self, name, dataset, model, save_dir):
        super().__init__(name, dataset, model, save_dir)

        # structured similarity index
        self.loss_criterion = SSIMLoss()

        #L2 Loss
        # self.loss_criterion = nn.MSELoss()

        # Values which can change based on loaded checkpoint
        self.start_epoch = 0
        self.epoch_numbers = []
        self.training_losses = []
        self.validation_losses = []

    # hooks
    def init_params_from_checkpoint_hook(self, load_from_checkpoint, resume_epoch_num):
        if load_from_checkpoint:
            # NOTE: resume_epoch_num can be None here if we want to load from the most recently saved checkpoint!
            checkpoint_path = self.get_model_checkpoint_path(resume_epoch_num)
            checkpoint = torch.load(checkpoint_path)

            # load previous state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Things we are keeping track of
            self.start_epoch = checkpoint['epoch']
            self.epoch_numbers = checkpoint['epoch_numbers']
            self.training_losses = checkpoint['training_losses']
            self.validation_losses = checkpoint['validation_losses']

            print(f"Model checkpoint for {self.name} is loaded from {checkpoint_path}!")

    def init_scheduler_hook(self, num_epochs):
        # optimizer is already defined in the super class constructor at this point
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader)
        )
        # print(f"Initialized scheduler")

    def calculate_loss_hook(self, data):
        images, _ = data
        latent_encoding, predictions = self.model(images)
        #trying a L2 loss
        loss = self.loss_criterion(images, predictions)

        # loss = ssim_loss(self.loss_criterion, images, predictions)
        return loss

    def calculate_train_batch_stats_hook(self):
        # Note: No accuracy to compute so leaving it as is.
        return dict()

    def calculate_avg_train_stats_hook(self, epoch_training_loss):
        # NOTE: no need to calculate avg training accuracy here
        avg_training_loss_for_epoch = epoch_training_loss / len(self.train_loader)
        return {
            "avg_training_loss": avg_training_loss_for_epoch
        }

    def validation_hook(self):
        '''
        :return: avg val loss for that epoch
        '''
        val_loss = 0.0

        # set to eval mode
        self.model.eval()

        with torch.no_grad():
            for val_batch_idx, val_data in enumerate(self.val_loader):
                val_images, _ = val_data
                _, val_predictions = self.model(val_images)

                # Validation loss update
                # val_loss += ssim_loss(self.loss_criterion, val_images, val_predictions).item()

                #L2 Loss
                val_loss += self.loss_criterion(val_images, val_predictions).item()

        # Calculate average validation loss for the epoch
        avg_val_loss_for_epoch = val_loss / len(self.val_loader)

        # show some sample predictions
        self.show_sample_reconstructions(self.val_loader)

        return {
            "avg_val_loss_for_epoch": avg_val_loss_for_epoch
        }

    def store_running_history_hook(self, epoch, avg_train_stats, avg_val_stats):
        self.epoch_numbers.append(epoch + 1)
        self.training_losses.append(avg_train_stats["avg_training_loss"])
        self.validation_losses.append(avg_val_stats["avg_val_loss_for_epoch"])

    def calculate_and_print_epoch_stats_hook(self, avg_train_stats, avg_val_stats):
        print(f"Epoch loss: {avg_train_stats['avg_training_loss']} | Val loss: {avg_val_stats['avg_val_loss_for_epoch']}")
        return {
            "epoch_loss": avg_train_stats["avg_training_loss"],
            "val_loss": avg_val_stats["avg_val_loss_for_epoch"]
        }

    def save_model_checkpoint_hook(self, epoch, avg_train_stats, avg_val_stats):
        # set it to train mode to save the weights (but doesn't matter apparently!)
        self.model.train()

        # create the directory if it doesn't exist
        model_save_directory = os.path.join(self.save_dir, self.name)
        os.makedirs(model_save_directory, exist_ok=True)

        # Checkpoint the model at the end of each epoch
        checkpoint_path = os.path.join(model_save_directory, f'model_epoch_{epoch + 1}.pt')
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch + 1,
                'epoch_numbers': self.epoch_numbers,
                'training_losses': self.training_losses,
                'validation_losses': self.validation_losses
            },
            checkpoint_path
        )
        print(f"Saved the model checkpoint for experiment {self.name} for epoch {epoch + 1}")

    def get_current_running_history_state_hook(self):
        return self.epoch_numbers, self.training_losses, self.validation_losses

    # test code
    def test_model(self):
        test_loss = 0.0

        # set to eval mode
        self.model.eval()

        with torch.no_grad():
            for val_batch_idx, val_data in enumerate(self.test_loader):
                test_images, _ = val_data
                _, test_predictions = self.model(test_images)

                # Validation loss update
                # test_loss += ssim_loss(self.loss_criterion, test_images, test_predictions).item()

                #trying L2 loss
                test_loss += self.loss_criterion(test_images, test_predictions).item()

        # Calculate average validation loss for the epoch
        avg_test_loss = test_loss / len(self.test_loader)

        # show some sample predictions
        self.show_sample_reconstructions(self.test_loader)

        return {
            "avg_test_loss": avg_test_loss
        }

    # util code
    def show_sample_reconstructions(self, dataloader, num_samples=1):
        self.model.eval()

        # Get random samples
        sample_indices = torch.randperm(len(dataloader.dataset))[:num_samples]
        subset_dataset = Subset(dataloader.dataset, sample_indices)

        dataloader = DeviceDataLoader(subset_dataset, self.batch_size)

        # Create a subplot grid
        fig, axes = plt.subplots(num_samples, 2, figsize=(9, 9))

        with torch.no_grad():
            for i, val_data in enumerate(dataloader):
                sample_image, _ = val_data

                # Forward pass through the model
                _, predicted_image = self.model(sample_image)

                # squeeze it
                sample_image = sample_image.squeeze().to("cpu")
                predicted_image = predicted_image.squeeze().to("cpu")

                # keep it ready for showcasing in matplotlib
                predicted_image = predicted_image.permute(1, 2, 0).numpy().astype(np.uint8)
                sample_image = sample_image.permute(1, 2, 0).numpy().astype(np.uint8)

                axes[0].imshow(sample_image)
                axes[0].set_title(f"Sample Original Image", color='green')
                axes[0].axis('off')

                axes[1].imshow(predicted_image)
                axes[1].set_title(f"Sample Reconstructed Image", color='red')
                axes[1].axis('off')

        plt.tight_layout()
        plt.show()


