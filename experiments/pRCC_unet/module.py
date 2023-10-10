import os
import random
import torch
from pytorch_msssim import SSIM
import config
from scripts.module import Module
from utils.loss import ssim_loss
import matplotlib.pyplot as plt


class pRCCModule(Module):
    def __init__(self, name, dataset, model, loss_criterion, save_dir):
        super().__init__(name, dataset, model, loss_criterion, save_dir)

        # structured similarity index
        self.loss_criterion = SSIM()

        # Values which can change based on loaded checkpoint
        self.start_epoch = 0
        self.epoch_numbers = []
        self.training_losses = []
        self.validation_losses = []

    # hooks
    def init_params_from_checkpoint_hook(self, resume_epoch_num):
        checkpoint_path = self.get_model_checkpoint_path(resume_epoch_num)
        checkpoint = torch.load(checkpoint_path)

        # load previous state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

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
        print(f"Initialized scheduler")

    def calculate_loss_hook(self, data):
        images = data
        predictions = self.model(images)
        loss = ssim_loss(self.loss_criterion, images, predictions)
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
                val_images = val_data
                val_predictions = self.model(val_images)

                # Validation loss update
                val_loss += ssim_loss(self.loss_criterion, val_images, val_predictions).item()

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
                'scheduler_state_dict': self.scheduler.state_dict(),
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
                test_images = val_data
                test_predictions = self.model(test_images)

                # Validation loss update
                test_loss += ssim_loss(self.loss_criterion, test_images, test_predictions).item()

        # Calculate average validation loss for the epoch
        avg_test_loss = test_loss / len(self.test_loader)

        # show some sample predictions
        self.show_sample_reconstructions(self.test_loader)

        return {
            "avg_test_loss": avg_test_loss
        }

    # util code
    def show_sample_reconstructions(self, dataloader, num_samples=3):
        self.model.eval()

        # Get random samples
        sample_indices = random.sample(range(len(dataloader.dataset)), num_samples)

        # Create a subplot grid
        fig, axes = plt.subplots(num_samples, 2, figsize=(9, 9))

        with torch.no_grad():
            for i, idx in enumerate(sample_indices):
                # Get a random sample from the data loader
                sample_image = dataloader.dataset[idx]
                sample_image = sample_image.unsqueeze(0)

                # Forward pass through the model
                predicted_image = self.model(sample_image)

                # keep it ready for showcasing in matplotlib
                predicted_image = predicted_image.permute(1, 2, 0)
                sample_image = sample_image.squeeze()
                sample_image = sample_image.permute(1, 2, 0)

                axes[i, 0].imshow(sample_image)
                axes[i, 0].set_title(f"Original Image #{i + 1}", color='green')
                axes[i, 0].axis('off')
                axes[i, 0].set_title('off')

                axes[i, 1].imshow(predicted_image)
                axes[i, 1].set_title(f"Reconstructed Image #{i + 1}", color='red')
                axes[i, 1].axis('off')

        plt.tight_layout()
        plt.show()
