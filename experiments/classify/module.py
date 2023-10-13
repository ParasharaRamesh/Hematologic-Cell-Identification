import os
import torch
from torch import nn
from torch.utils.data import Subset
import numpy as np
import config.params as config
from data.move.device_data_loader import DeviceDataLoader
from experiments.base.module import Module
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''
Can reuse this for both Camelyon and WBC classification 
'''
class ClassificationModule(Module):
    def __init__(self, name, dataset, model, save_dir, num_classes):
        super().__init__(name, dataset, model, save_dir)

        # structured similarity index
        self.loss_criterion = nn.CrossEntropyLoss()

        # Values which can change based on loaded checkpoint
        self.start_epoch = 0
        self.epoch_numbers = []
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

        self.num_classes = num_classes
        self.train_correct_predictions = 0

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
            self.training_accuracies = checkpoint['training_accuracies']
            self.validation_accuracies = checkpoint['validation_accuracies']

            print(f"Model checkpoint for {self.name} is loaded from {checkpoint_path}!")

    def calculate_train_batch_stats_hook(self):
        # Note: No accuracy to compute so leaving it as is.
        batch_stats = {
            "batch_acc": self.batch_accuracy
        }
        self.batch_accuracy = 0
        return batch_stats

    def calculate_avg_train_stats_hook(self, epoch_training_loss):
        # NOTE: no need to calculate avg training accuracy here
        avg_training_loss_for_epoch = epoch_training_loss / len(self.train_loader)
        avg_training_accuracy = self.train_correct_predictions / len(self.train_loader)
        epoch_train_stats = {
            "avg_training_loss": avg_training_loss_for_epoch,
            "avg_training_accuracy": avg_training_accuracy
        }

        # reset
        self.train_correct_predictions = 0
        return epoch_train_stats

    def store_running_history_hook(self, epoch, avg_train_stats, avg_val_stats):
        self.epoch_numbers.append(epoch + 1)
        self.training_losses.append(avg_train_stats["avg_training_loss"])
        self.training_accuracies.append(avg_train_stats["avg_training_accuracy"])

        self.validation_losses.append(avg_val_stats["avg_val_loss_for_epoch"])
        self.validation_accuracies.append(avg_val_stats["avg_val_accuracy"])

    def calculate_and_print_epoch_stats_hook(self, avg_train_stats, avg_val_stats):
        print(
            f"Epoch loss: {avg_train_stats['avg_training_loss']} | Train Acc: {avg_train_stats['avg_training_accuracy']} | Val Acc: {avg_val_stats['avg_val_accuracy']} | Val loss: {avg_val_stats['avg_val_loss_for_epoch']}")

        return {
            "epoch_loss": avg_train_stats['avg_training_loss'],
            "val_loss": avg_val_stats['avg_val_loss_for_epoch'],
            "train_acc": avg_train_stats['avg_training_accuracy'],
            "val_acc": avg_val_stats['avg_val_accuracy']
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
                'training_accuracies': self.training_accuracies,
                'validation_losses': self.validation_losses,
                'validation_accuracies': self.validation_accuracies
            },
            checkpoint_path
        )
        print(f"Saved the model checkpoint for experiment {self.name} for epoch {epoch + 1}")

    def get_current_running_history_state_hook(self):
        return self.epoch_numbers, self.training_losses, self.training_accuracies, self.validation_losses, self.validation_accuracies

    # util code
    def one_hot(self, labels):
        # Create an empty one-hot tensor
        one_hot_tensor = torch.zeros((labels.size(0), self.num_classes), dtype=torch.float32)

        # Use scatter to fill in the one-hot tensor
        one_hot_tensor.scatter_(1, labels.view(-1, 1), 1)

        return one_hot_tensor


if __name__ == '__main__':
    pass
