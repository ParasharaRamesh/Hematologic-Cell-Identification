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
class classificationModule(Module):
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

    def calculate_loss_hook(self, data):
        images, labels = data
        labels = self.one_hot(labels)
        output_logits = self.model(images)
        loss = self.loss_criterion(output_logits, labels)

        # compute the batch stats right here and save it
        output_probs = nn.Softmax(dim=1)(output_logits)
        predicted = torch.argmax(output_probs, 1)
        batch_correct_predictions = (predicted == labels).sum().item()
        batch_size = labels.size(0)

        # store the accuracies
        self.batch_accuracy = batch_correct_predictions / batch_size
        self.train_correct_predictions += batch_correct_predictions

        return loss

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

    def validation_hook(self):
        '''
        :return: avg val loss for that epoch
        '''
        val_loss = 0.0
        val_correct_predictions = 0
        total_val_samples = 0

        # set to eval mode
        self.model.eval()

        with torch.no_grad():
            for val_batch_idx, val_data in enumerate(self.val_loader):
                val_images, val_labels = val_data
                val_labels = self.one_hot(val_labels)

                val_logits = self.model(val_images)

                val_loss += self.loss_criterion(val_logits, val_labels).item()

                # Compute validation accuracy for this batch
                val_probs = nn.Softmax(dim=1)(val_logits)
                val_predicted = torch.argmax(val_probs, dim=1)
                total_val_samples += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

        # Calculate average validation loss for the epoch
        avg_val_loss_for_epoch = val_loss / len(self.val_loader)

        # Calculate validation accuracy for the epoch
        avg_val_accuracy = val_correct_predictions / total_val_samples

        return {
            "avg_val_loss_for_epoch": avg_val_loss_for_epoch,
            "avg_val_accuracy": avg_val_accuracy
        }

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

    # test code
    def test_model(self):
        test_loss = 0.0
        test_correct_predictions = 0
        total_test_samples = 0

        # set to eval mode
        self.model.eval()

        with torch.no_grad():
            for test_batch_idx, test_data in enumerate(self.test_loader):
                test_images, test_labels = test_data
                test_labels = self.one_hot(test_labels)

                test_logits = self.model(test_images)

                test_loss += self.loss_criterion(test_logits, test_labels).item()

                # Compute validation accuracy for this batch
                test_probs = nn.Softmax(dim=1)(test_logits)
                test_predicted = torch.argmax(test_probs, dim=1)
                total_test_samples += test_labels.size(0)
                test_correct_predictions += (test_predicted == test_labels).sum().item()

        # Calculate average validation loss for the epoch
        avg_test_loss = test_loss / len(self.test_loader)

        # Calculate validation accuracy for the epoch
        avg_test_accuracy = test_correct_predictions / total_test_samples

        return {
            "test_loss": avg_test_loss,
            "test_accuracy": avg_test_accuracy
        }

    # util code
    def one_hot(self, labels):
        # Create an empty one-hot tensor
        one_hot_tensor = torch.zeros((labels.size(0), self.num_classes), dtype=torch.float32)

        # Use scatter to fill in the one-hot tensor
        one_hot_tensor.scatter_(1, labels.view(-1, 1), 1)

        return one_hot_tensor


if __name__ == '__main__':
    pass
