import os
import torch
from torch import nn
from torch.utils.data import Subset
import numpy as np
import config.params as config
from data.move.device_data_loader import DeviceDataLoader
from experiments.base.trainer import Trainer
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from experiments.classify.trainer import ClassificationTrainer

class WBCClassifierTrainer(ClassificationTrainer):
    def __init__(self, name, dataset, model, save_dir, num_classes=5):
        super().__init__(name, dataset, model, save_dir, num_classes)
        # Class labels
        self.class_labels = {'Basophil': 0, 'Eosinophil': 1, 'Lymphocyte': 2, 'Monocyte': 3, 'Neutrophil': 4}
        # Prettify class labels
        self.class_names = list(self.class_labels.keys())

    # hooks
    def calculate_loss_hook(self, data):
        images, labels = data
        one_hot_labels = self.one_hot(labels)
        output_logits = self.model(images)
        loss = self.loss_criterion(output_logits, one_hot_labels)

        # compute the batch stats right here and save it
        output_probs = nn.Softmax(dim=1)(output_logits)
        predicted = torch.argmax(output_probs, 1)
        batch_correct_predictions = (predicted == labels).sum().item()
        batch_size = labels.size(0)

        # store the accuracies
        self.batch_accuracy = batch_correct_predictions / batch_size
        self.train_correct_predictions += batch_correct_predictions
        self.train_total_batches += labels.size(0)

        return loss

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
                one_hot_val_labels = self.one_hot(val_labels)

                val_logits = self.model(val_images)

                val_loss += self.loss_criterion(val_logits, one_hot_val_labels).item()

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

    def test_model(self):
        test_loss = 0.0
        test_correct_predictions = 0
        total_test_samples = 0

        # Initialize lists to store true labels and predicted labels
        true_labels = []
        predicted_labels = []

        # set to eval mode
        self.model.eval()

        with torch.no_grad():
            for test_batch_idx, test_data in enumerate(self.test_loader):
                test_images, test_labels = test_data
                one_hot_test_labels = self.one_hot(test_labels)

                test_logits = self.model(test_images)

                test_loss += self.loss_criterion(test_logits, one_hot_test_labels).item()

                # Compute validation accuracy for this batch
                test_probs = nn.Softmax(dim=1)(test_logits)
                test_predicted = torch.argmax(test_probs, dim=1)
                total_test_samples += test_labels.size(0)
                test_correct_predictions += (test_predicted == test_labels).sum().item()

                # Append true and predicted labels for each batch
                true_labels.extend(test_labels.cpu().numpy())
                predicted_labels.extend(test_predicted.cpu().numpy())

        # Calculate average validation loss for the epoch
        avg_test_loss = test_loss / len(self.test_loader)

        # Calculate validation accuracy for the epoch
        avg_test_accuracy = test_correct_predictions / total_test_samples

        conf_matrix, prettified_f1_scores, prettified_precision_scores, prettified_recall_scores = self.get_metrics_for_class_predictions(
            predicted_labels, true_labels
        )

        return {
            "test_loss": avg_test_loss,
            "test_accuracy": avg_test_accuracy,
            "conf_matrix": conf_matrix,
            "f1_scores": prettified_f1_scores,
            "recall_scores": prettified_recall_scores,
            "precision_scores": prettified_precision_scores
        }
    def get_metrics_for_class_predictions(self, predicted_labels, true_labels):
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        # Calculate F1 score, recall, and precision for each class
        f1_scores = f1_score(true_labels, predicted_labels, average=None)
        recall_scores = recall_score(true_labels, predicted_labels, average=None)
        precision_scores = precision_score(true_labels, predicted_labels, average=None)
        # Create dictionaries with prettified class labels
        prettified_f1_scores = {self.class_names[i]: f1_scores[i] for i in range(len(self.class_names))}
        prettified_recall_scores = {self.class_names[i]: recall_scores[i] for i in range(len(self.class_names))}
        prettified_precision_scores = {self.class_names[i]: precision_scores[i] for i in range(len(self.class_names))}
        # Print the confusion matrix with class labels
        self.print_confusion_matrix(conf_matrix)
        return conf_matrix, prettified_f1_scores, prettified_precision_scores, prettified_recall_scores

    def print_confusion_matrix(self, conf_matrix):
        num_classes = len(self.class_names)

        print("Confusion Matrix:")

        # Print header row with class names
        header = ["Actual\Predicted"] + self.class_names
        print("\t".join(header))

        # Print TP and TN for each class
        for i in range(num_classes):
            row = [self.class_names[i]]  # Class label

            # Calculate TP and TN for the class
            TP = conf_matrix[i, i]
            TN = sum([conf_matrix[j, k] for j in range(num_classes) for k in range(num_classes) if j != i and k != i])

            row.extend([f"TP:{TP}", f"TN:{TN}"])
            print("\t".join(row))


if __name__ == '__main__':
    pass
