import torch
import torch.nn as nn


def perform_validation(criterion, device, model, val_loader):
    val_loss = 0.0
    val_correct_predictions = 0
    total_val_samples = 0

    #set to eval mode
    model.eval()

    with torch.no_grad():
        for val_batch_idx, val_data in enumerate(val_loader):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_logits = model(val_images)

            # Validation loss update
            val_loss += criterion(val_logits, val_labels).item()

            # Compute validation accuracy for this batch
            val_probs = nn.Softmax(dim=1)(val_logits)
            val_predicted = torch.argmax(val_probs, dim=1)
            total_val_samples += val_labels.size(0)
            val_correct_predictions += (val_predicted == val_labels).sum().item()

    # Calculate average validation loss for the epoch
    avg_val_loss_for_epoch = val_loss / len(val_loader)
    # Calculate validation accuracy for the epoch
    avg_val_accuracy = val_correct_predictions / total_val_samples
    return avg_val_accuracy, avg_val_loss_for_epoch


