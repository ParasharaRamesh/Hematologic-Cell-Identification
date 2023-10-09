def perform_test(criterion, device, model, test_loader):
    test_loss = 0.0
    test_correct_predictions = 0
    total_val_samples = 0

    #set to eval mode
    model.eval()

    with torch.no_grad():
        for val_batch_idx, val_data in enumerate(test_loader):
            test_images, test_labels = val_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_logits = model(test_images)

            # Validation loss update
            test_loss += criterion(test_logits, test_labels).item()

            # Compute validation accuracy for this batch
            test_probs = nn.Softmax(dim=1)(test_logits)
            test_predicted = torch.argmax(test_probs, 1)
            total_val_samples += test_labels.size(0)
            test_correct_predictions += (test_predicted == test_labels).sum().item()

    # Calculate average validation loss for the epoch
    avg_test_loss = test_loss / len(test_loader)
    # Calculate validation accuracy for the epoch
    avg_test_accuracy = test_correct_predictions / total_val_samples
    return avg_test_accuracy, avg_test_loss