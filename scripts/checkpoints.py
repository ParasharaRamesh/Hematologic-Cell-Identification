import os
import torch


def load_checkpointed_model_params(model, optimizer, resume_checkpoint):
    checkpoint = torch.load(resume_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    # Things we are keeping track of
    epoch_numbers = checkpoint['epoch_numbers']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    training_accuracy = checkpoint['training_accuracy']
    validation_accuracy = checkpoint['validation_accuracy']
    print(f"Model checkpoint {resume_checkpoint} loaded! Will resume the epochs from number #{start_epoch}")
    return model, optimizer, start_epoch, epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy


def save_model_checkpoint(experiment, model, optimizer, params, epoch, epoch_numbers, training_losses,
                          validation_losses, training_accuracy, validation_accuracy):
    # set the model to train mode so that in case there was a validation before it doesnt impact the saved weights (as we have dropouts!)
    model.train()

    # create the directory if it doesn't exist
    model_save_directory = os.path.join(params["save_dir"], experiment)
    os.makedirs(model_save_directory, exist_ok=True)

    # Checkpoint the model at the end of each epoch
    checkpoint_path = os.path.join(params["save_dir"], experiment, f'model_epoch_{epoch + 1}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'epoch_numbers': epoch_numbers,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy,
    }, checkpoint_path)
    print(f"Save checkpointed the model at the path {checkpoint_path}")
