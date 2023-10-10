import torch
from torch import optim, nn
import os
from tqdm.auto import tqdm
from models.pRCC import *
from config.params import *
from scripts.checkpoints import load_checkpointed_model_params

'''
TODO.x

Can probably have a Trainer class defined which has two subclasses ClassificationTrainer and AutoEncoderTrainer

will be easier to compute the statistics this way!

'''


# TODO. Contains the code for the most generic training script which can accept hooks

def train_model(
        model,
        train_loader,
        val_loader,
        loss_criterion,
        num_epochs,
        experiment,
        epoch_saver_count=2,
        save_dir=None,
        resume_checkpoint=None,
        is_classification_task=False,
        start_epoch_from_zero=False
):
    '''

    :param model:
    :param train_loader:
    :param val_loader:
    :param loss_criterion:
    :param num_epochs:
    :param experiment:
    :param epoch_saver_count:
    :param save_dir: save in the same directory if not present
    :param resume_checkpoint:
    :param device:
    :param is_classification_task:
    :param start_epoch_from_zero:
    :return:
    '''
    # TODO. write docs
    # 0. Boiler plate needed for scheduler
    torch.cuda.empty_cache()

    # 1.a load the train params
    train_params = init_train_params(model, resume_checkpoint, is_classification_task, start_epoch_from_zero)

    # 1.b Based on the type load the params
    if is_classification_task:
        model, optimizer, start_epoch, epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy = train_params
    else:
        model, optimizer, start_epoch, epoch_numbers, training_losses, validation_losses = train_params

    # 1.c Set up one-cycle learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    # 2. Custom progress bar for total epochs with color and displaying average epoch loss
    total_progress_bar = tqdm(total=num_epochs, desc=f"Total Epochs", position=0,
                              bar_format="{desc}: {percentage}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                              dynamic_ncols=True, ncols=100, colour='red')

    # 3 training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # set to train mode
        model.train()

        epoch_training_loss = 0.0
        train_correct_predictions = 0
        total_samples = 0

        # Custom progress bar for each epoch with color
        epoch_progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}",
                                  position=1, leave=False, dynamic_ncols=True, ncols=100, colour='green')

        for batch_idx, data in enumerate(train_loader):
            # TODO.x 1 this is unique and can be a hook! (data_batch_hook)
            # get the data and outputs
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # TODO.x 2 this is also unique and can be a hook (loss_hook)
            output_logits = model(images)
            loss = loss_criterion(output_logits, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # scheduler update
            lr_scheduler.step()

            epoch_training_loss += loss.item()

            # batch stats
            # Compute training accuracy for this batch
            # TODO.x 3 this only applies if accuracy is being evaluated. (train_accuracy_hook)
            output_probs = nn.Softmax(dim=1)(output_logits)
            predicted = torch.argmax(output_probs, 1)
            batch_correct_predictions = (predicted == labels).sum().item()
            batch_size = labels.size(0)

            train_correct_predictions += batch_correct_predictions
            total_samples += batch_size  # batch size basically

            # TODO.x 4 this can be different ( part of the return of the train_hook)
            # Update the epoch progress bar (overwrite in place)
            epoch_progress_bar.set_postfix({
                "loss": loss.item(),
                "batch_acc": batch_correct_predictions / batch_size
            })
            epoch_progress_bar.update(1)

        # Close the epoch progress bar
        epoch_progress_bar.close()

        # Calculate average loss for the epoch
        avg_training_loss_for_epoch = epoch_training_loss / len(train_loader)

        # TODO.x not needed if not classification task
        # Calculate training accuracy for the epoch (not sure what hook)
        avg_training_accuracy = train_correct_predictions / total_samples

        # Validation loop
        # TODO.x 5 this can be different (validation_hook)
        avg_val_accuracy, avg_val_loss_for_epoch = perform_validation(criterion, device, model, val_loader)

        # TODO.x this can be different
        # Store values
        training_accuracy.append(avg_training_accuracy)
        training_losses.append(avg_training_loss_for_epoch)
        validation_accuracy.append(avg_val_accuracy)
        validation_losses.append(avg_val_loss_for_epoch)
        epoch_numbers.append(epoch + 1)

        # TODO.x this can be different
        # Update the total progress bar
        total_progress_bar.set_postfix(
            {
                "loss": avg_training_loss_for_epoch,
                "train_acc": avg_training_accuracy,
                "val_loss": avg_val_loss_for_epoch,
                "val_acc": avg_val_accuracy,
            }

        )

        # Close the tqdm bat
        total_progress_bar.update(1)

        # TODO.x this can be different
        # Print state
        print(
            f'Epoch {epoch + 1}: train_loss: {avg_training_loss_for_epoch} | train_accuracy: {avg_training_accuracy} | val_loss: {avg_val_loss_for_epoch} | val_accuracy: {avg_val_accuracy} '
        )

        # TODO.x save (checkpoint hook)
        # Save model checkpoint periodically
        need_to_save_model_checkpoint = (epoch + 1) % epoch_saver_count == 0
        if need_to_save_model_checkpoint:
            print(f"Going to save model @ Epoch:{epoch + 1}")
            save_model_checkpoint(
                experiment,
                model,
                optimizer,
                params,
                epoch,
                epoch_numbers,
                training_losses,
                validation_losses,
                training_accuracy,
                validation_accuracy
            )

    # Close the total progress bar
    total_progress_bar.close()

    # Return things needed for plotting
    if is_classification_task:
        return epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy
    else:
        return epoch_numbers, training_losses, validation_losses


def init_train_params(model, resume_checkpoint, is_accuracy_required=False, start_epoch_from_zero=False):
    '''


    :param model:
    :param resume_checkpoint:
    :param is_accuracy_required:
    :param start_epoch_from_zero:
    :return:
    '''
    # Things we are keeping track of
    start_epoch = 0
    epoch_numbers = []
    training_losses = []
    validation_losses = []

    training_accuracy = []
    validation_accuracy = []

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    checkpoint = None

    # load checkpoint
    if resume_checkpoint:
        checkpoint = load_checkpointed_model_params(
            model,
            optimizer,
            resume_checkpoint,
            is_accuracy_required
        )

    # in case accuracy is required
    if is_accuracy_required and resume_checkpoint:
        model, optimizer, start_epoch, epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy = checkpoint
        start_epoch = 0 if start_epoch_from_zero else start_epoch
        return model, optimizer, start_epoch, epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy
    elif is_accuracy_required and not resume_checkpoint:
        start_epoch = 0 if start_epoch_from_zero else start_epoch
        return model, optimizer, start_epoch, epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy
    elif not is_accuracy_required and resume_checkpoint:
        model, optimizer, start_epoch, epoch_numbers, training_losses, validation_losses = checkpoint
        start_epoch = 0 if start_epoch_from_zero else start_epoch
        return model, optimizer, start_epoch, epoch_numbers, training_losses, validation_losses
    else:
        start_epoch = 0 if start_epoch_from_zero else start_epoch
        return model, optimizer, start_epoch, epoch_numbers, training_losses, validation_losses


if __name__ == '__main__':
    pass
    # params = {
    #     'batch_size': 32,
    #     'learning_rate': 0.0045,
    #     'save_dir': 'model_ckpts'
    # }
    # train_data_loader = create_train_data_loader(32)
    # test_data_loader, validation_data_loader = create_test_and_validation_data_loader(32)
    #
    # full_experiment = "Full Data"
    # # Check if GPU is available, otherwise use CPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # full_cifar_model = CIFARClassifier()
    # full_cifar_model.to(device)

    # train_model(
    #     full_cifar_model,
    #     train_data_loader,
    #     validation_data_loader,
    #     2,
    #     params,
    #     full_experiment,
    #     epoch_saver_count=1,
    #     resume_checkpoint=None
    # )
