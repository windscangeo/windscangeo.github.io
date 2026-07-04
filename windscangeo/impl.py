import os
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn, optim
from tqdm import tqdm
import time

def early_stopping(valid_losses, patience_epochs, patience_loss):  # From @Jing
    """
    Early stopping function to determine if training should stop based on validation losses. From @ Jing Sun

    Args:
        valid_losses (list): List of validation losses recorded during training.
        patience_epochs (int): Number of epochs to wait before stopping if no improvement.
        patience_loss (float): Minimum change in validation loss to consider as an improvement.

    Returns:
        bool: True if training should stop, False otherwise.
    """
    if len(valid_losses) < patience_epochs:
        return False
    recent_losses = valid_losses[-patience_epochs:]

    if all(x >= recent_losses[0] for x in recent_losses):
        return True

    if max(recent_losses) - min(recent_losses) < patience_loss:
        return True
    return False


def manage_saved_models(directory):  # From @Jing
    """
    Manage saved model files in the specified directory by deleting older epoch files.
    Keeps only the latest epoch file and deletes all others. From @ Jing Sun

    Args:
        directory (str): The directory where model files are saved.
    """

    pattern = re.compile(r"epoch_(\d+)\.pth")
    epoch_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                epoch_num = int(match.group(1))
                file_path = os.path.join(root, file)
                epoch_files.append((file_path, epoch_num))

    # Check if there are more than 5 files
    if len(epoch_files) > 1:
        epoch_files.sort(key=lambda x: x[1])
        files_to_delete = len(epoch_files) - 1

        for i in range(files_to_delete):
            os.remove(epoch_files[i][0])


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    weight_decay,
    criterion,
    device,
    optimizer_choice,
    patience_epochs,
    patience_loss,
    path_folder,
):
    
    """
    Train the model with the given parameters dictionary and save the best validation outputs, labels, and model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        criterion (torch.nn.Module): Loss function to be used.
        device (torch.device): Device to run the model on (CPU or GPU).
        optimizer_choice (str): Choice of optimizer ('Adam', 'SGD', 'RMSprop').
        patience_epochs (int): Number of epochs to wait before stopping if no improvement in validation loss.
        patience_loss (float): Minimum change in validation loss to consider as an improvement.
        path_folder (str): Path to save the model checkpoints.

    Returns:
        best_val_outputs (numpy.ndarray): Best validation outputs from the model.
        best_val_labels (numpy.ndarray): Best validation labels corresponding to the outputs.
        best_model (torch.nn.Module): The best model based on validation loss.
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
    """


    if optimizer_choice == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer choice. Please choose 'Adam' or 'SGD'.")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    # Initialize lists to store loss values and validation predictions
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_outputs = None
    best_val_labels = None


    pbar = tqdm(range(num_epochs), desc="TRAIN : Training Progress")    
    for epoch in pbar:
        # Training Phase
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:

            # Move data to GPU
            images = images.to(device)
            targets = targets.to(device)            

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, targets)  # Ensure target shape matches output
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        #print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")
        pbar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}"})

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_outputs = []  # Temporary list to store outputs for this epoch
        val_labels = []  # Temporary list to store labels for this epoch
        with torch.no_grad():
            for images, targets in val_loader:
                # Move data to GPU
                images = images.to(device)
                targets = targets.to(device)

                # Get model output
                outputs = model(images).squeeze(-1)

                # Calculate loss
                loss = criterion(outputs, targets)  # Ensure target shape matches output
                val_loss += loss.item()

                # Append outputs and targets to lists
                val_outputs.append(outputs.cpu())  # Move to CPU for concatenation
                val_labels.append(targets.cpu())

        # Concatenate outputs and labels across all batches to ensure all samples are included
        val_outputs = torch.cat(val_outputs, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()

        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        #print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
        pbar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}", "Val Loss": f"{avg_val_loss:.4f}"})

        # Check if this is the best validation loss so far and store best outputs/labels
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_outputs = val_outputs
            best_val_labels = val_labels

            best_model = model
            best_model_state = model.state_dict()
            torch.save(
                best_model_state, os.path.join(path_folder, f"./epoch_{epoch + 1}.pth")
            )
            

        manage_saved_models(path_folder)

        if early_stopping(val_losses, patience_epochs, patience_loss):
            return (
                best_val_outputs,
                best_val_labels,
                best_model,
                train_losses,
                val_losses,
            )
        scheduler.step()
    
    

    return best_val_outputs, best_val_labels, model, train_losses, val_losses


def test_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset and return the outputs, targets, and average loss.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function to be used for evaluation.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        test_outputs (numpy.ndarray): Outputs from the model on the test dataset.
        test_targets (numpy.ndarray): Targets corresponding to the test outputs.
        avg_test_loss (float): Average loss on the test dataset.
    """

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference

        test_outputs = []
        test_targets = []
        test_loss = 0.0

        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Append outputs to the list
            test_outputs.append(outputs)
            test_targets.append(targets)

        avg_test_loss = test_loss / len(test_loader)
        print(f"EVAL : Test Loss: {avg_test_loss}")

        test_outputs = torch.cat(test_outputs, dim=0)
        test_outputs = test_outputs.cpu()
        test_outputs = test_outputs.numpy()

        test_targets = torch.cat(test_targets, dim=0)
        test_targets = test_targets.cpu()
        test_targets = test_targets.numpy()

    return test_outputs, test_targets, avg_test_loss


def inference_model(model, inference_loader, device):
    """
    Perform inference on the model using the provided DataLoader and return the outputs. Same as train_model but for a fixed given model.

    Args:
        model (torch.nn.Module): The trained model to be used for inference.
        inference_loader (torch.utils.data.DataLoader): DataLoader for the inference dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
        
    Returns:
        inference_outputs (numpy.ndarray): Outputs from the model on the inference dataset.
    """

    with torch.no_grad():  # Disable gradient calculation for inference

        inference_outputs = []

        for images in inference_loader:
            images = images.to(device)

            outputs = model(images).squeeze(-1)

            # Append outputs to the list
            inference_outputs.append(outputs)

        inference_outputs = torch.cat(inference_outputs, dim=0)
        inference_outputs = inference_outputs.cpu()
        inference_outputs = inference_outputs.numpy()

    return inference_outputs

