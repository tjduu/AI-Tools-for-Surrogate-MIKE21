import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def validate_model(model, val_dataloader, num_pca_components, criterion, device):
    """
    Validates the model on the validation dataset.

    Parameters:
        model (torch.nn.Module): The model to be validated.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_pca_components (int): Number of PCA components in the model output.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on.

    Returns:
        tuple: A tuple containing validation loss, MSE, MAE, and R² score.
    """
    model.eval()
    val_running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)

            val_loss = criterion(val_outputs, val_targets)
            val_running_loss += val_loss.item() * val_inputs.size(0)

            all_predictions.append(val_outputs.cpu().numpy())
            all_targets.append(val_targets.cpu().numpy())

    epoch_val_loss = val_running_loss / len(val_dataloader.dataset)

    all_predictions = np.concatenate(all_predictions).reshape(-1, num_pca_components)
    all_targets = np.concatenate(all_targets).reshape(-1, num_pca_components)

    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    return epoch_val_loss, mse, mae, r2


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    num_pca_components,
    optimizer,
    scheduler,
    early_stopping,
    device,
    num_epochs=30,
):
    """
    Trains the model and validates it at each epoch.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        num_pca_components (int): Number of PCA components in the model output.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        early_stopping (EarlyStopping): EarlyStopping object to monitor validation loss.
        device (torch.device): Device to run the model on.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        tuple: A tuple containing lists of training losses, validation losses, and validation metrics.
    """
    train_losses = []
    val_losses = []
    val_metrics = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)

        epoch_val_loss, mse, mae, r2 = validate_model(
            model, val_dataloader, num_pca_components, criterion, device
        )
        val_losses.append(epoch_val_loss)
        val_metrics.append({"mse": mse, "mae": mae, "r2": r2})

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}"
        )

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        if early_stopping is not None:
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_losses, val_losses, val_metrics


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss does not improve.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        verbose (bool): If True, prints a message for each validation loss check.
        delta (float): Minimum change in validation loss to be considered as an improvement.
        counter (int): Number of epochs without improvement.
        best_score (float): Best validation loss score.
        early_stop (bool): Flag indicating whether to stop early.
        val_loss_min (float): Minimum validation loss observed.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initializes the EarlyStopping object.

        Parameters:
            patience (int): Number of epochs to wait for improvement before stopping.
            verbose (bool): If True, prints a message for each validation loss check.
            delta (float): Minimum change in validation loss to be considered as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        """
        Checks if validation loss has improved and updates early stopping parameters.

        Parameters:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
