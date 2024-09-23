import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm


def flatten_and_normalize(data, scaler):
    """
    Flattens and normalizes the input data using the provided scaler.

    Parameters:
        data (np.ndarray): The input data to be flattened and normalized.
        scaler (sklearn.preprocessing): The scaler to be used for normalization.

    Returns:
        np.ndarray: The flattened and normalized data.
    """
    data_flat = data.reshape(data.shape[0], -1)
    return scaler.fit_transform(data_flat)


def plot_loss(train_losses, val_losses):
    """
    Plots the training and validation loss over epochs.

    Parameters:
        train_losses (list of float): The training losses.
        val_losses (list of float): The validation losses.

    Returns:
        None: Displays the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_percentage_of_underestimate(
    predictions, actuals, threshold_true=2.0, threshold_diff=1.0
):
    """
    Calculates the percentage of underestimates where actual values exceed a threshold.

    Parameters:
        predictions (np.ndarray): The predicted values.
        actuals (np.ndarray): The actual values.
        threshold_true (float): The threshold for actual values.
        threshold_diff (float): The threshold for considering a prediction as an underestimate.

    Returns:
        float: The percentage of underestimates.
    """
    mask = actuals >= threshold_true
    differences = np.abs(actuals - predictions)
    underestimates = (mask & (differences > threshold_diff)).sum()
    total_predictions = actuals.size
    return (underestimates / total_predictions) * 100


def calculate_cropped_metrics(predictions, actuals):
    """
    Calculates the MSE, R², MAE, and percentage of large errors on cropped prediction and actual arrays.

    Parameters:
        predictions (np.ndarray): The predicted values.
        actuals (np.ndarray): The actual values.

    Returns:
        tuple: MSE, R², MAE, and percentage of large errors.
    """
    predictions = np.squeeze(predictions)
    actuals = np.squeeze(actuals)

    cropped_predictions = predictions[:, :, 1:-1, 1:-1]
    cropped_actuals = actuals[:, :, 1:-1, 1:-1]

    if cropped_predictions.size == 0 or cropped_actuals.size == 0:
        raise ValueError("Cropped arrays have zero size.")

    cropped_predictions_flat = cropped_predictions.flatten()
    cropped_actuals_flat = cropped_actuals.flatten()

    cropped_mse = mean_squared_error(cropped_actuals_flat, cropped_predictions_flat)
    cropped_r2 = r2_score(cropped_actuals_flat, cropped_predictions_flat)
    cropped_mae = mean_absolute_error(cropped_actuals_flat, cropped_predictions_flat)

    percentage_large_errors = calculate_percentage_of_underestimate(
        cropped_predictions, cropped_actuals
    )

    return cropped_mse, cropped_r2, cropped_mae, percentage_large_errors


def evaluate(model, test_loader, criterion, dataset):
    """
    Evaluates the model on the test dataset and computes various metrics.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        criterion (torch.nn.Module): Loss function.
        dataset (Dataset): Dataset used for inverse transformations.

    Returns:
        None: Prints evaluation metrics and plots differences.
    """
    model.eval()
    test_loss = 0.0
    num_batches = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.unsqueeze(
                2
            )  # Shape: (batch_size, sequence_length, 1, 13, 14)
            batch_size_actual = inputs.shape[0]
            sequence_length_actual = inputs.shape[1]

            outputs = model(inputs)  # Forward pass through the model

            outputs_flat = outputs.cpu().numpy().reshape(-1, outputs.shape[-1])
            targets_flat = targets.cpu().numpy().reshape(-1, targets.shape[-1])

            outputs_denorm = dataset.inverse_transform_targets(outputs_flat)
            targets_denorm = dataset.inverse_transform_targets(targets_flat)

            outputs_denorm = outputs_denorm.reshape(
                batch_size_actual, sequence_length_actual, 13, 14, 1
            )
            targets_denorm = targets_denorm.reshape(
                batch_size_actual, sequence_length_actual, 13, 14, 1
            )

            outputs_denorm_tensor = torch.tensor(outputs_denorm, dtype=torch.float32)
            targets_denorm_tensor = torch.tensor(targets_denorm, dtype=torch.float32)

            loss = criterion(outputs_denorm_tensor, targets_denorm_tensor)
            test_loss += loss.item()
            num_batches += 1

            predictions.append(outputs_denorm)
            actuals.append(targets_denorm)

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    r2 = r2_score(actuals.flatten(), predictions.flatten())
    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    percentage_underestimate = calculate_percentage_of_underestimate(
        predictions, actuals
    )

    print(f"R² score: {r2:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Percentage of Underestimate: {percentage_underestimate:.2f}%")

    try:
        cropped_mse, cropped_r2, cropped_mae, percentage_large_errors = (
            calculate_cropped_metrics(predictions, actuals)
        )
        print(f"Cropped R²: {cropped_r2:.3f}")
        print(f"Cropped MSE: {cropped_mse:.3f}")
        print(f"Cropped MAE: {cropped_mae:.3f}")
        print(
            f"Percentage of large errors (actual >= 2m and error > 1): {percentage_large_errors:.2f}%"
        )
    except ValueError as e:
        print(f"Error in calculating cropped metrics: {e}")


def compare_maxHs(
    model, track_loader, track, device, sequence_length=97, sequence_index=0
):
    """
    Evaluates the model on a specific sequence and plots the differences between predicted and actual values.

    Parameters:
        model (torch.nn.Module): The trained model.
        track_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        track (Dataset): Dataset object with inverse_transform_targets and get_raw_target methods.
        sequence_length (int): Length of the sequence to evaluate. Default is 97.
        sequence_index (int): Index of the sequence to evaluate. Default is 0.
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing MAE and MSE for the evaluated sequence.
    """
    inputs, targets = track_loader.dataset[sequence_index]
    inputs = torch.unsqueeze(inputs, 0).contiguous()

    with torch.no_grad():
        outputs = model(inputs.to(device)).cpu().numpy()

    outputs_flat = outputs.reshape(-1, outputs.shape[-1])
    denormalized_outputs = track.inverse_transform_targets(outputs_flat)
    denormalized_targets = track.get_raw_target(sequence_index)

    denormalized_outputs = denormalized_outputs.reshape(sequence_length, 13, 14, 1)
    denormalized_targets = denormalized_targets.reshape(sequence_length, 13, 14, 1)

    max_values_output = denormalized_outputs.max(axis=0).squeeze()
    max_values_target = denormalized_targets.max(axis=0).squeeze()

    mae = mean_absolute_error(max_values_output.flatten(), max_values_target.flatten())
    mse = mean_squared_error(max_values_output.flatten(), max_values_target.flatten())

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)

    difference = max_values_target - max_values_output
    cmap = plt.cm.seismic
    norm = TwoSlopeNorm(
        vmin=-np.max(np.abs(difference)), vcenter=0, vmax=np.max(np.abs(difference))
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(difference, cmap=cmap, norm=norm, origin="lower")
    plt.colorbar(label="Difference(m)")
    plt.title(
        f"Difference = (Actual Hs - Predicted Hs) on track_id = {sequence_index} \nMAE: {mae:.4f}, MSE: {mse:.4f}"
    )
    plt.xlabel("Pixel X Index")
    plt.ylabel("Pixel Y Index")
    plt.show()


def get_model_outputs(model, dataset, device, sequence_index=0):
    """
    Get the model predictions and targets for a specific sequence.

    Parameters:
        model (torch.nn.Module): The trained model.
        dataset (Dataset): The dataset containing the sequences.
        device (torch.device): The device to run the model on.
        sequence_index (int): The index of the sequence to evaluate.

    Returns:
        np.ndarray: The denormalized model outputs.
        np.ndarray: The denormalized target values.
    """
    inputs, targets = dataset[sequence_index]
    inputs = torch.unsqueeze(inputs, 0).contiguous()

    with torch.no_grad():
        outputs = model(inputs.to(device)).cpu().numpy()

    outputs_flat = outputs.reshape(-1, outputs.shape[-1])
    denormalized_outputs = dataset.inverse_transform_targets(outputs_flat)
    denormalized_targets = dataset.get_raw_target(sequence_index)

    return denormalized_outputs, denormalized_targets


def plot_pixel_value_comparison(
    denormalized_outputs,
    denormalized_targets,
    pixel_x=None,
    pixel_y=None,
    sequence_length=97,
):
    """
    Plots the predicted vs. actual pixel values and their differences across the sequence.

    Parameters:
        denormalized_outputs (np.ndarray): The predicted outputs after inverse transformation.
        denormalized_targets (np.ndarray): The actual target values after inverse transformation.
        pixel_x (int): The x-coordinate of the pixel.
        pixel_y (int): The y-coordinate of the pixel.
        sequence_length (int): The length of the sequence (number of time-steps).

    Returns:
        None: Displays the plot.
    """
    pixel_values_output = np.squeeze(denormalized_outputs)[:, pixel_x, pixel_y]
    pixel_values_target = np.squeeze(denormalized_targets)[:, pixel_x, pixel_y]

    plt.figure(figsize=(10, 5))
    plt.plot(pixel_values_output, label="Prediction", marker="o")
    plt.plot(pixel_values_target, label="Target", marker="x")
    plt.plot(
        abs(pixel_values_output - pixel_values_target), label="Difference", marker="v"
    )

    plt.title(f"Hs at pixel ({pixel_x}, {pixel_y}) over {sequence_length} Sequences")
    plt.xlabel("Time-steps(0.5h)")
    plt.ylabel("Hs")
    plt.legend()
    plt.show()


def plot_model_predictions(
    denormalized_outputs, denormalized_targets, sequence_length=97
):
    """
    Plots the model predictions, targets, and differences for a specific sequence.

    Parameters:
        denormalized_outputs (np.ndarray): The predicted outputs after inverse transformation.
        denormalized_targets (np.ndarray): The actual target values after inverse transformation.
        sequence_length (int): The length of the sequence to plot.

    Returns:
        None: Displays the plots.
    """
    denormalized_outputs = denormalized_outputs.reshape(-1, sequence_length, 13, 14, 1)
    denormalized_targets = denormalized_targets.reshape(-1, sequence_length, 13, 14, 1)

    plt.figure(figsize=(20, 20))
    for i in range(sequence_length):
        plt.subplot(10, 10, i + 1)
        img = np.squeeze(denormalized_outputs[0, i])
        plt.imshow(img, cmap="viridis", origin="lower")
        plt.title(f"Pred {i + 1}")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.suptitle("Predictions")
    plt.show()

    plt.figure(figsize=(20, 20))
    for i in range(sequence_length):
        plt.subplot(10, 10, i + 1)
        img = np.squeeze(denormalized_targets[0, i])
        plt.imshow(img, cmap="viridis", origin="lower")
        plt.title(f"Target {i + 1}")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.suptitle("Targets")
    plt.show()

    plt.figure(figsize=(20, 20))
    for i in range(sequence_length):
        plt.subplot(10, 10, i + 1)
        diff_img = abs(
            np.squeeze(denormalized_targets[0, i])
            - np.squeeze(denormalized_outputs[0, i])
        )
        plt.imshow(diff_img, cmap="viridis", origin="lower")
        plt.title(f"Diff {i + 1}")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.suptitle("Differences")
    plt.show()
