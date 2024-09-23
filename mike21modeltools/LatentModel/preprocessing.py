import numpy as np
import pandas as pd
import glob
import os


def combine_mike_files(input_path, output_path):
    """
    Combines all MIKE files in the specified input directory into a single DataFrame, filters the data,
    and saves it as a CSV file.

    Parameters:
        input_path (str): Directory path containing the input CSV files.
        output_path (str): File path to save the combined CSV file.

    Returns:
        pd.DataFrame: The combined and filtered DataFrame.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    if not input_path.endswith("/"):
        input_path += "/"

    all_files = glob.glob(os.path.join(input_path, "*.csv"))
    dfs = [pd.read_csv(filename) for filename in all_files]

    combined_df = pd.concat(dfs, ignore_index=True).fillna(0)

    time_steps_per_storm = (
        combined_df.groupby(["track_id", "pt_id"]).size().reset_index(name="time_steps")
    )
    rows_to_remove_df = time_steps_per_storm[time_steps_per_storm.time_steps != 97]

    combined_df_filtered = combined_df.merge(
        rows_to_remove_df, on=["track_id", "pt_id"], how="left", indicator=True
    )
    combined_df_filtered = combined_df_filtered[
        combined_df_filtered["_merge"] == "left_only"
    ].drop(columns=["_merge", "time_steps"])

    combined_df_filtered.to_csv(output_path, index=False)

    return combined_df_filtered


def prepare_datasets(
    data_input, sequence_length: int, train_split: float = 0.8, val_split: float = 0.15
):
    """
    Prepares training, validation, and testing datasets from sequential data stored in a NumPy array or file.

    Parameters:
        data_input (str or np.ndarray): Path to the .npy file containing the data or a NumPy array.
        sequence_length (int): Length of each sequence.
        train_split (float): Proportion of data to be used for training. Default is 0.8.
        val_split (float): Proportion of data to be used for validation. Default is 0.15.

    Returns:
        tuple: A tuple containing train_data, val_data, and test_data as NumPy arrays.
    """
    if isinstance(data_input, str):
        data = np.load(data_input)
    elif isinstance(data_input, np.ndarray):
        data = data_input
    else:
        raise TypeError(
            "data_input must be a file path (str) or a NumPy array (np.ndarray)."
        )

    total_sequences = len(data) // sequence_length

    if total_sequences == 0:
        raise ValueError("Data size is too small for the given sequence length.")

    train_sequences = int(train_split * total_sequences)
    val_sequences = int(val_split * total_sequences)
    test_sequences = total_sequences - train_sequences - val_sequences

    if test_sequences <= 0:
        raise ValueError(
            "Insufficient data for the test set. Adjust the split proportions or provide more data."
        )

    train_end = train_sequences * sequence_length
    val_end = (train_sequences + val_sequences) * sequence_length

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end : val_end + (test_sequences * sequence_length)]

    return train_data, val_data, test_data


def generate_raw_images(df):
    """
    Generates raw images for each unique track_id in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        np.ndarray: A 4D array of shape (total time steps, 13, 14, 3) containing the generated images.
    """
    unique_tracks = df["track_id"].unique()
    all_images = []

    for track_id in unique_tracks:
        track_data = df[df["track_id"] == track_id]
        time_steps = track_data["time"].unique()

        track_images = np.zeros((len(time_steps), 13, 14, 3))

        for i, time_step in enumerate(time_steps):
            time_data = track_data[track_data["time"] == time_step]
            image = np.zeros((13, 14, 3))

            for _, row in time_data.iterrows():
                pt_id = row["pt_id"] - 1
                row_idx = pt_id // 14
                col_idx = pt_id % 14

                image[row_idx, col_idx, 0] = row["Hs"]
                image[row_idx, col_idx, 1] = row["Wspd"]
                image[row_idx, col_idx, 2] = row["Wdir"]

            track_images[i] = image

        all_images.append(track_images)

    return np.concatenate(all_images, axis=0)


def generate_raw_images_one_track(df, track_id):
    """
    Generates raw images for a specific track_id.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        track_id (int): The specific track_id to generate images for.

    Returns:
        np.ndarray: A 4D array of shape (time steps, 13, 14, 3) containing the generated images.
    """
    track_data = df[df["track_id"] == track_id]

    if track_data.empty:
        raise ValueError(f"No data found for track_id: {track_id}")

    time_steps = track_data["time"].unique()
    track_images = np.zeros((len(time_steps), 13, 14, 3))

    for i, time_step in enumerate(time_steps):
        time_data = track_data[track_data["time"] == time_step]
        image = np.zeros((13, 14, 3))

        for _, row in time_data.iterrows():
            pt_id = row["pt_id"] - 1
            row_idx = pt_id // 14
            col_idx = pt_id % 14

            image[row_idx, col_idx, 0] = row["Hs"]
            image[row_idx, col_idx, 1] = row["Wspd"]
            image[row_idx, col_idx, 2] = row["Wdir"]

        track_images[i] = image

    return track_images
