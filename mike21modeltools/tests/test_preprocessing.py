import pytest
import numpy as np
import pandas as pd
import os

from mike21modeltools.LatentModel.preprocessing import (
    prepare_datasets,
    generate_raw_images,
    generate_raw_images_one_track,
)


@pytest.fixture
def generate_test_csv(
    start_time="2024-01-01 00:00:00",
    time_interval="30min",
    num_tracks=1,
    num_pts=182,
    num_time_steps=97,
):
    """
    Generates a test DataFrame with randomized data and actual timestamps for time.

    Parameters:
        start_time (str): Start time for the time series.
        time_interval (str): Frequency string (e.g., '30T' for 30 minutes) for time steps.
        num_tracks (int): Number of unique track_ids.
        num_pts (int): Number of unique pt_ids per track.
        num_time_steps (int): Number of time steps per track.

    Returns:
        pd.DataFrame: Generated DataFrame with randomized data and actual time.
    """
    data = []

    # Generate a time range for the specified number of time steps
    time_range = pd.date_range(
        start=start_time, periods=num_time_steps, freq=time_interval
    )

    for track_id in range(1, num_tracks + 1):
        for time_step in time_range:
            for pt_id in range(1, num_pts + 1):
                data.append(
                    {
                        "track_id": track_id,
                        "pt_id": int(pt_id),
                        "time": time_step,
                        "Hs": np.random.rand(),  # Random wave height
                        "Wspd": np.random.rand() * 10,  # Random wind speed
                        "Wdir": np.random.rand() * 360,  # Random wind direction
                    }
                )

    return pd.DataFrame(data)


@pytest.fixture
def sample_npy_data():
    return np.random.rand(1000, 13, 14, 3)


def test_prepare_datasets(sample_npy_data):
    sequence_length = 10
    train_data, val_data, test_data = prepare_datasets(sample_npy_data, sequence_length)

    total_sequences = len(sample_npy_data) // sequence_length
    assert (
        train_data.shape[0] == int(0.8 * total_sequences) * sequence_length
    ), "Incorrect training data size"
    assert (
        val_data.shape[0] == int(0.15 * total_sequences) * sequence_length
    ), "Incorrect validation data size"
    assert (
        test_data.shape[0]
        == (total_sequences - int(0.8 * total_sequences) - int(0.15 * total_sequences))
        * sequence_length
    ), "Incorrect test data size"


def test_prepare_datasets_error():
    with pytest.raises(ValueError):
        prepare_datasets(np.random.rand(5, 13, 14, 3), sequence_length=97)


def test_generate_raw_images(generate_test_csv):
    images = generate_raw_images(generate_test_csv)

    assert images.shape == (97, 13, 14, 3), "Generated images have incorrect shape"


def test_generate_raw_images_one_track_no_data():
    empty_df = pd.DataFrame(columns=["track_id", "pt_id", "time", "Hs", "Wspd", "Wdir"])

    with pytest.raises(ValueError):
        generate_raw_images_one_track(empty_df, track_id=1)
