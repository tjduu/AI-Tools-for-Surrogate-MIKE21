import pytest
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mike21modeltools.LatentModel.dataset import (
    SequenceDatasetPCA,
)  # Adjust this import based on where your class is located


@pytest.fixture
def sample_data():
    np.random.seed(0)
    return np.random.rand(100, 13, 14, 3)


@pytest.fixture
def pca_model_input():
    return PCA(n_components=5).fit(np.random.rand(100, 13 * 14 * 3))


@pytest.fixture
def pca_model_target():
    # Create and fit a sample PCA model for target data
    return PCA(n_components=5).fit(np.random.rand(100, 13 * 14))


@pytest.fixture
def sequence_dataset(sample_data, pca_model_input, pca_model_target):
    return SequenceDatasetPCA(
        data=sample_data,
        sequence_length=10,
        pca_model_input=pca_model_input,
        pca_model_target=pca_model_target,
        normalize=True,
    )


def test_sequence_dataset_initialization(sequence_dataset):
    assert (
        len(sequence_dataset) == 10
    ), "Length of dataset does not match expected value"
    assert (
        sequence_dataset.inputs.shape[0] == 100
    ), "Number of inputs sequences is incorrect"
    assert (
        sequence_dataset.targets.shape[0] == 100
    ), "Number of target sequences is incorrect"


def test_sequence_dataset_getitem(sequence_dataset):
    # Test if __getitem__ returns the correct sequences
    inputs, targets = sequence_dataset[0]
    assert isinstance(inputs, torch.Tensor), "Inputs should be a torch tensor"
    assert isinstance(targets, torch.Tensor), "Targets should be a torch tensor"
    assert inputs.shape == (10, 5), "Input sequence shape is incorrect"
    assert targets.shape == (10, 5), "Target sequence shape is incorrect"


def test_sequence_dataset_inverse_transform(sequence_dataset):
    pca_outputs = np.random.rand(10, 5)  # Assuming 5 PCA components
    inv_targets = sequence_dataset.inverse_transform_targets(pca_outputs)
    assert inv_targets.shape == (
        10,
        13,
        14,
        1,
    ), "Inverse transformed target shape is incorrect"


def test_sequence_dataset_get_raw_target(sequence_dataset):
    # Test if get_raw_target correctly retrieves raw data
    raw_target = sequence_dataset.get_raw_target(0)
    assert raw_target.shape == (10, 13, 14, 1), "Raw target shape is incorrect"
    assert np.array_equal(
        raw_target, sequence_dataset.raw_data[:10, :, :, :1]
    ), "Raw target values are incorrect"


def test_sequence_dataset_no_normalization(
    sample_data, pca_model_input, pca_model_target
):
    # Test if the dataset can be created without normalization
    dataset = SequenceDatasetPCA(
        data=sample_data,
        sequence_length=10,
        pca_model_input=pca_model_input,
        pca_model_target=pca_model_target,
        normalize=False,
    )
    assert len(dataset) == 10, "Dataset length is incorrect without normalization"
