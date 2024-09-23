import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class SequenceDatasetPCA(Dataset):
    """
    A PyTorch Dataset class to handle sequence data with PCA transformation
    and normalization.

    Attributes:
        sequence_length (int): Length of the sequences to generate.
        normalize (bool): Whether to apply MinMax normalization.
        scaler_input (MinMaxScaler): Scaler for input normalization.
        scaler_target (MinMaxScaler): Scaler for target normalization.
        pca_model_input (PCA): PCA model for input transformation.
        pca_model_target (PCA): PCA model for target transformation.
        raw_data (array-like): The raw input data.
        inputs (array-like): The processed input sequences.
        targets (array-like): The processed target sequences.
        num_sequences (int): Number of sequences that can be generated from the data.
    """

    def __init__(
        self,
        data,
        sequence_length=97,
        pca_model_input=None,
        pca_model_target=None,
        normalize=True,
    ):
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.scaler_input = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.pca_model_input = pca_model_input
        self.pca_model_target = pca_model_target

        self.raw_data = data
        self.inputs, self.targets = self._prepare_data(data)
        self.num_sequences = len(self.inputs) // self.sequence_length

        if (
            len(self.inputs) < self.sequence_length
            or len(self.targets) < self.sequence_length
        ):
            raise ValueError(
                "Dataset is too short to form any sequence with the given sequence length."
            )

    def _prepare_data(self, data):
        num_samples, height, width, channels = data.shape

        # Normalize and apply PCA to inputs
        inputs_flat = data.reshape(num_samples, height * width * channels)
        if self.normalize:
            inputs_flat = self.scaler_input.fit_transform(inputs_flat)
        inputs_pca = self.pca_model_input.transform(inputs_flat)

        # Normalize and apply PCA to targets
        targets_flat = data[:, :, :, :1].reshape(num_samples, height * width)
        if self.normalize:
            targets_flat = self.scaler_target.fit_transform(targets_flat)
        targets_pca = self.pca_model_target.transform(targets_flat)

        return inputs_pca, targets_pca

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        inputs_sequence = self.inputs[start_idx:end_idx]
        targets_sequence = self.targets[start_idx:end_idx]

        inputs = torch.tensor(inputs_sequence, dtype=torch.float32)
        targets = torch.tensor(targets_sequence, dtype=torch.float32)

        return inputs, targets

    def inverse_transform_targets(self, data):
        num_samples, num_pca_components = data.shape[:2]
        data_2d = data.reshape(-1, num_pca_components)
        data_inv_pca = self.pca_model_target.inverse_transform(data_2d)
        if self.normalize:
            data_inv_pca = self.scaler_target.inverse_transform(data_inv_pca)

        target_shape = (num_samples, 13, 14, 1)  # Adjust as needed
        data_inv_pca = data_inv_pca.reshape(target_shape)
        return data_inv_pca

    def get_raw_target(self, sequence_idx):
        start_idx = sequence_idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        return self.raw_data[start_idx:end_idx, :, :, :1]
