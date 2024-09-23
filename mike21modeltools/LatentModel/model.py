import torch.nn as nn
import torch


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) module with stacked dilated convolutions,
    ReLU activation, and dropout.

    Attributes:
        network (nn.Sequential): The sequential model containing all layers of the TCN.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Initializes the TemporalConvNet.

        Args:
            num_inputs (int): Number of input channels.
            num_channels (list of int): List containing the number of channels for each layer.
            kernel_size (int): Size of the convolutional kernel. Default is 2.
            dropout (float): Dropout probability. Default is 0.2.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation_size,
                    dilation=dilation_size,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_LSTMModel(nn.Module):
    """
    Model combining a Temporal Convolutional Network (TCN) with an LSTM and a fully connected layer.

    Attributes:
        seq_length (int): The sequence length of the input data.
        tcn (TemporalConvNet): The TCN module.
        lstm (nn.LSTM): The LSTM module.
        fc (nn.Linear): The fully connected output layer.
    """

    def __init__(
        self,
        num_inputs,
        num_channels,
        hidden_size,
        num_layers,
        output_size,
        seq_length,
        kernel_size=2,
        dropout=0.2,
    ):
        """
        Initializes the TCN-LSTM model.

        Args:
            num_inputs (int): Number of input channels for the TCN.
            num_channels (list of int): List containing the number of channels for each TCN layer.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of recurrent layers in the LSTM.
            output_size (int): Number of output features.
            seq_length (int): The sequence length of the input data.
            kernel_size (int): Size of the TCN convolutional kernel. Default is 2.
            dropout (float): Dropout probability. Default is 0.2.
        """
        super(TCN_LSTMModel, self).__init__()
        self.seq_length = seq_length
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.lstm = nn.LSTM(num_channels[-1], hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, _, _ = x.size()
        # Transpose to (batch_size, num_features, seq_len) for TCN
        x = x.transpose(1, 2)
        # Apply TCN
        tcn_out = self.tcn(x)
        # Transpose back to (batch_size, seq_len, num_channels[-1])
        tcn_out = tcn_out.transpose(1, 2)

        # Ensure tcn_out matches seq_length
        tcn_out = tcn_out[:, : self.seq_length, :]

        # Apply LSTM
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
            x.device
        )
        lstm_out, _ = self.lstm(tcn_out, (h0, c0))
        # Apply fully connected layer
        output = self.fc(lstm_out)
        return output
