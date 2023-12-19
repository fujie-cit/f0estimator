import torch
import torch.nn as nn

from espnet2.asr.encoder.contextual_block_conformer_encoder import ContextualBlockConformerEncoder
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class F0EstimationModelCNN(torch.nn.Module):
    def __init__(self,
                 input_size=160, hidden_size=256, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        self.embed = Conv2dSubsamplingWOPosEnc(
            input_size, # input_size
            hidden_size, # output_size
            0.1, # dropout_rate (not used)
            kernels=[3, 3], 
            strides=[2, 2]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers += [
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        self.hidden_layer = nn.Sequential(*hidden_layers)
        self.fc = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.out = nn.Conv1d(in_channels=hidden_size, out_channels=3, kernel_size=1)

    def forward(self, x, lengths):
        # embedding
        masks = (~make_pad_mask(lengths)[:, None, :]).to(x.device)
        x, masks = self.embed(x, masks)
        x = self.layer_norm(x)
        # hidden layers
        x = x.transpose(1, 2)
        x = self.hidden_layer(x)
        # output layer
        x = self.fc(x)
        x = torch.relu(x)
        out = self.out(x)

        f0 = out[:, 0, :].squeeze(1)
        df0 = out[:, 1, :].squeeze(1)
        vuv = out[:, 2, :].squeeze(1)
        olens = masks.squeeze(1).sum(1)

        # import ipdb; ipdb.set_trace()
        
        return f0, df0, vuv, olens


class F0EstimationModelLSTM(torch.nn.Module):
    def __init__(self, input_size=160, hidden_size=256, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        self.embed = Conv2dSubsamplingWOPosEnc(
            input_size, # input_size
            hidden_size, # output_size
            0.1, # dropout_rate (not used)
            kernels=[3, 3], 
            strides=[2, 2]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_hidden_layers, dropout=dropout_rate, batch_first=True, bidirectional=False)
        self.fc = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.out = nn.Conv1d(in_channels=hidden_size, out_channels=3, kernel_size=1)
    
    def forward(self, x, lengths):
        # embedding
        masks = (~make_pad_mask(lengths)[:, None, :]).to(x.device)
        x, masks = self.embed(x, masks)
        x = self.layer_norm(x)
        
        # LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        x = lstm_out

        # output layer
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = torch.relu(x)
        out = self.out(x)

        f0 = out[:, 0, :].squeeze(1)
        df0 = out[:, 1, :].squeeze(1)
        vuv = out[:, 2, :].squeeze(1)
        olens = masks.squeeze(1).sum(1)

        # import ipdb; ipdb.set_trace()

        return f0, df0, vuv, olens


class F0EstimationModelCBS(torch.nn.Module):
    def __init__(self, input_size=160, hidden_size=256, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        self.encoder = ContextualBlockConformerEncoder(
            input_size=input_size, 
            output_size=hidden_size,
            num_blocks=num_hidden_layers,
            dropout_rate=dropout_rate,)
        self.fc1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.out = nn.Conv1d(in_channels=hidden_size, out_channels=3, kernel_size=1)
    
    def forward(self, x, lengths):
        # embedding & encoder
        x, out_lengths, _ = self.encoder(x, lengths)

        # output layer
        x = x.transpose_(1, 2)
        x = self.fc1(x)
        x = torch.relu(x)
        out = self.out(x)

        f0 = out[:, 0, :].squeeze(1)
        df0 = out[:, 1, :].squeeze(1)
        vuv = out[:, 2, :].squeeze(1)

        return f0, df0, vuv, out_lengths

