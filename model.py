import torch
import torch.nn as nn

from espnet2.asr.encoder.contextual_block_conformer_encoder import ContextualBlockConformerEncoder
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,
)
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask


class F0EstimationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Conv2dSubsamplingWOPosEnc(
            160, # input_size
            256, # output_size
            0.1, # dropout_rate
            kernels=[3, 3], 
            strides=[2, 2]
        )
        self.lstm = nn.LSTM(256, 256, 2, dropout=0.2, batch_first=True, bidirectional=False)
        self.fc1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.f0 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.df0 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.vuv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
    
    def forward(self, x, lengths):
        masks = (~make_pad_mask(lengths)[:, None, :]).to(x.device)
        x, masks = self.embed(x, masks)
        lstm_out, (hn, cn) = self.lstm(x)
        x = lstm_out
        x = x.transpose(1, 2)
        # x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)

        f0 = self.f0(x)
        f0 = f0.squeeze(1)

        df0 = self.df0(x)
        df0 = df0.squeeze(1)

        vuv = self.vuv(x)
        vuv = vuv.squeeze(1)

        olens = masks.squeeze(1).sum(1)
        # import ipdb; ipdb.set_trace()
        
        return f0, df0, vuv, olens

class F0EstimationModel_(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ContextualBlockConformerEncoder(
            input_size=160, 
            output_size=256,
            num_blocks=3,)
        self.fc1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.f0 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.df0 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.vuv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
    
    def forward(self, x, lengths):
        x, out_lengths, _ = self.encoder(x, lengths)
        x = x.transpose_(1, 2)

        x = self.fc1(x)
        x = torch.relu(x)

        f0 = self.f0(x)
        f0 = f0.squeeze(1)

        df0 = self.df0(x)
        df0 = df0.squeeze(1)

        vuv = self.vuv(x)
        vuv = vuv.squeeze(1)

        return f0, df0, vuv, out_lengths


class F0EstimationModel_(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ContextualBlockConformerEncoder(
            input_size=160, 
            output_size=256,
            num_blocks=3,)
        self.fc1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.f0 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.df0 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.vuv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
    
    def forward(self, x, lengths):
        x, out_lengths, _ = self.encoder(x, lengths)
        x = x.transpose_(1, 2)

        x = self.fc1(x)
        x = torch.relu(x)

        f0 = self.f0(x)
        f0 = f0.squeeze(1)

        df0 = self.df0(x)
        df0 = df0.squeeze(1)

        vuv = self.vuv(x)
        vuv = vuv.squeeze(1)

        return f0, df0, vuv, out_lengths

            

