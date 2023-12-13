from typing import Any
import warnings
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from espnet2.train.dataset import ESPnetDataset

class WavF0Dataset(Dataset):
    def __init__(self, 
                 wav_scp_filename,
                 f0data_scp_filename,):
        super().__init__()
        self.espnet_dataset = ESPnetDataset(
            [(wav_scp_filename, "wav", "sound"),
             (f0data_scp_filename, "f0data", "npy")])
  
    def __getitem__(self, index) -> Any:
        return self.espnet_dataset[index]
    
    def __len__(self) -> int:
        return len(self.espnet_dataset.loader_dict['wav'])


def collate_wav_f0(batch):
    utt_ids = []
    wav = []
    wav_lengths = []
    f0data = []
    f0data_lengths = []
    
    for (utt, data) in batch:
        utt_ids.append(utt)
        wav.append(torch.tensor(data["wav"], dtype=torch.float32))
        wav_lengths.append(len(data["wav"]))
        f0data.append(torch.tensor(data["f0data"], dtype=torch.float32))
        f0data_lengths.append(len(data["f0data"]))

    wav = pad_sequence(wav, batch_first=True)
    wav_lengths = torch.tensor(wav_lengths, dtype=torch.long)
    f0data = pad_sequence(f0data, batch_first=True)
    f0data_lengths = torch.tensor(f0data_lengths, dtype=torch.long)

    return {'wav': wav, 'wav_lengths': wav_lengths, 'f0data': f0data, 'f0data_lengths': f0data_lengths, 'utt_id': utt_ids}
