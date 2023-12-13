import numpy as np
import pandas as pd
import torch

class F0DataNormalizer(object):
    """F0DataNormalizer
    """

    def __init__(self, f0data_stats_filename):
        """
        Args:
            f0data_stats_filename (str): f0data_stats_filename
        """
        super().__init__()
        df = pd.read_csv(f0data_stats_filename, 
                         index_col=0, header=None, sep=' ')
        data = df.iloc[:, :].to_numpy()
        data_mean = data[:, :3].mean(axis=0)
        self.f0data_mean = torch.tensor(data_mean, dtype=torch.float32).unsqueeze(0)

        # self.f0data_std  = torch.tensor(data[3:6], dtype=torch.float32).unsqueeze(0)
        data_std = data[:, :3].std(axis=0) + data[:, 3:6].mean(axis=0)
        # data_std = np.array([2.0, 0.5, 2.0])
        self.f0data_std  = torch.tensor(data_std, dtype=torch.float32).unsqueeze(0)


    def normalize(self, f0data):
        """F0の値を正規化する

        Args:
            f0data (torch.Tensor): (B, T, 3)

        Returns:
            torch.Tensor: (B, T, 3)
        """
        f0data_ = f0data.clone()
        f0data_ = (f0data - self.f0data_mean) / self.f0data_std
        # f0data_ = torch.tanh(f0data_)
        f0data_[:, :, 0][f0data[:, :, 0] <= 0] = 0
        f0data_[:, :, 1][f0data[:, :, 2] <= 0] = 0
        f0data_[:, :, 2][f0data[:, :, 2] <= 0] = 0

        # import ipdb; ipdb.set_trace()
        return f0data_
    
    def denormalize(self, f0data):
        """正規化されたF0の値を元に戻す

        Args:
            f0data (torch.Tensor): (B, T, 3)

        Returns:
            torch.Tensor: (B, T, 3)
        """
        f0data_ = f0data.clone()
        # f0data_ = torch.atanh(f0data_)
        f0data_ = f0data_ * self.f0data_std + self.f0data_mean
        # f0data_[:, :, 0][f0data[:, :, 0] <= 0] = 0
        # f0data_[:, :, 1][f0data[:, :, 2] <= 0] = 0
        # f0data_[:, :, 2][f0data[:, :, 2] <= 0] = 0
        return f0data_
