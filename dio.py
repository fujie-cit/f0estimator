import warnings
import tqdm

import numpy as np
import torch

from espnet2.tts.feats_extract.dio import Dio
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.train.dataset import ESPnetDataset

class DioProcessor(object):
    def __init__(self, 
            fs=16000,
            n_fft=1024,
            hop_length=128,
            use_log_f0=True,
            num_sub=4,):
        self.dio = Dio(fs=fs, n_fft=n_fft, hop_length=hop_length, use_log_f0=use_log_f0,
                       use_token_averaged_f0=False, use_continuous_f0=False)
        self.num_sub = num_sub

    def __call__(self, wav,):
        assert isinstance(wav, np.ndarray), f"wav must be np.ndarray, but got {type(wav)}"
        assert len(wav.shape) == 1, f"wav must be mono, but got {wav.shape[0]} channels"
        
        wav_tensor = torch.tensor(wav, dtype=torch.float32)
        wav_lengths = torch.tensor([len(wav)], dtype=torch.long)
        f0, f0_lengths = self.dio(wav_tensor.unsqueeze(0), wav_lengths)

        f0 = f0.squeeze(0, 2).cpu().numpy()
        
        num_sub = self.num_sub

        # 長さがnum_subの倍数になるようにパディング
        if f0.shape[0] % num_sub != 0:
            pad = num_sub - f0.shape[0] % num_sub
            f0 = np.concatenate([f0, np.zeros(pad)], axis=0)

        # サブサンプルの準備
        f0s = f0.reshape(-1, num_sub)

        # 代表値（f0p）の計算
        if num_sub % 2 == 0:
            # num_subが偶数の場合は，中間の２つの値の平均を取る
            # ただし，0は平均の計算に含めず，両者0の場合は0を返す
            f0p = f0s[:, num_sub//2-1:num_sub//2+1].copy()
            f0p[f0p == 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                f0p = np.nanmean(f0p, axis=1)
            f0p[np.isnan(f0p)] = 0
        else:
            # num_subが奇数の場合は，単純に中間の値を取る
            f0p = f0s[:, num_sub//2].copy()

        # 1次近似の計算
        f0dy = f0s.copy()
        f0dx = np.arange(num_sub) - (num_sub - 1) / 2
        f0ds = []
        for i in range(f0dy.shape[0]):
            nonzero_count = np.sum(f0dy[i] > 0)
            if nonzero_count > num_sub / 2:
                dx = f0dx[f0dy[i] > 0]
                
                dy = f0dy[i][f0dy[i] > 0]
                f0ds.append(np.polyfit(dx, dy, 1))
            else:
                f0ds.append([0, 0])
        f0ds = np.stack(f0ds)

        result = np.concatenate([f0p.reshape(-1, 1), f0ds], axis=1)

        return result


def calc_f0data_stats(f0data):
    f0data = f0data.copy()

    f0data[:, 0][f0data[:, 0] <= 0] = np.nan
    f0data[:, 1][f0data[:, 2] <= 0] = np.nan
    f0data[:, 2][f0data[:, 2] <= 0] = np.nan
    
    f0data_mean = np.nanmean(f0data, axis=0)
    f0data_std = np.nanstd(f0data, axis=0)

    f0data_mean[np.isnan(f0data_mean)] = 0.0
    f0data_std[np.isnan(f0data_std)] = 1.0

    return f0data_mean, f0data_std

def convert_wav_sp_to_f0data_sp(wav_scp_filename, 
                                out_dir, out_scp_filename,
                                out_stats_filename,
                                dio_processor_params):
    dio_processor = DioProcessor(**dio_processor_params)
    dataset = ESPnetDataset([(wav_scp_filename, "wav", "sound"),])

    num_data = len(dataset.loader_dict["wav"])

    npy_scp_writer = NpyScpWriter(out_dir, out_scp_filename)

    if out_stats_filename is not None:
        with open(out_stats_filename, 'w') as f:
            pass

    for i in tqdm.tqdm(range(num_data)):
        utt_id, data = dataset[i]
        # import ipdb; ipdb.set_trace()
        f0data = dio_processor(data["wav"])
        npy_scp_writer[utt_id] = f0data

        f0mean, f0std = calc_f0data_stats(f0data)

        if out_stats_filename is not None:
            str_f0stats = ' '.join([str(x) for x in f0mean.tolist() + f0std.tolist()])
            with open(out_stats_filename, 'a') as f:
                f.write(f"{utt_id} {str_f0stats}\n")


def run_convert(wav_scp_filename, 
                out_dir, out_scp_filename,
                out_stats_filename,
                dio_processor_params):
    assert path.exists(wav_scp_filename), f"{wav_scp_filename} does not exist"
    assert not path.exists(out_dir), f"{out_dir} already exists (remove it first)"

    # out_dir が存在しなければ作成
    if not path.exists(out_dir):
        os.makedirs(out_dir)

   # out_scp_filename のディレクトリが存在しなければ作成
    out_scp_dir = path.dirname(out_scp_filename)
    if not path.exists(out_scp_dir):
        os.makedirs(out_scp_dir)


    convert_wav_sp_to_f0data_sp(wav_scp_filename, 
                                out_dir, out_scp_filename,
                                out_stats_filename,
                                dio_processor_params)

if __name__ == "__main__":
    import yaml
    import argparse
    from os import path
    import os
    import shutil

    # コマンドライン引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config_filename = args.config
    cfg = yaml.safe_load(open(config_filename, 'r'))


    run_convert(cfg["train_wav_scp_filename"],
                cfg["train_f0data_dir"], 
                cfg["train_f0data_scp_filename"],
                cfg["train_f0data_stats_filename"],
                cfg["dio"])
    shutil.copy(config_filename, 
                path.dirname(cfg["train_f0data_scp_filename"]))

    run_convert(cfg["val_wav_scp_filename"],
                cfg["val_f0data_dir"], 
                cfg["val_f0data_scp_filename"],
                None,
                cfg["dio"])
    shutil.copy(config_filename, 
                path.dirname(cfg["val_f0data_scp_filename"]))
