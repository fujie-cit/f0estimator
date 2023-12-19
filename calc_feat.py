
import yaml
from torch.utils.data import DataLoader
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.train.dataset import ESPnetDataset
from espnet2.fileio.npy_scp import NpyScpWriter

import torch

from os import path
import os
import tqdm

def main(config_filename):
    cfg = yaml.safe_load(open(config_filename))
    frontend = DefaultFrontend(**cfg['frontend'])

    info = [
        (cfg['train_wav_scp_filename'],
         cfg['train_feat_dir'],
         cfg['train_feat_scp_filename']),
        (cfg['val_wav_scp_filename'],
         cfg['val_feat_dir'],
         cfg['val_feat_scp_filename']),
    ]

    for wav_scp_filename, out_data_dir, out_scp_filename in info:
        if not path.exists(out_data_dir):
            os.makedirs(out_data_dir, exist_ok=True)
        if not path.exists(path.dirname(out_scp_filename)):
            os.makedirs(path.dirname(out_scp_filename), exist_ok=True)

        print (wav_scp_filename)
        dataset = ESPnetDataset(
            [(wav_scp_filename, "wav", "sound")],
        )
        num_data = len(dataset.loader_dict["wav"])
        npy_scp_writer = NpyScpWriter(out_data_dir, out_scp_filename)

        for i in tqdm.tqdm(range(num_data)):
            utt_id, data = dataset[i]
            wav = torch.tensor(data["wav"]).unsqueeze(0)
            wav_length = torch.tensor([len(wav)]).int()
            feats, _ = frontend(wav, wav_length)
            npy_scp_writer[utt_id] = feats[0].cpu().numpy()


if __name__ == '__main__':
    # コマンドライン引数から，設定ファイルとログディレクトリのパスを取得
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', default='config.yaml')
    args = parser.parse_args()
    main(args.config_filename)
