import yaml
from os import path
import os
import shutil
import warnings
import copy

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
from timm.scheduler import CosineLRScheduler

from sklearn.model_selection import GroupKFold

from dataset import WavF0Dataset, collate_wav_f0
from model import F0EstimationModel
from train import train
from preprocess import F0DataNormalizer

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.specaug.specaug import SpecAug

def main(config_filename, log_dir):
    # Check if config file exists
    if not path.exists(config_filename):
        raise FileNotFoundError(f"{config_filename} not found.")

    # Check if log_dir exists
    if not path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        warnings.warn(f"{log_dir} already exists.")

    # Copy config file to log_dir
    shutil.copy(config_filename, log_dir)

    # Load config file
    cfg = yaml.safe_load(open(config_filename))

    # Set random seed
    random_seed = cfg['random_seed']
    import random
    random.seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    import torch
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    train_wav_scp_filename = cfg['train_wav_scp_filename']
    train_f0data_scp_filename = cfg['train_f0data_scp_filename']
    train_dataset = WavF0Dataset(train_wav_scp_filename, train_f0data_scp_filename)

    # # subset for test
    # from torch.utils.data import Subset
    # train_dataset = Subset(train_dataset, range(1000))

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_wav_f0, num_workers=4)

    val_wav_scp_filename = cfg['val_wav_scp_filename']
    val_f0data_scp_filename = cfg['val_f0data_scp_filename']
    val_dataset = WavF0Dataset(val_wav_scp_filename, val_f0data_scp_filename)
    
    # # subset for test
    # val_dataset = Subset(val_dataset, range(100))
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, collate_fn=collate_wav_f0, num_workers=4)

    # # model settings
    # input_dim = cfg['input_dim']
    # hidden_dim = cfg['hidden_dim']
    # num_layers = cfg['num_layers']
    # output_dim = cfg['output_dim']

    # optimizer settings
    learning_rate = cfg['learning_rate']
    
    # construct model and save the initial parameters
    device = cfg['device']
    model = F0EstimationModel()
    model = model.to(device)

    frontend = DefaultFrontend(**cfg['frontend'])
    frontend = frontend.to(device)
    spec_aug = SpecAug(**cfg['spec_augment'])

    f0data_stats_filename = cfg['train_f0data_stats_filename']
    f0data_normalizer = F0DataNormalizer(f0data_stats_filename)
    
    # construct model, optimizer, criterion
    optimizer = Adam(model.parameters(), lr=learning_rate)
    sch_warmup_epochs = cfg['sch_warmup_epochs']
    sch_min_lr = float(cfg['sch_min_lr'])
    sch_t_initial = cfg['sch_t_initial']
    scheduler = CosineLRScheduler(optimizer, 
            t_initial=sch_t_initial, lr_min=sch_min_lr, warmup_t=sch_warmup_epochs, 
            warmup_lr_init=1e-7, warmup_prefix=True)

    # train
    num_epochs = cfg['num_epochs']
    train(model, frontend, spec_aug, train_dataloader, val_dataloader, optimizer, scheduler, 
          num_epochs, 
          100, # log_interval
          500, # val_interval
          log_dir, device, f0data_normalizer, f0_target_column=2)
        
if __name__ == '__main__':
    # コマンドライン引数から，設定ファイルとログディレクトリのパスを取得
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', default='config.yaml')
    parser.add_argument('log_dir', default='exp/exp1')
    args = parser.parse_args()
    main(args.config_filename, args.log_dir)
