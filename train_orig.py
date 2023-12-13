import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.specaug import SpecAug

import os
from os import path
import datetime 
import copy
import tqdm

def forward_one_step(model: nn.Module,
                     frontend: AbsFrontend,
                     spec_aug: SpecAug,
                     batch: dict,
                     f0_criterion: nn.Module,
                     df0_critetion: nn.Module,
                     vuv_criterion: nn.Module,
                     device='cpu',
                     f0data_normalizer=None,
                     f0_target_column=2,):
    wav = batch['wav']
    wav_lengths = batch['wav_lengths']

    f0data = batch['f0data']
    f0data_lengths = batch['f0data_lengths']
    
    if f0data_normalizer is not None:
        f0data = f0data_normalizer.normalize(f0data)

    f0_target = f0data[:, :, f0_target_column]
    df0_target = f0data[:, :, 1]
    vuv_target = (f0data[:, :, 0] > 0).float()

    inputs, input_lengths = frontend(wav, wav_lengths)
    if spec_aug is not None:
        inputs, input_lengths = spec_aug(inputs, input_lengths)

    inputs = inputs.to(device)
    
    f0_pred, df0_pred, vuv_pred, pred_lengths = model(inputs, input_lengths)

    # f0_pred, df0_pred, vuv_pred の長さを f0data の長さに合わせる
    if f0_pred.shape[1] > f0_target.shape[1]:
        f0_pred = f0_pred[:, :f0_target.shape[1]]
        df0_pred = df0_pred[:, :f0_target.shape[1]]
        vuv_pred = vuv_pred[:, :f0_target.shape[1]]
    elif f0_pred.shape[1] < f0_target.shape[1]:
        f0_target = f0_target[:, :f0_pred.shape[1]]
        df0_target = df0_target[:, :f0_pred.shape[1]]
        vuv_target = vuv_target[:, :f0_pred.shape[1]]

    # f0_pred, df0_pred, vuv_pred の長さを f0data_lengths の長さに合わせる
    f0data_lengths = f0data_lengths.to(device)
    pred_lengths = torch.min(pred_lengths, f0data_lengths)
    pred_lengths = pred_lengths.clamp(max=f0_pred.shape[1])

    # lossを計算するためのマスクを生成する
    # f0, df0 に関しては，f0data_lengthsの範囲，かつ f0 が 0 でないところがTrue
    # それ以外は False
    f0_mask = torch.zeros_like(f0_target)
    for i in range(f0_target.shape[0]):
        f0_mask[i, :pred_lengths[i]] = f0_target[i, :pred_lengths[i]] > 0
    f0_mask = f0_mask.bool()
    # vuv に関しては，f0data_lengthsの範囲がTrue
    vuv_mask = torch.zeros_like(vuv_target)
    for i in range(vuv_target.shape[0]):
        vuv_mask[i, :pred_lengths[i]] = 1
    vuv_mask = vuv_mask.bool()

    f0_count = f0_mask.sum().item()
    df0_count = f0_mask.sum().item()
    vuv_count = vuv_mask.sum().item()

    f0_target = f0_target.to(device)
    f0_loss = f0_criterion(f0_pred, f0_target)[f0_mask].sum() / f0_count
    
    df0_target = df0_target.to(device)
    df0_loss = df0_critetion(df0_pred, df0_target)[f0_mask].sum() / df0_count
    
    vuv_target = vuv_target.to(device)
    vuv_loss = vuv_criterion(vuv_pred, vuv_target)[vuv_mask].sum() / vuv_count

    # import ipdb; ipdb.set_trace()

    return f0_loss, df0_loss, vuv_loss


def train(model: nn.Module,
          frontend: AbsFrontend,
          spec_aug: SpecAug,
          train_datalodader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: Optimizer,
          scheduler: LRScheduler,
          num_epochs: int,
          log_dir_path: str,
          device='cpu',
          f0data_normalizer=None,
          f0_target_column=2):
    
    f0_criterion = nn.MSELoss(reduction='none')
    df0_critetion = nn.MSELoss(reduction='none')
    vuv_criterion = nn.BCEWithLogitsLoss(reduction='none')

    log_filename = path.join(log_dir_path, 'log.txt')
    with open(log_filename, 'w') as f:
        print(f"Training started at {datetime.datetime.now()}", file=f)

    best_val_loss = 1e+10
    best_param = None

    train_f0_losses = []
    train_df0_losses = []
    train_vuv_losses = []
    val_f0_losses = [] 
    val_df0_losses = []
    val_vuv_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_total_f0_loss = 0
        train_total_df0_loss = 0
        train_total_vuv_loss = 0

        count = 0
        for batch in tqdm.tqdm(train_datalodader):
            # update learning rate (timm scheduler case)
            scheduler.step(epoch * len(train_datalodader) + count)

            optimizer.zero_grad()
            f0_loss, df0_loss, vuv_loss = forward_one_step(
                model, frontend, spec_aug, batch, 
                f0_criterion, df0_critetion, vuv_criterion, 
                device, f0data_normalizer, f0_target_column)

            total_loss = f0_loss + df0_loss + vuv_loss
            total_loss.backward()
            optimizer.step()

            train_total_f0_loss += f0_loss.item()
            train_total_df0_loss += df0_loss.item()
            train_total_vuv_loss += vuv_loss.item()

            count += 1
            if count % 100 == 0:
                log_line = f"Epoch {epoch+1}/{num_epochs}, Iter {count}/{len(train_datalodader)}, Train Loss: F0 {train_total_f0_loss/count:.3f}, DF0 {train_total_df0_loss/count:.3f}, VUV {train_total_vuv_loss/count:.3f}"
                print(log_line)
                    
        # Validation
        model.eval()
        val_total_f0_loss = 0
        val_total_df0_loss = 0
        val_total_vuv_loss = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(val_dataloader):
                f0_loss, df0_loss, vuv_loss = forward_one_step(
                    model, frontend, None, batch,
                    f0_criterion, df0_critetion, vuv_criterion,
                    device, f0data_normalizer, f0_target_column)
                val_total_f0_loss += f0_loss.item()
                val_total_df0_loss += df0_loss.item()
                val_total_vuv_loss += vuv_loss.item()

        # Evaluation metrics
        train_f0_loss = train_total_f0_loss/len(train_datalodader)
        train_df0_loss = train_total_df0_loss/len(train_datalodader)
        train_vuv_loss = train_total_vuv_loss/len(train_datalodader)

        val_f0_loss = val_total_f0_loss/len(val_dataloader)
        val_df0_loss = val_total_df0_loss/len(val_dataloader)
        val_vuv_loss = val_total_vuv_loss/len(val_dataloader)

        learning_rate = optimizer.param_groups[0]['lr']

        log_line = f"Epoch {epoch+1}/{num_epochs}, Train Loss: F0 {train_f0_loss:.3f}, DF0 {train_df0_loss:.3f}, VUV {train_vuv_loss:.3f}, Val Loss: F0 {val_f0_loss:.3f}, DF0 {val_df0_loss:.3f}, VUV {val_vuv_loss:.3f}, LR: {learning_rate:.6f}"
        print(log_line)
        with open(log_filename, 'a') as f:
            print(log_line, file=f)

        train_f0_losses.append(train_f0_loss)
        train_df0_losses.append(train_df0_loss)
        train_vuv_losses.append(train_vuv_loss)
        val_f0_losses.append(val_f0_loss)
        val_df0_losses.append(val_df0_loss)
        val_vuv_losses.append(val_vuv_loss)

        # Update best model
        val_total_loss = val_f0_loss + val_df0_loss + val_vuv_loss
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_param = copy.deepcopy(model.state_dict())
            log_line = "best model updated."
            print(log_line)
            with open(log_filename, 'a') as f:
                print(log_line, file=f)

        # Save the current model
        model_filename = path.join(log_dir_path, f"model_{epoch+1}.pth")
        torch.save(model, model_filename)
        prev_model_filename = path.join(log_dir_path, f"model_{epoch}.pth")
        if path.exists(prev_model_filename):
            os.remove(prev_model_filename)
        
        # update learning rate (torch scheduler case)
        # scheduler.step()

    # Save the best model
    model.load_state_dict(best_param)
    model_filename = path.join(log_dir_path, "model_best.pth")
    torch.save(model, model_filename)

    # Plot the loss curve
    plt.plot(train_f0_losses, label="train f0")
    plt.plot(train_df0_losses, label="train df0")
    plt.plot(train_vuv_losses, label="train vuv")
    plt.plot(val_f0_losses, label="val f0")
    plt.plot(val_df0_losses, label="val df0")
    plt.plot(val_vuv_losses, label="val vuv")

    plt.legend()
    plt.title("loss")
    plt.savefig(path.join(log_dir_path, "loss.png"))
    plt.cla()
    
