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
                     f0_target_column=2,
                     f0_loss_weight=1.0,
                     df0_loss_weight=1.0,
                     vuv_loss_weight=1.0,):
    wav = batch['wav']
    wav_lengths = batch['wav_lengths']

    wav = wav.to(device)
    wav_lengths = wav_lengths.to(device)

    f0data = batch['f0data']
    f0data_lengths = batch['f0data_lengths']

    vuv_target = (f0data[:, :, f0_target_column] > 0).float()

    if f0data_normalizer is not None:
        f0data = f0data_normalizer.normalize(f0data)

    f0_target = f0data[:, :, f0_target_column]
    df0_target = f0data[:, :, 1]

    inputs, input_lengths = frontend(wav, wav_lengths)
    if spec_aug is not None:
        inputs, input_lengths = spec_aug(inputs, input_lengths)
    
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
    # f0, df0 に関しては，f0data_lengthsの範囲，かつ vuv_target が 1 の範囲が True
    # それ以外は False
    f0_mask = torch.zeros_like(f0_target)
    for i in range(f0_target.shape[0]):
        # f0_mask[i, :pred_lengths[i]] = f0_target[i, :pred_lengths[i]] > 0
        f0_mask[i, :pred_lengths[i]] = vuv_target[i, :pred_lengths[i]] > 0
    f0_mask = f0_mask.bool()
    # vuv に関しては，f0data_lengthsの範囲がTrue
    vuv_mask = torch.zeros_like(vuv_target)
    for i in range(vuv_target.shape[0]):
        vuv_mask[i, :pred_lengths[i]] = 1
    vuv_mask = vuv_mask.bool()

    # f0_count = f0_mask.sum().item()
    # df0_count = f0_mask.sum().item()
    # vuv_count = vuv_mask.sum().item()

    f0_target = f0_target.to(device)
    f0_loss = f0_criterion(f0_pred, f0_target)[f0_mask].mean() * f0_loss_weight
    
    df0_target = df0_target.to(device)
    df0_loss = df0_critetion(df0_pred, df0_target)[f0_mask].mean() * df0_loss_weight
    
    vuv_target = vuv_target.to(device)
    vuv_loss = vuv_criterion(vuv_pred, vuv_target)[vuv_mask].mean() * vuv_loss_weight

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
          log_interval: int,
          eval_interval: int,
          log_dir_path: str,
          device='cpu',
          f0data_normalizer=None,
          f0_target_column=2,):
    
    # f0_criterion = nn.MSELoss(reduction='none')
    # df0_critetion = nn.MSELoss(reduction='none')
    f0_criterion = nn.L1Loss(reduction='none')
    df0_critetion = nn.L1Loss(reduction='none')
    vuv_criterion = nn.BCEWithLogitsLoss(reduction='none')

    log_filename = path.join(log_dir_path, 'log.txt')
    with open(log_filename, 'w') as f:
        print(f"Training started at {datetime.datetime.now()}", file=f)

    best_val_loss = 1e+10
    best_param = None

    train_steps = []
    train_f0_losses = []
    train_df0_losses = []
    train_vuv_losses = []
    
    val_steps = []
    val_f0_losses = [] 
    val_df0_losses = []
    val_vuv_losses = []

    epoch = 0
    step = 0
    steps_per_epoch = len(train_datalodader)
    train_iterator = iter(train_datalodader)

    batch = next(train_iterator)
    train_total_f0_loss = 0
    train_total_df0_loss = 0
    train_total_vuv_loss = 0

    best_model_filename = None

    pbar = tqdm.tqdm(total=len(train_datalodader))

    while epoch < num_epochs:
        # update learning rate (timm scheduler case)
        if scheduler is not None:
            scheduler.step(step)

        model.train()
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

        if (step + 1) % log_interval == 0:
            train_steps.append(step + 1)
            steps_in_epoch = (step + 1) % steps_per_epoch
            train_f0_losses.append(train_total_f0_loss / steps_in_epoch)
            train_df0_losses.append(train_total_df0_loss / steps_in_epoch)
            train_vuv_losses.append(train_total_vuv_loss / steps_in_epoch)

            epoch = (step + 1) / steps_per_epoch
            lr = optimizer.param_groups[0]['lr']
            log_line = f"Epoch {epoch:.2f}, Iter {step + 1}, " \
                + f"Train Loss: F0 {train_total_f0_loss/steps_in_epoch:.3f}, " \
                + f"DF0 {train_total_df0_loss/steps_in_epoch:.3f}, " \
                + f"VUV {train_total_vuv_loss/steps_in_epoch:.3f}, LR: {lr:.6f}"
            print(log_line)

        evaluated = False
        if (step + 1) % eval_interval == 0:
            evaluated = True
            val_total_f0_loss = 0
            val_total_df0_loss = 0
            val_total_vuv_loss = 0

            with torch.no_grad():
                model.eval()
                for batch in tqdm.tqdm(val_dataloader):
                    f0_loss, df0_loss, vuv_loss = forward_one_step(
                        model, frontend, None, batch,
                        f0_criterion, df0_critetion, vuv_criterion,
                        device, f0data_normalizer, f0_target_column)
                    val_total_f0_loss += f0_loss.item()
                    val_total_df0_loss += df0_loss.item()
                    val_total_vuv_loss += vuv_loss.item()

            # Evaluation metrics
            val_steps.append(step + 1)
            val_f0_losses.append(val_total_f0_loss/len(val_dataloader))
            val_df0_losses.append(val_total_df0_loss/len(val_dataloader))
            val_vuv_losses.append(val_total_vuv_loss/len(val_dataloader))

            val_total_loss = val_f0_losses[-1] + val_df0_losses[-1] + val_vuv_losses[-1]

            epoch = (step + 1) / steps_per_epoch
            lr = optimizer.param_groups[0]['lr']
            log_line = f"Epoch {epoch:.2f}, Iter {step + 1}, " \
                + f"Train Loss: F0 {train_total_f0_loss/steps_in_epoch:.3f}, " \
                + f"DF0 {train_total_df0_loss/steps_in_epoch:.3f}, " \
                + f"VUV {train_total_vuv_loss/steps_in_epoch:.3f}, " \
                + f"Eval Loss: F0 {val_total_f0_loss/len(val_dataloader):.3f}, " \
                + f"DF0 {val_total_df0_loss/len(val_dataloader):.3f}, " \
                + f"VUV {val_total_vuv_loss/len(val_dataloader):.3f}, " \
                + f"LR: {lr:.6f}"
            print(log_line)
            with open(log_filename, 'a') as f:
                print(log_line, file=f)

            # Update best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                best_param = copy.deepcopy(model.state_dict())
                log_line = "best model updated."
                print(log_line)
                with open(log_filename, 'a') as f:
                    print(log_line, file=f)

                # Save the current model
                model_filename = path.join(log_dir_path, f"model_{step+1}.pth")
                torch.save(model.state_dict(), model_filename)
                if best_model_filename is not None and path.exists(best_model_filename):
                    os.remove(best_model_filename)
                best_model_filename = model_filename

        step += 1
        batch = next(train_iterator, None)
        pbar.update(1)
        if batch is None:
            train_iterator = iter(train_datalodader)
            batch = next(train_iterator)
            pbar.close()
            pbar = tqdm.tqdm(total=len(train_datalodader))

            train_total_f0_loss = 0
            train_total_df0_loss = 0
            train_total_vuv_loss = 0

            if not evaluated:
                val_total_f0_loss = 0
                val_total_df0_loss = 0
                val_total_vuv_loss = 0

                with torch.no_grad():
                    model.eval()
                    for batch in tqdm.tqdm(val_dataloader):
                        f0_loss, df0_loss, vuv_loss = forward_one_step(
                            model, frontend, None, batch,
                            f0_criterion, df0_critetion, vuv_criterion,
                            device, f0data_normalizer, f0_target_column)
                        val_total_f0_loss += f0_loss.item()
                        val_total_df0_loss += df0_loss.item()
                        val_total_vuv_loss += vuv_loss.item()

                # Evaluation metrics
                val_steps.append(step)
                val_f0_losses.append(val_total_f0_loss/len(val_dataloader))
                val_df0_losses.append(val_total_df0_loss/len(val_dataloader))
                val_vuv_losses.append(val_total_vuv_loss/len(val_dataloader))

                val_total_loss = val_f0_losses[-1] + val_df0_losses[-1] + val_vuv_losses[-1]

                epoch = (step + 1) / steps_per_epoch
                lr = optimizer.param_groups[0]['lr']
                log_line = f"Epoch {epoch:.2f}, Iter {step + 1}, " \
                    + f"Train Loss: F0 {train_total_f0_loss/steps_in_epoch:.3f}, " \
                    + f"DF0 {train_total_df0_loss/steps_in_epoch:.3f}, " \
                    + f"VUV {train_total_vuv_loss/steps_in_epoch:.3f}, " \
                    + f"Eval Loss: F0 {val_total_f0_loss/len(val_dataloader):.3f}, " \
                    + f"DF0 {val_total_df0_loss/len(val_dataloader):.3f}, " \
                    + f"VUV {val_total_vuv_loss/len(val_dataloader):.3f}, " \
                    + f"LR: {lr:.6f}"
                print(log_line)
                with open(log_filename, 'a') as f:
                    print(log_line, file=f)

                # Update best model
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    best_param = copy.deepcopy(model.state_dict())
                    log_line = "best model updated."
                    print(log_line)
                    with open(log_filename, 'a') as f:
                        print(log_line, file=f)

                    # Save the current model
                    model_filename = path.join(log_dir_path, f"model_{step+1}.pth")
                    torch.save(model.state_dict(), model_filename)
                    if best_model_filename is not None and path.exists(best_model_filename):
                        os.remove(best_model_filename)
                    best_model_filename = model_filename

    # Save the best model
    model.load_state_dict(best_param)
    model_filename = path.join(log_dir_path, "model_best.pth")
    torch.save(model.state_dict(), model_filename)

    # Plot the loss curve
    plt.plot(train_steps, train_f0_losses, label="train f0")
    plt.plot(train_steps, train_df0_losses, label="train df0")
    plt.plot(train_steps, train_vuv_losses, label="train vuv")
    plt.plot(val_steps, val_f0_losses, label="val f0")
    plt.plot(val_steps, val_df0_losses, label="val df0")
    plt.plot(val_steps, val_vuv_losses, label="val vuv")

    plt.legend()
    plt.title("loss")
    plt.savefig(path.join(log_dir_path, "loss.png"))
    plt.cla()
    
