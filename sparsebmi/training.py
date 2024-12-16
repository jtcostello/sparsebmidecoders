import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .pruning import ModelPruner, get_percent_pruned
from .utils import LinearScheduler, LinearLRScheduler


def train_prune_model(model, loader_train, loader_val, optimizer, lr_scheduler, num_iters, loss_fn,
                      check_every=100, verbose=True, is_rnn=False, pruner=None, prune_scheduler=None,
                      device='cpu', wandb=None):
    """
    Trains a model for num_iters iterations, pruning at each step.
    Pruner can be left as None if you don't want to prune.
    """
    mses = []
    corrs = []
    drop_fracs = []
    iters = []

    # Training loop
    iter = 0
    done = False
    model.train()
    while not done:
        for x, y in loader_train:
            # x = batch['chans']          # [batch_size x num_chans x hist_bins]
            # y = batch['states']         # [batch_size x num_fings]
            x, y = x.to(device=device), y.to(device=device)

            # forward pass
            optimizer.zero_grad()
            if is_rnn:
                yhat, _ = model(x, h=None)
            else:
                yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # prune
            if pruner is not None:
                drop_frac = prune_scheduler.get_val()
                pruner.prune(drop_frac)
                prune_scheduler.step()
            else:
                drop_frac = 0

            iter += 1
            if iter % check_every == 0:
                # evaluate model
                model.eval()
                corr, mse = eval_model(model, loader_val, device)
                model.train()
                if verbose:
                    print(f"Fraction of weights removed: {drop_frac:.2f}")
                    print(f'Iteration {iter}, test corr = {corr}, mse = {mse}')

                mses.append(mse)
                drop_fracs.append(drop_frac)
                corrs.append(np.mean(corr))
                iters.append(iter)

                if wandb is not None:
                    wandb.log({'test_corr': np.mean(corr), 'test_mse': mse, 'drop_frac': drop_frac, 'iter': iter})

            if iter >= num_iters:
                done = True
                break

    return iters, corrs, mses, drop_fracs


def train_prune_rewind_model(model, prune_steps, loader_train, loader_val, loss_fn, cfg,
                             device='cpu', verbose_train=False):
    """
    Iteratively trains and prunes a model, and then rewinds the weights to initial values buts with new pruning mask
    for the next round of training/pruning.
    For example, if prune_steps = [0, 95, 99], we train from 0 to 95% sparsity, rewind the weights, and then train
    from 0 to 99% sparsity.

    The input model should not have any pruning applied to it yet.
    """
    iters = []
    prune_vals = []
    models = []
    masks_inited = False
    total_iters = 0

    for i in range(len(prune_steps)-1):
        prune_start = prune_steps[i]
        prune_end = prune_steps[i+1]
        print(f"\n\nstarting pruning from {prune_start}% to {prune_end}%...")

        # init pruner and optimizer
        pruner = ModelPruner(model, model.get_pruning_params(), prev_prune_pct=prune_start, finalize_every=cfg.finalize_every)
        prune_scheduler = LinearScheduler(prune_start, prune_end, cfg.iters_before_drop, cfg.iters_to_max_drop)

        optimizer = optim.Adam(model.parameters(), lr=cfg.start_lr, weight_decay=cfg.weight_decay)
        lr_scheduler = LinearLRScheduler(optimizer, start_lr=cfg.start_lr, end_lr=cfg.end_lr, num_steps=cfg.num_iters)

        # store a copy of the initial model weights
        if not masks_inited:
            pruner.prune(0)  # initialize the pruning masks
            masks_inited = True
            untrained_state_dict = copy.deepcopy(model.state_dict())

        # train and prune
        # (note that prune masks may be finalized during training to save memory)
        model.train()
        _, corrs, mses, _ = train_prune_model(model, loader_train, loader_val, optimizer, lr_scheduler,
                                              cfg.num_iters, loss_fn, check_every=cfg.check_every,
                                              verbose=verbose_train, is_rnn=cfg.is_rnn, pruner=pruner,
                                              prune_scheduler=prune_scheduler, device=device)
        total_iters += cfg.num_iters

        model.eval()
        corr, mse = eval_model(model, loader_val, device)
        print(f'validation corr = {corr}, mse = {mse}')
        model.train()

        # save pruned model (finalize first to remove prune mask history)
        pruner.finalize()
        iters.append(total_iters)
        prune_vals.append(prune_end)
        models.append(copy.deepcopy(model.to('cpu')))

        # stop if we're at the last pruning step
        if i == len(prune_steps)-2:
            break

        # re-prune to add back the final mask (this shouldn't actually change any weight values)
        pruner.prune(prune_end)

        # create a new model with the initial weights but the new pruning masks
        pruning_masks = {name: param.clone() for name, param in model.state_dict().items() if 'mask' in name}
        model.load_state_dict(untrained_state_dict)
        # apply the learned weight masks
        for name, mask in pruning_masks.items():
            if name in model.state_dict():
                model.state_dict()[name].data.copy_(mask.data)
        model.to(device)

    print(f"\n\n Final percent pruned = {get_percent_pruned(model, verbose=False):.2f}")

    return iters, prune_vals, models


def train_chanselect_model(model, loader_train, loader_val, optimizer, lr_scheduler, num_iters, loss_fn,
                           check_every=100, verbose=True, is_rnn=False, prune_scheduler=None, device='cpu',
                           wandb=None, is_multigate=False, num_gates=1):
    """
    Trains a model for num_iters iterations, dropping channels at each step.
    `prune_scheduler` can be left as None if you don't want to drop channels.
    """
    mses = []
    corrs = []
    drop_fracs = []
    iters = []

    # Training loop
    iter = 0
    done = False
    model.train()
    while not done:
        for x, y in loader_train:
            # x = batch['chans']          # [batch_size x num_chans x hist_bins]
            # y = batch['states']         # [batch_size x num_fings]
            x, y = x.to(device=device), y.to(device=device)

            # forward pass
            optimizer.zero_grad()
            if is_multigate:
                # loop over each gate, copying the batch
                yhat_list = []
                for i in range(num_gates):
                    this_x = x.clone()
                    yhat = model(this_x, day_idx=i)
                    if is_rnn:
                        yhat = yhat[0]
                    yhat_list.append(yhat)
                yhat = torch.vstack(yhat_list)
                y_all = torch.vstack([y for _ in range(num_gates)])
                loss = loss_fn(yhat, y_all)
            else:
                if is_rnn:
                    yhat, _ = model(x, h=None)
                else:
                    yhat = model(x)
                loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # drop channels
            if prune_scheduler is not None:
                drop_frac = prune_scheduler.get_val()   # could be a list if multi-gate
                model.freeze_channel_fraction(drop_frac, verbose=False)
                prune_scheduler.step()
            else:
                drop_frac = 0

            iter += 1
            if iter % check_every == 0:
                # evaluate model
                model.eval()
                corr, mse = eval_model(model, loader_val, device)
                model.train()
                if verbose:
                    if is_multigate:
                        print(f"Fraction of channels removed: {[f'{f:.2f}' for f in drop_frac]}")
                    else:
                        print(f"Fraction of channels removed: {drop_frac:.2f}")
                    print(f'Iteration {iter}, test corr = {corr}, mse = {mse}')

                mses.append(mse)
                drop_fracs.append(drop_frac)
                corrs.append(np.mean(corr))
                iters.append(iter)

                if wandb is not None:
                    wandb.log({'test_corr': np.mean(corr), 'test_mse': mse, 'drop_frac': drop_frac, 'iter': iter})

            if iter >= num_iters:
                done = True
                break

    return iters, corrs, mses, drop_fracs


def train_chanselect_rewind_model(model, prune_steps, loader_train, loader_val, cfg,
                                  device='cpu', verbose_train=False):
    """
    Iteratively trains and prunes a model, and then rewinds the weights to initial values buts with new channel mask
    for the next round of training/pruning.
    For example, if prune_steps = [0, 95, 99], we train from 0 to 95% sparsity, rewind the weights, and then train
    from 0 to 99% sparsity.

    The input model should not have any pruning applied to it yet.
    """

    # store a copy of the initial model weights
    untrained_state_dict = copy.deepcopy(model.state_dict())

    iters = []
    prune_vals = []
    models = []
    total_iters = 0

    for i in range(len(prune_steps)-1):
        prune_start = prune_steps[i]
        prune_end = prune_steps[i+1]
        print(f"\n\nstarting pruning from {100*prune_start}% to {100*prune_end}%...")

        # init pruner and optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg.start_lr, weight_decay=cfg.weight_decay)
        lr_scheduler = LinearLRScheduler(optimizer, start_lr=cfg.start_lr, end_lr=cfg.end_lr, num_steps=cfg.num_iters)
        prune_scheduler = LinearScheduler(prune_start, prune_end, cfg.iters_before_drop, cfg.iters_to_max_drop)

        # train & drop channels
        model.train()
        _, corrs, mses, _ = train_chanselect_model(model, loader_train, loader_val, optimizer, lr_scheduler,
                                                   cfg.num_iters, model.compute_loss, check_every=cfg.check_every,
                                                   verbose=verbose_train, is_rnn=cfg.is_rnn,
                                                   prune_scheduler=prune_scheduler, device=device)
        total_iters += cfg.num_iters

        model.eval()
        corr, mse = eval_model(model, loader_val, device)
        print(f'validation corr = {corr}, mse = {mse}')
        model.train()

        # save pruned model
        iters.append(total_iters)
        prune_vals.append(prune_end)
        models.append(copy.deepcopy(model.to('cpu')))

        # stop if we're at the last pruning step
        if i == len(prune_steps)-2:
            break

        # create a new model with the initial weights but the new pruning masks
        orig_channel_freeze_mask = model.channel_freeze_mask.clone()
        model.load_state_dict(untrained_state_dict)
        model.channel_freeze_mask.data.copy_(orig_channel_freeze_mask)

        # freeze the current amount to not change anything, but add back in the gradient hook to prevent updates
        model.freeze_channel_fraction(prune_end, verbose=False)
        model.to(device)

    print(f"\n\n Final percent pruned = {get_percent_pruned(model, verbose=False):.2f}")
    return iters, prune_vals, models


def train_prune_chanselect_model(model, loader_train, loader_val, optimizer, lr_scheduler, num_iters, loss_fn,
                                 check_every=100, verbose=True, is_rnn=False, pruner=None,
                                 prune_scheduler_channels=None, prune_scheduler_prune=None,
                                 device='cpu', wandb=None, is_multigate=False, num_gates=1):
    """
    Trains a model for num_iters iterations, dropping channels AND pruning weights at each step.
    `prune_scheduler` can be left as None if you don't want to drop channels.
    """
    mses = []
    corrs = []
    drop_fracs = []
    iters = []

    # Training loop
    iter = 0
    done = False
    model.train()
    while not done:
        for x, y in loader_train:
            # x = batch['chans']          # [batch_size x num_chans x hist_bins]
            # y = batch['states']         # [batch_size x num_fings]
            x, y = x.to(device=device), y.to(device=device)

            # forward pass
            optimizer.zero_grad()
            if is_multigate:
                raise NotImplementedError("Multigate not implemented for prune & chanselect")
                # loop over each gate, copying the batch
                yhat_list = []
                for i in range(num_gates):
                    this_x = x.clone()
                    yhat = model(this_x, day_idx=i)
                    if is_rnn:
                        yhat = yhat[0]
                    yhat_list.append(yhat)
                yhat = torch.vstack(yhat_list)
                y_all = torch.vstack([y for _ in range(num_gates)])
                loss = loss_fn(yhat, y_all)
            else:
                if is_rnn:
                    yhat, _ = model(x, h=None)
                else:
                    yhat = model(x)
                loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # drop channels
            if prune_scheduler_channels is not None:
                drop_frac_chans = prune_scheduler_channels.get_val()   # could be a list if multi-gate
                model.freeze_channel_fraction(drop_frac_chans, verbose=False)
                prune_scheduler_channels.step()
            else:
                drop_frac_chans = 0

            # prune weights
            if pruner is not None:
                drop_frac_weights = prune_scheduler_prune.get_val()
                pruner.prune(drop_frac_weights)
                prune_scheduler_prune.step()
            else:
                drop_frac_weights = 0

            iter += 1
            if iter % check_every == 0:
                # evaluate model
                model.eval()
                corr, mse = eval_model(model, loader_val, device)
                model.train()
                if verbose:
                    if is_multigate:
                        print(f"Fraction of channels removed: {[f'{f:.2f}' for f in drop_frac_chans]}")
                    else:
                        print(f"Fraction of channels removed: {drop_frac_chans:.2f}")
                    print(f'Iteration {iter}, test corr = {corr}, mse = {mse}')

                mses.append(mse)
                drop_fracs.append(drop_frac_chans)
                corrs.append(np.mean(corr))
                iters.append(iter)

                if wandb is not None:
                    wandb.log({'test_corr': np.mean(corr), 'test_mse': mse, 'drop_frac': drop_frac_chans, 'iter': iter})

            if iter >= num_iters:
                done = True
                break

    return iters, corrs, mses, drop_fracs


def run_model_forward(model, loader, device='cpu', return_x=False, day_idx=None):
    """
    Runs the model using the provided dataloader
    """
    with torch.no_grad():
        all_y = []
        all_yhat = []
        all_x = []
        for x, y in loader:
            # x = batch['chans'].to(device=device)    # shape (num_samps, num_chans, seq_len)
            # y = batch['states'].to(device=device)   # shape (num_samps, num_outputs)
            x, y = x.to(device=device), y.to(device=device)

            if day_idx is not None:
                yhat = model(x, day_idx=day_idx)
            else:
                yhat = model.forward(x)

            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]

            all_y.append(y)
            all_yhat.append(yhat)
            all_x.append(x)

        all_y = torch.cat(all_y, dim=0)
        all_yhat = torch.cat(all_yhat, dim=0)
        all_x = torch.cat(all_x, dim=0)

    if return_x:
        return all_x, all_y, all_yhat
    else:
        return all_y, all_yhat


def corrcoef(x, y):
    num_feats = x.shape[1]
    return np.diag(np.corrcoef(x, y, rowvar=False)[:num_feats, num_feats:])


def eval_model(model, loader, device, day_idx=None):
    """ Evaluate a model on a given dataset """
    y, yhat = run_model_forward(model, loader, device=device, day_idx=day_idx)
    corr = corrcoef(y.cpu().detach(), yhat.cpu().detach())
    mse = nn.MSELoss()(yhat, y).cpu().item()
    return corr, mse


def extract_loader_data(loader):
    """ Extracts all data from a dataloader """
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
    return torch.vstack(all_x), torch.vstack(all_y)
