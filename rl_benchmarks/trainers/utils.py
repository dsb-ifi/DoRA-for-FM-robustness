# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training and inference step for slide-level experiments."""

from typing import Optional, Tuple, Dict, List

import numpy as np
import torch


def slide_level_train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    gc_step: Optional[int] = 1,
    rrt_model: Optional[torch.nn.Module] = None,
    rrt_mil: Optional[torch.nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training step for slide-level experiments. This will serve as the
    ``train_step`` in ``TorchTrainer``class.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    train_dataloader: torch.utils.data.DataLoader
        Training data loader.
    criterion: nn.Module
        The loss criterion used for training.
    optimizer: Callable = Adam
        The optimizer class to use.
    device : str = "cpu"
        The device to use for training and evaluation.
    gc_step: Optional[int] = 1
        The number of gradient accumulation steps.
    """
    model.train()
    use_rrt = False
    shred = True # ABMIL skips metadata cols if rrt model not present
    use_rrt_mil = False
    if not "none" in str(type(rrt_mil)).lower():
        use_rrt_mil = True
        rrt_mil.train()
    elif not "none" in str(type(rrt_model)).lower():
        # We have a rrt module
        use_rrt = True
        shred = False
        rrt_model.train()
    
    _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

    for i, batch in enumerate(train_dataloader):
        # Get data.
        features, mask, labels, slide = batch

        # Put on device.
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device).float()

        # Compute logits and loss.
        if use_rrt_mil:
            logits = rrt_mil(features)
        elif use_rrt:
            features = rrt_model(features)
            logits = model(features, mask, shred)
        else:
            logits = model(features, mask, shred)
        loss = criterion(logits, labels)

        # Optional: Gradient accumulation.
        loss = loss / gc_step
        loss.backward()

        if ((i + 1) % gc_step == 0) or ((i + 1) == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        # Stack logits & labels to compute epoch metrics.
        _epoch_loss.append(loss.detach().cpu().numpy())
        _epoch_logits.append(logits.detach())
        _epoch_labels.append(labels.detach())

    #print("MASK pos #", mask.sum()) # 0

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels


def slide_level_val_step(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    rrt_model: Optional[torch.nn.Module] = None,
    rrt_mil: Optional[torch.nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference step for slide-level experiments. This will serve as the
    ``val_step`` in ``TorchTrainer``class.

        Parameters
        ----------
        model: nn.Module
            The PyTorch model to be trained.
        val_dataloader: torch.utils.data.DataLoader
            Inference data loader.
        criterion: nn.Module
            The loss criterion used for training.
        device : str = "cpu"
            The device to use for training and evaluation.
    """
    model.eval()
    use_rrt = False
    shred = True # ABMIL skips metadata cols if rrt model not present
    use_rrt_mil = False
    if not "none" in str(type(rrt_mil)).lower():
        use_rrt_mil = True
        rrt_mil.eval()
    elif not "none" in str(type(rrt_model)).lower():
        # We have a rrt module
        use_rrt = True
        shred = False
        rrt_model.eval()

    with torch.no_grad():
        _epoch_loss, _epoch_logits, _epoch_labels = [], [], []
        _slides = []

        for batch in val_dataloader:
            # Get data.
            features, mask, labels, slide = batch

            # Put on device.
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device).float()

            # Compute logits and loss.
            if use_rrt_mil:
                logits = rrt_mil(features)
            elif use_rrt:
                features = rrt_model(features)
                logits = model(features, mask, shred)
            else:
                logits = model(features, mask, shred)
            loss = criterion(logits, labels)

            _epoch_loss.append(loss.detach().cpu().numpy())
            _epoch_logits.append(logits.detach())
            _epoch_labels.append(labels.detach())
            _slides.append(slide)
        #print("MASK val pos #", mask.sum()) #0

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()
    _slides = np.concatenate(_slides)

    return _epoch_loss, _epoch_logits, _epoch_labels, _slides
