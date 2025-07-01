# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Torch trainer class for slide-level downstream tasks."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .utils import (
    slide_level_train_step,
    slide_level_val_step,
)


class TorchTrainer(BaseTrainer):
    """Trainer class for training and evaluating PyTorch models.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    criterion: nn.Module
        The loss criterion used for training.
    metrics: Dict[str, Callable]
        Dictionary of metrics functions to evaluate the model's performance.
    batch_size: int = 16
        The batch size for training and evaluation
    num_epochs : int = 10
        The number of training epochs.
    learning_rate: float = 1.0e-3
        The learning rate for the optimizer.
    weight_decay: float = 0.0
        The weight decay for the optimizer.
    gc_step: Optional[int] = 1
        The number of gradient accumulation steps.
    device : str = "cpu"
        The device to use for training and evaluation.
    optimizer: Callable = Adam
        The optimizer class to use.
    train_step: Callable = slide_level_train_step
        The function for training step.
    val_step: Callable = slide_level_val_step
        The function for validation step.
    collator: Optional[Callable] = None
        The collator function for data preprocessing.
    use_tqdm: bool = True
        Whether to use tqdm progress bar during training.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics: Dict[str, Callable],
        batch_size: int = 16,
        num_epochs: int = 10,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 0.0,
        gc_step: Optional[int] = 1,
        device: str = "cpu",
        optimizer: Callable = AdamW,
        train_step: Callable = slide_level_train_step,
        val_step: Callable = slide_level_val_step,
        collator: Optional[Callable] = None,
        use_tqdm: bool = True,
        rrt_model: Optional[nn.Module] = None,
        rrt_mil: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.rrt_model = rrt_model
        self.rrt_mil = rrt_mil
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics

        self.use_rrt = False
        self.use_rrt_mil = False
        if not "none" in str(type(rrt_mil)).lower():
            print("Use rrt mil model")
            self.use_rrt_mil = True
        elif not "none" in str(type(rrt_model)).lower():
            # We have no rrt module
            self.use_rrt = True
        # type(model) -> ABMIL
        else:
            print("Using no re-embedding")

        self.train_step = train_step
        self.val_step = val_step

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.gc_step = gc_step
        self.collator = collator
        self.device = device
        self.use_tqdm = use_tqdm

        self.train_losses: List[float]
        self.val_losses: List[float]
        self.train_metrics: Dict[str, List[float]]
        self.val_metrics: Dict[str, List[float]]
        self.classifications: Dict[str, float]

        print("Device", self.device)

    def train(
        self,
        train_set: Subset,
        val_set: Subset,
        slide_labels: Optional[Dict[str, float]] = {},
        balance_loader: Optional[bool] = False,
        find_wrongslides: bool = True,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Train the model using the provided training and validation datasets.

        Parameters
        ----------
        train_set: Subset
            The training dataset.
        val_set: Subset
            The validation dataset.

        Returns
        -------
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]
            2 dictionaries containing the training and validation metrics for each epoch.
        """
        # Dataloaders.
        if balance_loader:
            print("Balancing training dataloader.")
            #num_classes = len(set(train_set.dataset.labels))
            use_labels = train_set.dataset.labels[train_set.indices]
            unique, counts = np.unique(use_labels, return_counts=True)
            class_weights = [sum(counts)/c for c in counts]
            sample_w = [class_weights[int(l)] for l in use_labels]
            train_sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

            train_dataloader = DataLoader(
                dataset=train_set,
                shuffle=False,  # Cannot shuffle with a sampler
                batch_size=self.batch_size,
                pin_memory=True,
                collate_fn=self.collator,
                drop_last=True,
                sampler=train_sampler,
            )
            # unique, counts = np.unique(val_set.dataset.labels, return_counts=True)
            # class_weights = [sum(counts)/c for c in counts]
            # sample_w = [class_weights[l] for l in val_set.dataset.labels]
            # val_sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
            # val_dataloader = DataLoader(
            #     dataset=val_set,
            #     shuffle=False,
            #     batch_size=self.batch_size,
            #     pin_memory=True,
            #     collate_fn=self.collator,
            #     drop_last=False,
            #     sampler=val_sampler,
            # )
        else:
            train_dataloader = DataLoader(
                dataset=train_set,
                shuffle=True,
                batch_size=self.batch_size,
                pin_memory=True,
                collate_fn=self.collator,
                drop_last=True,
            )

        val_dataloader = DataLoader(
            dataset=val_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
        )

        # Prepare modules.
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)
        
        if self.use_rrt_mil:
            rrt_model = None
            rrt_mil = self.rrt_mil.to(self.device)
            optimizer = self.optimizer(
                params=rrt_mil.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.use_rrt:
            rrt_mil = None
            rrt_model = self.rrt_model.to(self.device)
            optimizer = self.optimizer(
                params=list(rrt_model.parameters()) +list(model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            rrt_mil = None
            rrt_model = None
            optimizer = self.optimizer(
                params=model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        # Training.
        train_losses, val_losses = [], []
        train_metrics: Dict[str, List[float]] = {
            k: [] for k in self.metrics.keys()
        }
        val_metrics: Dict[str, List[float]] = {
            k: [] for k in self.metrics.keys()
        }

        loop = (
            tqdm(range(self.num_epochs), total=self.num_epochs)
            if self.use_tqdm
            else range(self.num_epochs)
        )

        for ep in loop:
            if ep==0:
                # Add metrics for c-index before training starts.
                # Use val step here.
                train_epoch_loss, train_epoch_logits, train_epoch_labels, slides = self.val_step(
                    model=model,
                    val_dataloader=train_dataloader,
                    criterion=criterion,
                    device=self.device,
                    rrt_model = rrt_model,
                    rrt_mil = rrt_mil,
                )
                val_epoch_loss, val_epoch_logits, val_epoch_labels, slides = self.val_step(
                    model=model,
                    val_dataloader=val_dataloader,
                    criterion=criterion,
                    device=self.device,
                    rrt_model = rrt_model,
                    rrt_mil = rrt_mil,
                )
                # Compute metrics.
                for k, m in self.metrics.items():
                    train_metric = m(train_epoch_labels, train_epoch_logits)
                    val_metric = m(val_epoch_labels, val_epoch_logits)

                    train_metrics[k].append(train_metric)
                    val_metrics[k].append(val_metric)

                train_losses.append(train_epoch_loss)
                val_losses.append(val_epoch_loss)
            # Train step.
            (
                train_epoch_loss,
                train_epoch_logits,
                train_epoch_labels,
            ) = self.train_step(
                model=model,
                train_dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                gc_step=self.gc_step,
                rrt_model = rrt_model,
                rrt_mil = rrt_mil,
            )
            # Inference step.
            val_epoch_loss, val_epoch_logits, val_epoch_labels, slides = self.val_step(
                model=model,
                val_dataloader=val_dataloader,
                criterion=criterion,
                device=self.device,
                rrt_model = rrt_model,
                rrt_mil = rrt_mil,
            )

            # if ep==self.num_epochs-1:
            #     sig = nn.Sigmoid()
            #     val_pred = sig(torch.from_numpy(val_epoch_logits))
            #     val_pred = (val_pred>=0.5).numpy()
            #     wrong_slides = slides[val_pred[:,0] != val_epoch_labels[:,0]]

            # Compute metrics.
            for k, m in self.metrics.items():
                train_metric = m(train_epoch_labels, train_epoch_logits)
                val_metric = m(val_epoch_labels, val_epoch_logits)

                train_metrics[k].append(train_metric)
                val_metrics[k].append(val_metric)

            train_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)
            

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(train_losses)), train_losses/np.max(val_losses), label="Train Cox Loss", color='y')
        # plt.plot(np.arange(len(train_metrics['cindex'])), train_metrics['cindex'], label="Train c-index", color='y', linestyle='--')
        # plt.plot(np.arange(len(val_losses)), val_losses/np.max(val_losses), label="Val Cox Loss, scaled", color='g')
        # plt.plot(np.arange(len(val_metrics['cindex'])), val_metrics['cindex'], label="Val c-index", color='g', linestyle='--')  
        # plt.title("CoxLoss and c-index over epochs")
        # plt.legend()
        # plt.savefig("rrt_singleT_500.png")
        # plt.clf()

        return train_metrics, train_losses, val_metrics, val_losses, val_epoch_labels, val_epoch_logits

    def evaluate(
        self,
        test_set: Subset,
    ) -> Dict[str, float]:
        """Evaluate the model using the provided test dataset.

        Parameters
        ----------
        test_set: Subset
            The test dataset.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the test metrics.
        """
        # Dataloader.
        test_dataloader = DataLoader(
            dataset=test_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
        )

        # Prepare modules.
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)
        if self.use_rrt:
            rrt_model = self.rrt_model.to(self.device)
        else:
            rrt_model = None

        # Inference step.
        _, test_epoch_logits, test_epoch_labels = self.val_step(
            model=model,
            val_dataloader=test_dataloader,
            criterion=criterion,
            device=self.device,
            rrt_model=rrt_model
        )

        # Compute metrics.
        test_metrics = {
            k: m(test_epoch_labels, test_epoch_logits)
            for k, m in self.metrics.items()
        }

        return test_metrics

    def predict(
        self,
        test_set: Subset,
    ) -> Tuple[np.array, np.array]:
        """Make predictions using the provided test dataset.

        Parameters
        ----------
        test_set: Subset
            The test dataset.

        Returns
        --------
        Tuple[np.array, np.array]
            A tuple containing the test labels and logits.
        """
        # Dataloader
        test_dataloader = DataLoader(
            dataset=test_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
        )

        # Prepare modules
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)
        if self.use_rrt:
            rrt_model = self.rrt_model.to(self.device)
        else:
            rrt_model = None

        # Val step
        _, test_epoch_logits, test_epoch_labels = self.val_step(
            model=model,
            val_dataloader=test_dataloader,
            criterion=criterion,
            device=self.device,
            rrt_model=rrt_model
        )

        return test_epoch_labels, test_epoch_logits

    def compute_metrics(
        self, labels: np.array, logits: np.array
    ) -> Dict[str, float]:
        """Compute metrics using the provided labels and logits.

        Parameters
        ----------
        labels: np.ndarray
            The ground truth labels.
        logits: np.ndarray
            The predicted logits.

        Returns:
        Dict[str, float]
            A dictionary containing the computed metrics.
        """
        test_metrics = {
            k: metric(labels, logits) for k, metric in self.metrics.items()
        }
        return test_metrics
