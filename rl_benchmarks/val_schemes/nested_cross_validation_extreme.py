# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Nested cross-validation class."""

import copy
from typing import Dict, List, Optional

import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset, ConcatDataset
import pandas as pd

from .utils import (
    generate_permutations,
    split_cv,
    update_trainer_cfg,
)


class NestedCrossValidationExtreme:
    r"""Class implementing Nested cross-validation. Nested cross-validation
    involves two levels of cross-validation, an outer and inner cross-validation.
    Within the training outer folds, an inner cross-validation is performed for
    hyperparameters tuning and model selection. The best model configuration is
    chosen based on the average performance across the inner folds. This
    selected model is then evaluated on the corresponding validation outer fold,
    which was not used during model selection. The performance metrics obtained
    from each validation outer fold are averaged to estimate the model
    generalization performance. This eliminates the bias introduced by standard
    cross-validation procedure as the test data in each iteration of the outer
    cross-validation has not been used to optimize the performance of the model
    in any way, and may therefore provide a more reliable criterion for choosing
    the best model. In our study, we performed 5x5 nested cross-validation with
    no repeats (five inner and five outer splits). During nested-CV, we test
    different values of the initial learning rate and weight decay, namely
    $\{0.001,0.0001\}$ for learning rate and $\{0, 0.0001\}$ for weight decay,
    respectively. The optimal number of epochs is determined within each oute
    split through the 5-fold inner CV based on the validation metric (AUC as
    a default).


    Parameters
    ----------
    trainer_cfg: DictConfig
        Trainer configuration. Examples are available in
        ``rl_benchmarks/conf/slide_level_task/cross_validation/task/*.yaml``
        configurations files. ``trainer_cfg`` aims to instantiate the following
        trainer function: ``_target_: rl_benchmarks.trainers.TorchTrainer``.
    grid_search_params: Optional[Dict[str, List[float]]] = None
        Grid search parameters. Example:
        ``{'learning_rate': [1e-4, 1e-5], 'weight_decay': [0.0, 1e-6]}``.
        Best configuration is selected based on the inner cross-validations.
        If ``None``, no hyperparameters tuning is performed.
    n_repeats_outer: int = 1
        Number of repetitions of the outer cross-validation.
    n_splits_outer: int = 5
        Number of outer splits.
    n_repeats_inner: int = 1
        Number of repetitions of the inner cross-validations.
    n_splits_inner: int = 5
        Number of inner splits.
    stratified: bool = True
        Whether to stratify the splits.
    split_mode: str = "patient_split"
        Which mode of stratification. Other modes are ``'random_split'`` and
        ``'center_split'``. Default is ``'patient_split'`` to avoid any data
        leaking between folds.
    """

    def __init__(
        self,
        trainer_cfg: DictConfig,
        grid_search_params: Optional[Dict[str, List[float]]] = None,
        n_repeats_outer: int = 1,
        n_splits_outer: int = 5,
        n_repeats_inner: int = 1,
        n_splits_inner: int = 5,
        stratified: bool = True,
        split_mode: str = "patient_split",
    ):
        self.n_repeats_outer = n_repeats_outer
        self.n_repeats_inner = n_repeats_inner
        self.n_splits_inner = n_splits_inner
        self.stratified = stratified
        self.split_mode = split_mode

        self.trainer_cfg = trainer_cfg
        self.grid_search_params = OmegaConf.create(grid_search_params)

    def run(self, test_dataset: Dataset, train_dataset: List[Dataset], epochs: int) -> List[Dict[str, Dict[str, List[float]]]]:
        """Main function ot run the whole nested cross-validation.
        Parameters
        ----------
        dataset: Dataset
            Dataset to perform nested CV on.

        Returns
        -------
        ``cv_train_val_metrics``, ``cv_test_metrics``: Dict[str, Dict[str, List[float]]]
            Outer-folds metrics. Each dictionnary is of type
            ``cv_train_val_metrics['metric_name'][f'repeat_{r}_split_{s}'] = metric_values```
            where ``'metric_name'`` is either "auc" or "cindex" and values and
            ``metric_values`` is the list of the metric values for all epochs.
            The average training metrics are computed as follows on the outer folds:
            >>> for k, v in cv_train_val_metrics.items():
            >>>     logger.info(
            >>>         f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.3f} "
            >>>         f"({np.std([_v[-1] for _v in v.values()]):.3f})"
            >>>     )
            For each training outer folds, the last epoch is considered as the
            best epoch, hence ``[-1]``. Indeed the ``n_outer_folds`` optimal
            models, each derived with optimal sets of parameters and optimal
            number of epochs during corresponding inner CV, are re-trained for
            exactly ``optimal_epoch`` epochs, on the outer training folds.
            Those models are also evaluated at each epoch on the outer test fold,
            till ``optimal_epoch`` epochs. The mean of the last element of the
            metric values averages the test metrics for each outer test fold,
            hence giving the average nested-cv test metric.
       
        test_dataset: testing/validation dataset
        train_data: training dataset
        """
        # Start logging.
        #logger.info("Running nested cross-validation.")

        # Automatic setting of in_features and out_features
        # If test_dataset is not a list this works... FIX
        if type(test_dataset)!=list:
            n_features, n_labels = (test_dataset.n_features, test_dataset.n_labels)
            slide_labels = list(test_dataset.slide_path)
        else:
            n_features, n_labels = (train_dataset.n_features, train_dataset.n_labels)
            slide_labels = list(train_dataset.slide_path)

        # Update the main configuration with ``n_features`` and
        # ``n_labels`` for correctly instantiating the MIL
        # aggregation model (``in_features`` and ``out_features``,
        # respectively).
        # Set number of epochs
        use_cfg = {"num_epochs": epochs}

        # Initialize the trainer with the optimal configuration.
        trainer_cfg = copy.deepcopy(self.trainer_cfg)
        trainer_cfg.update(use_cfg)
        update_trainer_cfg(trainer_cfg, n_features, n_labels)

        # Log trainer info.
        logger.info("---- Trainer info ----")
        logger.info(trainer_cfg)

        # Enter the outer cross-validation loop.
        cv_train_val_metrics: Dict[str, List] = {}
        cv_test_metrics: Dict[str, List] = {}

        for r_outer in range(self.n_repeats_outer):
            # Repeating the experiment(s)
            # In this "extreme case" splits will be the same for each repeat
            # Splits based on dataset: dataset is test/val, add_data is train

            # We use the test_dataset as test and train_dataset as train
            logger.info(
                f"Outer cross-validation: repeat_{r_outer+1}"
            )
            # Get ``train_val_dataset`` and ``test_dataset``
            # Inner cross-validation, 
            # Possibly gridsearching and model selection will take place on the ``train_val_dataset``
            if type(test_dataset)==list:
                for d in test_dataset:
                    d.shuffle = False
                test_dataset = ConcatDataset(test_dataset)
            else:
                test_dataset.shuffle = False
            
            #print("Test cohort(s):", test_cohort)
            if type(train_dataset)==list:
                for d in train_dataset:
                    d.shuffle = True
                train_val_dataset = ConcatDataset(train_dataset)
            else:
                train_dataset.shuffle = True
                train_val_dataset = train_dataset


            # No grid-search is performed.
            
            if trainer_cfg.model.in_features == 765:
                print("Setting in features in linear model manually")
                trainer_cfg.model.in_features = 768
            elif trainer_cfg.model.in_features == 2557:
                print("Setting in features in linear model manually")
                trainer_cfg.model.in_features = 2560
            trainer = instantiate(trainer_cfg)

            balance_loader = True
            # Re train with the optimal configuration on the whole
            # inner folds, and assess performance on the outer fold.
            train_val_metrics, train_val_loss, test_metrics, test_loss, _test_labels, _test_logits = trainer.train(
                train_val_dataset, test_dataset, None, balance_loader=balance_loader
            )

            # Store outer-folds metrics.
            for k, v in train_val_metrics.items():
                if k in cv_train_val_metrics:
                    cv_train_val_metrics[k][
                        f"repeat_{r_outer+1}"
                    ] = v
                else:
                    cv_train_val_metrics[k] = {
                        f"repeat_{r_outer+1}": v
                    }

            for k, v in test_metrics.items():
                if k in cv_test_metrics:
                    cv_test_metrics[k][
                        f"repeat_{r_outer+1}"
                    ] = v
                else:
                    cv_test_metrics[k] = {
                        f"repeat_{r_outer+1}": v
                    }


            # Log outer split metrics.
            logger.info("---- Outer splits metrics ----")
            logger.info(f"Repeat {r_outer+1}")
            logger.info("Training folds metrics:")
            for k, v in train_val_metrics.items():
                logger.info(f"   {k}: {v[-1]:.3f}")
            logger.info("Test fold metrics:")
            #logger.info(f"  for test cohort {test_cohort}")
            for k, v in test_metrics.items():
                logger.info(f"   {k}: {v[-1]:.4f}")

        # Calculate metrics over all outer
        # Log outer cross-validation metrics.
        #logger.info("")
        logger.success("---- FINAL RESULTS ----")
        # logger.success("---- Outer cross-validation train metrics ----")
        # for k, v in cv_train_val_metrics.items():
        #     # The last epoch is the best epoch (as training ended at the
        #     # best epoch identified during inner CV).
        #     logger.success(
        #         f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.4f} "
        #         f"({np.std([_v[-1] for _v in v.values()]):.3f})"
        #     )
        logger.success("---- Outer cross-validation test metrics ----")
        for k, v in cv_test_metrics.items():
            logger.success(
                f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.4f} "
                f"({np.std([_v[-1] for _v in v.values()]):.3f})"
            )

        #logger.success("----- Slide classifications -----")
        #print(f"# slides considered: {len(slide_labels_all.keys())}", flush=True)
        print("Consider c-index results")
        return cv_train_val_metrics["cindex"], cv_test_metrics["cindex"]