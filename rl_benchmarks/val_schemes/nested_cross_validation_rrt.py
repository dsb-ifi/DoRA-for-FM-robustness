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


class NestedCrossValidationRRT:
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
        self.n_splits_outer = n_splits_outer
        self.n_repeats_inner = n_repeats_inner
        self.n_splits_inner = n_splits_inner
        self.stratified = stratified
        self.split_mode = split_mode

        self.trainer_cfg = trainer_cfg
        self.grid_search_params = OmegaConf.create(grid_search_params)

    def run(self, dataset: Dataset, add_data: List[Dataset], mix_datasets: bool=True, single_d: str="test") -> List[Dict[str, Dict[str, List[float]]]]:
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
        """
        # Start logging.
        logger.info("Running nested cross-validation.")

        # Automatic setting of in_features and out_features
        n_features, n_labels = (dataset.n_features, dataset.n_labels)

        slide_labels = list(dataset.slide_path)

        if add_data:
            logger.info(f"Running with {len(add_data)} added training datasets!")
            for i in range(len(add_data)):
                ni_features, ni_labels = (add_data[i].n_features, add_data[i].n_labels)
                assert (ni_features==n_features) and (ni_labels==n_labels), "All datasets must have features of the same dimensions and labels"
                slide_labels.append(list(add_data[i].slide_path))

        slide_labels = {str(i): 0 for i in slide_labels}

        #print(f"n_features: {n_features}, n_labels: {n_labels}")
        
        # Update the main configuration with ``n_features`` and
        # ``n_labels`` for correctly instantiating the MIL
        # aggregation model (``in_features`` and ``out_features``,
        # respectively).
        update_trainer_cfg(self.trainer_cfg, n_features, n_labels)

        # Log trainer info.
        logger.info("---- Trainer info ----")
        logger.info(self.trainer_cfg)

        # Enter the outer cross-validation loop.
        cv_train_val_metrics: Dict[str, List] = {}
        cv_test_metrics: Dict[str, List] = {}
        slide_labels_all: Dict[str, List] = {}
        r_test_metrics: Dict[str, List] = {}
        #wrong_slides = []

        for r_outer in range(self.n_repeats_outer):
            # Split the dataset into ``self.n_splits_outer`` outer folds using
            # the ``split_cv`` function. Splits are different for each repeat
            # depending on the ``random_state`` parameter.
            if add_data:
                all_data = []
                for i in range(len(add_data)+1):
                    if i==0:
                        all_data.append(dataset)
                    else:
                        all_data.append(add_data[i-1])

                if not mix_datasets:
                    logger.info(f"Running cross-dataset cross-evaluation.") 
                    logger.info(f"Overriding the number of outer splits from {self.n_splits_outer} to the number of datasets we have: {len(all_data)}")
                    print(f"Overriding # outer splits to {len(all_data)}")
                    self.n_splits_outer = 1 + len(add_data)
            else:
                test_logits = np.array([])
                test_labels = np.array([])

            if mix_datasets:
                for d in add_data:
                    dataset.merge_datasets(d)

            splits_outer = split_cv(
                dataset=dataset,
                n_splits=self.n_splits_outer,
                split_mode=self.split_mode,
                random_state=r_outer,
            )

            # Now iterate on the outer folds.
            for s_outer, (train_val_indices, test_indices) in enumerate(
                splits_outer
            ):
                logger.info(
                    f"Outer cross-validation: repeat_{r_outer+1}_split_{s_outer+1}"
                )
                # Split outer dataset into ``train_val_dataset`` and
                # ``test_dataset``. Inner cross-validation, gridsearching and
                # model selection will take place on the ``train_val_dataset``
                # where best model's generalization error will be evaluated
                # on the ``test_dataset``, unseen during inner CV.
                if add_data and not mix_datasets:
                    if single_d.lower()=="train":
                        # Train on single cohort at a time
                        train_val_dataset = all_data[s_outer]
                        test_dataset = all_data.copy()
                        test_dataset.pop(s_outer)
                        test_dataset = ConcatDataset(test_dataset)
                        test_cohort = str([d.cohort for d in test_dataset.datasets])
                        #logger.info(f"Test data cohort is {test_cohort}")
                    else:
                        test_dataset = all_data[s_outer]
                        test_cohort = test_dataset.cohort
                        #logger.info(f"Test data cohort is {test_cohort}")
                        train_val_dataset = all_data.copy()
                        train_val_dataset.pop(s_outer)
                        train_val_dataset = ConcatDataset(train_val_dataset)

                else:
                    train_val_dataset = Subset(dataset, indices=train_val_indices)
                    test_dataset = Subset(dataset, indices=test_indices)

                df_test=[]
                for i in range(len(test_dataset)):
                    df_test.append(test_dataset[i][2])

                # Hyperparameter tuning (if applicable).
                print("gridsearch params: ", self.grid_search_params.keys(), self.grid_search_params.values())
                grid_values_len = sum([len(v) for v in self.grid_search_params.values()])
                if grid_values_len > 2:#self.grid_search_params: #False: #
                    logger.info("Running hyperparameters tunning.")

                    # Create all possible config files given the dictionnary
                    # ``self.grid_search_params``.
                    permutations_dicts = generate_permutations(
                        OmegaConf.to_container(self.grid_search_params)
                    )

                    # Log hyperparameter tuning info.
                    logger.info("---- Hyperparameter tuning info ----")
                    logger.info(
                        f"Number of possible configurations: {len(permutations_dicts)}"
                    )
                    
                    # Iterate over each possible set of parameters, ie config
                    # files.
                    list_cfgs, list_val_metrics = [], []
                    for i, sub_cfg in enumerate(permutations_dicts):
                        # Use main config file.
                        trainer_cfg = copy.deepcopy(self.trainer_cfg)
                        # Update the main configuration file with the current
                        # set of parameters we would like to perform CV on.
                        trainer_cfg.update(sub_cfg)
                        # Update this configuration with ``n_features`` and
                        # ``n_labels`` for correctly instantiating the MIL
                        # aggregation model (``in_features`` and ``out_features``,
                        # respectively).
                        update_trainer_cfg(trainer_cfg, n_features, n_labels)
                        # Log current configuration.
                        logger.info(f"Grid search #{i} / {len(permutations_dicts)}")
                        logger.info(f"Current config: {sub_cfg}")
                        logger.info(f"(At outer repeat number {r_outer+1}, outer split {s_outer+1})")

                        # Enter inner cross-validation.
                        cv_train_metrics: Dict[str, List] = {}
                        cv_val_metrics: Dict[str, List] = {}
                        for r_inner in range(self.n_repeats_inner):
                            # As done before, split the dataset into
                            # ``self.n_splits_outer`` outer folds using the
                            # ``split_cv`` function. Splits are different for
                            # each repeat depending on the ``random_state``
                            # parameter.
                            splits_inner = split_cv(
                                dataset=train_val_dataset,
                                n_splits=self.n_splits_inner,
                                split_mode=self.split_mode,
                                random_state=r_inner,
                            )
                            for s_inner, (
                                train_indices,
                                val_indices,
                            ) in enumerate(splits_inner):
                                logger.info(
                                    f"Inner cross-validation: repeat_{r_inner+1}_split_{s_inner+1}"
                                )
                                # Split inner-cv dataset into training folds
                                # and a validation fold.
                                train_dataset = Subset(
                                    train_val_dataset, indices=train_indices
                                )
                                val_dataset = Subset(
                                    train_val_dataset, indices=val_indices
                                )

                                # Instantiate the trainer. For each fold of
                                # the inner cross-validation, the trainer is
                                # re-initialized.
                                trainer = instantiate(trainer_cfg)

                                # Perform training and retrieve end of
                                # training metrics.
                                train_metrics, val_metrics, _, train_losses, val_losses = trainer.train(
                                    train_dataset, val_dataset
                                )

                                # Store metrics.
                                for k, v in train_metrics.items():
                                    if k in cv_train_metrics:
                                        cv_train_metrics[k][
                                            f"repeat_{r_inner+1}_split_{s_inner+1}"
                                        ] = v
                                    else:
                                        cv_train_metrics[k] = {
                                            f"repeat_{r_inner+1}_split_{s_inner+1}": v
                                        }

                                for k, v in val_metrics.items():
                                    if k in cv_val_metrics:
                                        cv_val_metrics[k][
                                            f"repeat_{r_inner+1}_split_{s_inner+1}"
                                        ] = v
                                    else:
                                        cv_val_metrics[k] = {
                                            f"repeat_{r_inner+1}_split_{s_inner+1}": v
                                        }

                                # Log inner split metrics.
                                logger.info("---- Inner splits metrics ----")
                                logger.info(
                                    f"Repeat {r_inner+1}, Split {s_inner+1}"
                                )
                                logger.info("Training folds metrics:")
                                for k, v in train_metrics.items():
                                    logger.info(f"   {k}: {v[-1]:.3f}")
                                logger.info("Validation fold metrics:")
                                for k, v in val_metrics.items():
                                    logger.info(f"   {k}: {v[-1]:.3f}")
                                #print(f"Best val cindex at epoch number {np.argmax(val_metrics['cindex'])} out of {len(val_metrics['cindex'])} total epochs trained. At {np.max(val_metrics['cindex'])}")

                                #df = pd.DataFrame({"tloss": train_losses, "vloss": val_losses})
                                #df.to_csv(f"gs_{i}_losses_inner"+str(s_inner)+"_outer"+str(s_outer)+".csv")

                        logger.info(
                            "---- Inner cross-validation train metrics ----"
                        )
                        for k, v in cv_train_metrics.items():
                            logger.info(
                                f"   {k}: {np.mean(list(v.values())):.3f} "
                                f"({np.std(list(v.values())):.3f})"
                            )

                        logger.info(
                            "---- Inner cross-validation validation metrics ----"
                        )
                        for k, v in cv_val_metrics.items():
                            logger.info(
                                f"   {k}: {np.mean(list(v.values())):.3f} "
                                f"({np.std(list(v.values())):.3f})"
                            )
                        # Gather metrics from the inner cross-validation at
                        # each of the training epochs, so that to select
                        # the best epoch.
                        metrics_names = cv_val_metrics.keys()
                        if "cindex" in metrics_names:
                            mean_cv_val_metrics_per_epochs = (
                                np.array(
                                    list(cv_val_metrics["cindex"].values())
                                )
                                .reshape(
                                    (
                                        self.n_splits_inner,
                                        trainer_cfg.num_epochs,
                                    )
                                )
                                .mean(axis=0)
                            ) # Average over splits. So len is num_epochs
                        elif "auc" in metrics_names:
                            mean_cv_val_metrics_per_epochs = (
                                np.array(list(cv_val_metrics["auc"].values()))
                                .reshape(
                                    (
                                        self.n_splits_inner,
                                        trainer_cfg.num_epochs,
                                    )
                                )
                                .mean(axis=0)
                            )
                        else:
                            raise ValueError(
                                "Neither `cindex` nor `auc` were found in metrics."
                            )
                        # Best epoch selection.
                        optimal_epoch = mean_cv_val_metrics_per_epochs.argmax()
                        print("OPTIMAL EPOCH", optimal_epoch, "with man cv_val", mean_cv_val_metrics_per_epochs[optimal_epoch])
                        optimal_epoch = len(mean_cv_val_metrics_per_epochs)-1
                        print("Setting optimal epoch to max:", optimal_epoch)
            
                        # We report the metrics on the best epoch. (averaged over inner splits)
                        list_val_metrics.append(
                            mean_cv_val_metrics_per_epochs[optimal_epoch]
                        )
                        optimal_epoch += 1 # Because of zero indexing
                        if optimal_epoch<10:
                            optimal_epoch = 10
                        sub_cfg.update({"num_epochs": int(optimal_epoch)})
                        list_cfgs.append(sub_cfg)

                    # Determine optimal config.
                    optimal_cfg = list_cfgs[np.argmax(list_val_metrics)]
                    # Log optimal config,
                    logger.info("---- Optimal config ----")
                    logger.info(optimal_cfg)
                else:
                    logger.info("No gridsearch performed.")
                    # If no grid-search is performed.
                    optimal_cfg = {}

                # Initialize the trainer with the optimal configuration.
                trainer_cfg = copy.deepcopy(self.trainer_cfg)
                trainer_cfg.update(optimal_cfg)
                update_trainer_cfg(trainer_cfg, n_features, n_labels)
                trainer = instantiate(trainer_cfg)

                # if len(list(set(train_val_dataset.dataset.labels)))==2:
                #     balance_loader = True
                # else:
                #     balance_loader = False
                balance_loader = False
                # Re train with the optimal configuration on the whole
                # inner folds, and assess performance on the outer fold.
                print(f"Retraining with optimal config") #. Outer split {r_outer}, outer repeat {s_outer}")
                train_val_metrics, train_val_loss, test_metrics, test_loss, _test_labels, _test_logits = trainer.train(
                    train_val_dataset, test_dataset, None, balance_loader=balance_loader
                )

                if not add_data:
                    # register (logits, label)
                    #print("Test logits:", type(_test_logits), _test_logits.shape)
                    #norm_logits = (_test_logits-_test_logits.min()) / (_test_logits.max()-_test_logits.min()) # Scale between 0 to 1
                    #norm_logits = _test_logits/_test_logits.mean() # Scale by mean -> new mean is 1
                    norm_logits = (_test_logits - _test_logits.mean()) /  _test_logits.std()
                    print("normalized logits stats:", norm_logits.mean(), norm_logits.std())
                    test_labels = np.append(test_labels, _test_labels)
                    test_logits = np.append(test_logits, norm_logits)
                
                # if add_data and not mix_datasets:
                #     if test_cohort=="UNN_NSCLC":
                #         wrong_slides = np.concatenate([wrong_slides, _wrong_slides])
                # elif dataset.cohort=="UNN_NSCLC":
                #     wrong_slides = np.concatenate([wrong_slides, _wrong_slides])

                #df = pd.DataFrame({"train_val_loss": train_val_loss, "test_loss": test_loss})
                #df.to_csv(f"long_losses_outer"+str(s_outer)+".csv")

                # Store outer-folds metrics.
                for k, v in train_val_metrics.items():
                    if k in cv_train_val_metrics:
                        cv_train_val_metrics[k][
                            f"repeat_{r_outer+1}_split_{s_outer+1}"
                        ] = v
                    else:
                        cv_train_val_metrics[k] = {
                            f"repeat_{r_outer+1}_split_{s_outer+1}": v
                        }

                for k, v in test_metrics.items():
                    if k in cv_test_metrics:
                        cv_test_metrics[k][
                            f"repeat_{r_outer+1}_split_{s_outer+1}"
                        ] = v
                    else:
                        cv_test_metrics[k] = {
                            f"repeat_{r_outer+1}_split_{s_outer+1}": v
                        }

                for k, v in slide_labels.items():
                    if k in slide_labels_all:
                        slide_labels_all[k][
                            f"repeat_{r_outer+1}_split_{s_outer+1}"
                        ] = v
                    else:
                        slide_labels_all[k] = {
                            f"repeat_{r_outer+1}_split_{s_outer+1}": v
                        }

                # Log outer split metrics.
                logger.info("---- Outer splits metrics ----")
                logger.info(f"Repeat {r_outer+1}, Split {s_outer+1}")
                logger.info("Training folds metrics:")
                for k, v in train_val_metrics.items():
                    logger.info(f"   {k}: {v[-1]:.3f}")
                logger.info("Test fold metrics:")
                if add_data and not mix_datasets:
                    logger.info(f"  for test cohort {test_cohort}")
                else:
                    logger.info(f"  for test split ")
                for k, v in test_metrics.items():
                    logger.info(f"   {k}: {v[-1]:.3f}")

                #df = pd.DataFrame({"tvloss": train_val_loss, "testloss": test_loss})
                #df.to_csv("losses_test_outer"+str(s_outer)+".csv")
            
            # Compute c-index based on logits from all folds
            if not add_data:
                rep_test_metrics = trainer.compute_metrics(test_labels, test_logits)
                print("rep cindex is ", rep_test_metrics)
                for k, v in rep_test_metrics.items():
                    if k in r_test_metrics:
                        r_test_metrics[k][
                            f"repeat_{r_outer+1}"
                        ] = v
                    else:
                        r_test_metrics[k] = {
                            f"repeat_{r_outer+1}": v
                        }

        # Calculate metrics over 
        
        # Log outer cross-validation metrics.
        logger.info("")
        logger.success("---- FINAL RESULTS ----")
        logger.success("---- Outer cross-validation train metrics ----")
        for k, v in cv_train_val_metrics.items():
            # The last epoch is the best epoch (as training ended at the
            # best epoch identified during inner CV).
            logger.success(
                f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.3f} "
                f"({np.std([_v[-1] for _v in v.values()]):.3f})"
            )
        logger.success("---- Outer cross-validation test metrics ----")
        for k, v in cv_test_metrics.items():
            logger.success(
                f"   {k}: {np.mean([_v[-1] for _v in v.values()]):.3f} "
                f"({np.std([_v[-1] for _v in v.values()]):.3f})"
            )

        logger.success("----- Slide classifications -----")
        if self.n_repeats_outer > 1 and [k for k in cv_test_metrics.keys()][0]=="cindex":
            splits1 = [v[-1] for k,v in cv_test_metrics['cindex'].items() if str("split_1") in k]
            logger.info(f"Splits 1 has metrics: {np.mean(splits1):.3f}  ({np.std(splits1):.3f})")
            splits2 = [v[-1] for k,v in cv_test_metrics['cindex'].items() if str("split_2") in k]
            logger.info(f"Splits 2 has metrics: {np.mean(splits2):.3f}  ({np.std(splits2):.3f})")
            splits3 = [v[-1] for k,v in cv_test_metrics['cindex'].items() if str("split_3") in k]
            logger.info(f"Splits 3 has metrics: {np.mean(splits3):.3f}  ({np.std(splits3):.3f})")
        else:
            for metric in cv_test_metrics.keys():
                logger.info(f"Metric {metric}")
                splits1 = [v[-1] for k,v in cv_test_metrics[metric].items() if str("split_1") in k]
                logger.info(f"Splits 1 has metrics: {np.mean(splits1):.3f}  ({np.std(splits1):.3f})")
                splits2 = [v[-1] for k,v in cv_test_metrics[metric].items() if str("split_2") in k]
                logger.info(f"Splits 2 has metrics: {np.mean(splits2):.3f}  ({np.std(splits2):.3f})")
                splits3 = [v[-1] for k,v in cv_test_metrics[metric].items() if str("split_3") in k]
                logger.info(f"Splits 3 has metrics: {np.mean(splits3):.3f}  ({np.std(splits3):.3f})")

        if not add_data:
            logger.info(f"Metrics over datasets")
            for k, v in r_test_metrics.items():
                logger.success(
                    f"   {k}: {np.mean(list(v.values())):.3f} "
                    f"({np.std(list(v.values())):.3f})"
                )

        if add_data and not mix_datasets:
            logger.info("Cross-dataset analysis.")
        print(f"# slides considered: {len(slide_labels_all.keys())}", flush=True)
        
        # Make sure slide keys are str, not Path. And save the results
        slide_labels_all = {str(k):v for k,v in slide_labels_all.items()}
        try:
            datatype = list(slide_labels_all.keys())[0].split("/")[3]
        except Exception as e:
            print("failed to retrieve datatype", e)
            datatype = str(np.random.randint(0,999))

        datatype += "_"+str(np.random.randint(0,999))

        # import json
        # with open("logs_root_dmz/cross_validation/"+datatype+".json", "w") as out:
        #     json.dump(slide_labels_all, out)
        # logger.success(f"- Slide labels successfully saved at {datatype} -")

        # import IPython
        # IPython.embed()
        # Read previous wrong slides and add new ones

        # import pickle
        # with open("logs_root_dmz_dino2/wrongslides/UNN", "rb") as out:
        #     ss = pickle.load(out)
        # print("Old registered wrong slides of shape", ss.shape)
        # new_list = np.concatenate([wrong_slides, ss])
        # #new_list = wrong_slides
        # with open("logs_root_dmz_dino2/wrongslides/UNN", "wb") as out:
        #     pickle.dump(new_list, out)

        # #print(wrong_slides)
        # array, counts = np.unique(new_list, return_counts=True)
        # print("Most common mistakes in ", array[counts==counts.max()])
        # print(f"That is a total of {(counts==counts.max()).sum()} slides. They have all been mistakenly classified {counts.max()} times.")
        

        return cv_train_val_metrics, cv_test_metrics
