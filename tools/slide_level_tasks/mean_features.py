
# Load features
# Create dataloader

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Script to perform nested cross-validation on slide-level tasks."""

import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/vilde/code/Phikon/HistoSSLscaling")

from rl_benchmarks.utils import (
    log_slide_dataset_info,
    params2experiment_name,
    resume_training,
    save_pickle,
    store_params,
    pad_collate_fn
)
from rl_benchmarks.constants import (
    LOGS_ROOT_DIR,
    PREPROCESSED_DATA_DIR,
)


@hydra.main(
    version_base=None,
    config_path="../../conf/slide_level_task/cross_validation/",
    config_name="config_extreme",
)
def main(params: DictConfig) -> None:
    """Perform nested cross-validation for a given task.
    The Hydra configuration files defining all slide-level downstream tasks
    can be found in ``'conf/slide_level_tasks/cross_validation/*.yaml'``."""
    # Get parameters.
    model_cfg = params["model"]
    task_cfg = params["task"]
    data_cfg = task_cfg["data"]
    validation_scheme_cfg = task_cfg["validation_scheme"]
    validation_scheme_cfg["trainer_cfg"]["model"] = model_cfg
    tune_version = params["tune_version"]


    training_datasets = list(params["training_datasets"])
    testing_datasets = list(params["testing_datasets"])


    # Load the data to perform nested CV on.
    feature_extractor_name = data_cfg["feature_extractor"]
    features_root_dir = params["features_root_dir"]
    if features_root_dir is None:
        features_root_dir = PREPROCESSED_DATA_DIR
    
    if "tuned" in feature_extractor_name:
        features_root_dir = (
            Path(features_root_dir)
            / "slides_classification"
            / "features_tuned/_2" #/ep20
            / feature_extractor_name
            / tune_version
        )
        print("Using features from tuning: ", tune_version)
    else:
        features_root_dir = (
            Path(features_root_dir)
            / "slides_classification"
            / "features_2"
            / feature_extractor_name
        )
    logger.info(f"Retrieving features from: {features_root_dir}")

    # Load training data
    datasets = []
    for i in range(len(training_datasets)):
        int_datacfg = data_cfg["train"].copy()
        int_datacfg["cohort"] = training_datasets[i]
        if int_datacfg["cohort"] == "UNN_NSCLC":
            feat_dir = str(features_root_dir)#.replace("features", "features_gray")
            d = instantiate(
                int_datacfg, features_root_dir=Path(feat_dir)
            )
        else:
            d = instantiate(
                int_datacfg, features_root_dir=features_root_dir
            )
        datasets.append(d)
    # Load test data
    for i in range(len(testing_datasets)):
        int_datacfg = data_cfg["train"].copy()
        int_datacfg["cohort"] = testing_datasets[i]
        if int_datacfg["cohort"] == "UNN_NSCLC":
            feat_dir = str(features_root_dir)#.replace("features", "features_gray")
            d = instantiate(
                int_datacfg, features_root_dir=Path(feat_dir)
            )
        else:
            d = instantiate(
                int_datacfg, features_root_dir=features_root_dir
            )
        datasets.append(d)

    print("Data is loaded")
    use_data = ConcatDataset(datasets)

    save_at = Path(features_root_dir) / "mean"
    print("ONLY SAVING LUSC FEATURES!!")

    for i in range(len(use_data)):
        feature, label, slide = use_data[i]
        if "TCGA" in slide:
            save_new = Path(save_at) / "TCGA" / "TCGA_LUSC" / slide
        elif "UNN" in slide:
            save_new = Path(save_at) / "UNN_LUSC" / slide
        if "S36" in slide:
            save_new = Path(save_at) / "S36_LUSC" / slide
        save_new.mkdir(parents=True, exist_ok=True)

        # Calculate feature means.
        feature = feature[..., 3:] # Removes metadata
        if feature.shape[0] < 1000:
            print("Feature shape < 1000 :)", feature.shape[0])
        # Average over the number of tiles
        new_feature = feature.mean(axis=0)
        if int(new_feature.shape[0]) != 2560:
            print("Feature shape", feature.shape)
            print(new_feature.shape, save_new)
            sys.exit()
        #print("New feature shape", new_feature.shape)
        np.save(str(save_new / "features.npy"), new_feature)

    print("Mean ved for all slides in ", features_root_dir)






if __name__ == "__main__":
    print("In file", flush=True)
    main()  # pylint: disable=no-value-for-parameter
