# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Script to perform nested cross-validation on slide-level tasks."""

import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append("/home/vilde/code/Phikon/HistoSSLscaling")

from rl_benchmarks.utils import (
    log_slide_dataset_info,
    params2experiment_name,
    resume_training,
    save_pickle,
    store_params,
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
    n_tiles = int(data_cfg["n_tiles"])

    training_datasets = list(params["training_datasets"])
    testing_datasets = list(params["testing_datasets"])

    # Create experiment name.
    experiment_name = OmegaConf.to_container(copy.deepcopy(params))
    # Remove parameters from `experiment_name` that do not justify to
    # re-run the experiments, if different.
    experiment_name.pop("features_root_dir")
    experiment_name.pop("logs_root_dir")
    experiment_name["task"]["validation_scheme"]["trainer_cfg"].pop("device")
    experiment_name = params2experiment_name(experiment_name)

    # Create experiment folder and check if the experiment is already completed.
    logs_root_dir = params["logs_root_dir"]
    if logs_root_dir is None:
        logs_root_dir = LOGS_ROOT_DIR

    experiment_folder = (
        Path(logs_root_dir) / "cross_validation" / experiment_name
    )
    resume_training(experiment_folder)

    # Store experiment status (will be set to True in case of success).
    experiment_status = {"completed": False}
    path = experiment_folder / "status.pkl"
    save_pickle(path, experiment_status)

    # Create log file.
    path = experiment_folder / f"{experiment_name}.log"
    log_path_id = logger.add(path)

    # Start logging.
    logger.info("Running cross-validation script...\n")
    logger.info(
        f"Experiment name: {experiment_name}.\n"
        f"Experiment folder: {experiment_folder}\n"
    )
    print("Running cross-validation script...")
    print(f"Experiment name: {experiment_name}.\nExperiment folder: {experiment_folder}\n", flush=True )

    # Store parameters.
    store_params(experiment_folder, params)

    # Log main task Hydra configuration.
    logger.info("---- Task configuration info ----")

    logger.info(f"Training datasets: {training_datasets}")
    logger.info(f"Testing datasets: {testing_datasets}")
    logger.info("Run configuration: \n" + OmegaConf.to_yaml(params))
    logger.info("\n")

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

    if "linear" in str(model_cfg).lower():
        features_root_dir = features_root_dir / "mean"
    logger.info(f"Retrieving features from: {features_root_dir}")

    # Instantiate and log dataset info.
    # test_dataset = instantiate(
    #     data_cfg["train"], features_root_dir=features_root_dir
    # )
    # logger.info("---- Dataset info ----")
    # log_slide_dataset_info(test_dataset)

    # Load training data
    train_datasets = []
    for i in range(len(training_datasets)):
        int_datacfg = data_cfg["train"].copy()
        int_datacfg["cohort"] = training_datasets[i]
        if int_datacfg["cohort"] == "UNN_NSCLC":
            feat_dir = str(features_root_dir)#.replace("features", "features_gray")
            d = instantiate(
                int_datacfg, features_root_dir=Path(feat_dir), n_tiles=n_tiles, shuffle=True
            )
        else:
            d = instantiate(
                int_datacfg, features_root_dir=features_root_dir, n_tiles=n_tiles, shuffle=True
            )
        train_datasets.append(d)
        log_slide_dataset_info(d)
    if len(train_datasets)==1:
        #logger.info("Train datsets len 1")
        train_datasets = train_datasets[0]

    # Load test data
    test_datasets = []
    for i in range(len(testing_datasets)):
        int_datacfg = data_cfg["train"].copy()
        int_datacfg["cohort"] = testing_datasets[i]
        if int_datacfg["cohort"] == "UNN_NSCLC":
            feat_dir = str(features_root_dir)#.replace("features", "features_gray")
            d = instantiate(
                int_datacfg, features_root_dir=Path(feat_dir), n_tiles=n_tiles, shuffle=False
            )
        else:
            d = instantiate(
                int_datacfg, features_root_dir=features_root_dir, n_tiles=n_tiles, shuffle=False
            )
        test_datasets.append(d)
        log_slide_dataset_info(d)
    if len(test_datasets)==1:
        #logger.info("Test datsets len 1")
        test_datasets = test_datasets[0]

    print("Data is loaded")

    # Instantiate validation scheme, which is here NestedCV.
    validation_scheme = instantiate(
        validation_scheme_cfg,
        _recursive_=False,
    )

    # Run Nested CV.
    print("Start CV run")
    max_epochs = validation_scheme_cfg["trainer_cfg"]["num_epochs"]
    n_repeats = validation_scheme_cfg["n_repeats_outer"]
    train_metrics, test_metrics = validation_scheme.run(test_dataset=test_datasets, train_dataset=train_datasets, epochs=max_epochs)
    train_c = np.empty((max_epochs+1,n_repeats))
    test_c = np.empty((max_epochs+1,n_repeats))

    for i,v in enumerate(train_metrics.values()):
        train_c[:,i] = v
    for i,v in enumerate(test_metrics.values()):
        test_c[:,i] = v

    #np.save(experiment_folder / "train_metrics.npy", train_c)
    #np.save(experiment_folder / "test_metrics.npy", test_c)

    avg_train = train_c.mean(axis=1)
    avg_test = test_c.mean(axis=1)
    std_test = test_c.std(axis=1)
    
    results = {"test avg":avg_test, "train avg": avg_train, "std test":std_test}
    df = pd.DataFrame(results)
    df.to_csv(experiment_folder / "performance.csv", index=False)

    n_reps = test_c.shape[1]
    plt.rcParams.update({'font.size':15})
    fig = plt.figure(figsize=(8,6))
    plt.plot(range(max_epochs+1), avg_train, label="Train", color='orange')
    plt.plot(range(max_epochs+1), avg_test, label="Test", color='red')
    plt.fill_between(range(max_epochs+1), avg_test-std_test, avg_test+std_test, alpha=0.3, color='yellow', label='+- std')
    #plt.xticks(list(range(max_epochs)), labels=list(range(1,1+max_epochs)))
    plt.title(f"c-index scores VS total epochs trained\nmean over {n_reps} runs")
    plt.xlabel("Epochs trained")
    plt.ylabel("average c-index")
    plt.ylim(np.min(np.concatenate([avg_train, avg_test, [0.49]])), np.max(np.concatenate([avg_test+std_test, [0.615]])))
    plt.legend()
    plt.tight_layout()
    plt.savefig(experiment_folder / "performance.png")

    # Remove logger's sink.
    logger.remove(log_path_id)

    # Store experiment status.
    experiment_status = {"completed": True}
    path = experiment_folder / "status.pkl"
    save_pickle(path, experiment_status)

    print(f"Features from {features_root_dir}")
    print("Experiment name", experiment_name)


if __name__ == "__main__":
    print("In CV file", flush=True)
    main()  # pylint: disable=no-value-for-parameter
