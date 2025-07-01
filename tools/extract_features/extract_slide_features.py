# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extraction script for WSIs."""

import json
import shutil
from pathlib import Path

import hydra
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from PIL.Image import DecompressionBombError
import torch
from transformers import ViTModel
from tqdm import tqdm

import sys
sys.path.append("/home/vilde/code/Phikon/HistoSSLscaling")

from rl_benchmarks.constants import (
    AVAILABLE_COHORTS,
    PREPROCESSED_DATA_DIR,
    TILE_SIZES,
)
from rl_benchmarks.models import ParallelExtractor
from rl_benchmarks.utils import extract_from_slide, get_slide_config, set_seed


class InsufficientTilesError(RuntimeError):
    """Error raised when a slide does not have enough tiles."""


class MPPNotAvailableError(Exception):
    """Exception raised when the asked MPP is not available in the slide."""


# Here to avoid Pillow DecompressionBombError that could appear
# If the number of pixels is greater than twice MAX_IMAGE_PIXELS.
# See the Pillow documentation (https://pillow.readthedocs.io/en/stable/reference/Image.html)
Image.MAX_IMAGE_PIXELS = None


@hydra.main(
    version_base=None,
    config_path="../../conf/extract_features/",
    config_name="slide_config",
)
def extract_slide_features(params: DictConfig) -> None:
    """Perform feature extraction for a given dataset of slides.
    The Hydra configuration file can be found in `conf/extract_features/slide_config.yaml`.
    See `conf/extract_features/slide_dataset` and `conf/extract_features/feature_extractor`
    for the list of available datasets and feature extractors, respectively."""
    # Set seed for features extraction (in case of random subsampling).
    set_seed()

    # Prepare output directory where features will be stored.
    features_output_dir = params["features_output_dir"]
    dataset_cfg = params["slide_dataset"]
    cohort = dataset_cfg["cohort"]
    aug = params["augmentation"]
    if features_output_dir is None:
        if cohort not in AVAILABLE_COHORTS:
            raise ValueError(
                f"{cohort} is not found. Available cohorts can be found in"
                " ``rl_benchmarks.constants::AVAILABLE_COHORTS``."
            )
        features_output_dir = (
            PREPROCESSED_DATA_DIR / "slide_classification" / "features"
        )
    else:
        features_output_dir = Path(features_output_dir)
    hydra_cfg = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    feature_extractor_name = hydra_cfg["feature_extractor"]
    print("Using feature extractor", feature_extractor_name, flush=True)
    if "TCGA" in cohort:
        features_output_dir = (
            features_output_dir / feature_extractor_name / "TCGA" / cohort
        )
    else:
        features_output_dir = (
            features_output_dir / feature_extractor_name / cohort
        )
    if "tuned" in feature_extractor_name:
        features_output_dir = Path(str(features_output_dir).replace("phikon_tuned", "phikon_tuned/"+str(params["pretrained_dir"]).split("/")[-2]))
        features_output_dir = Path(str(features_output_dir).replace("v2_tuned", "v2_tuned/"+str(params["pretrained_dir"]).split("/")[-2]))
        features_output_dir = Path(str(features_output_dir).replace("p2_tuned", "p2_tuned/"+str(params["pretrained_dir"]).split("/")[-2]))
        features_output_dir = Path(str(features_output_dir).replace("features", "features_tuned/"))
    features_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Storage folder: {features_output_dir}.")

    # Define parameters for the features extraction process.
    feature_extractor_cfg = params["feature_extractor"]
    device = params["device"]
    print(f"Device {device}, feature extractor {feature_extractor_name}", flush=True)
    tile_size = params["tile_size"]
    n_tiles = params["n_tiles"]
    if tile_size == "auto":
        tile_size = TILE_SIZES[feature_extractor_name]
    else:
        assert (
            TILE_SIZES[feature_extractor_name] == tile_size
        ), f"Please specify a tile size (in pixels) that matches the original implementation, see constants.TILE_SIZES dictionary for details: {TILE_SIZES}"
    random_sampling = params["random_sampling"]
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]

    # Load the data.
    dataset_loading_fn = instantiate(dataset_cfg)
    dataset = dataset_loading_fn(
        features_root_dir=features_output_dir, tile_size=tile_size
    )
    slides_paths = dataset.slide_path.values
    coords_paths = dataset.coords_path.values

    print("len slides / coords:", len(slides_paths), len(coords_paths))

    # Prepare the feature extractor.
    extractor = instantiate(feature_extractor_cfg)
    dinov2_tune=False
    if "dinov2" in params["pretrained_dir"]:
        dinov2_tune=True
    if "tuned" in feature_extractor_name:
        extractor.load_pretrained(params["pretrained_dir"], dinov2_tune)
        print("Using tuned model from", params["pretrained_dir"])
    extractor = ParallelExtractor(extractor, gpu=device)   
    print("Device", device)

    # import IPython
    # IPython.embed()
    # sys.exit()

    # Save output config.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_features_output_dir = Path(hydra_cfg["runtime"]["output_dir"])
    hydra_yaml_cfg = hydra_features_output_dir / ".hydra" / "config.yaml"
    with open(hydra_yaml_cfg, "r", encoding="utf-8") as stream:
        hydra_yaml_cfg = yaml.safe_load(stream)


    shutil.copyfile(
        Path(__file__).resolve(),
        features_output_dir / "extraction_script.py",
    )

    output_cfg = get_slide_config(
        params, features_output_dir, slides_paths, coords_paths, hydra_yaml_cfg
    )
    with open(
        features_output_dir / "extraction_params.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(output_cfg, fp)


    print("hydra", hydra_yaml_cfg)
    print(f"Output dir: {features_output_dir}")

    # Perform tiling.
    for i, slide_path in tqdm(
        enumerate(slides_paths), total=len(slides_paths)
    ):
        print("In for loop", i, str(slide_path).split("/")[-2:], flush=True)
        # Create storage folder specific to the slide being processed.

        if "NLST" in cohort:
            s_id = "_".join(str(slide_path).split("/")[-2:]) # Patient ID _ slide ID
            slide_export_dir = features_output_dir / Path(s_id)
        else:
            slide_export_dir = features_output_dir / Path(slide_path).name

        slide_export_dir.mkdir(exist_ok=True)

        # Get features full path for the slide being processed..
        slide_features_path = slide_export_dir / "features.npy"

        if not aug and slide_features_path.exists():
            logger.info(
                f"  Extraction for slide {Path(slide_path).name} already "
                f"done. \n"
                f"  Features can be found at {slide_export_dir}"
            )
            print("  Extraction already done.", flush=True)
            continue

        # Get coordinates.
        coords_path = dataset.coords_path.values[i]
        coords = np.load(coords_path).astype(int)

        try:
            #print("in try", flush=True)
            features = extract_from_slide(
                slide=slide_path,
                level=coords[0, 0],
                coords=coords[:, 1:],
                feature_extractor=extractor,
                tile_size=tile_size,
                n_tiles=n_tiles,
                random_sampling=random_sampling,
                batch_size=batch_size,
                num_workers=num_workers,
                augmentation=aug,
            )
            #print("Finish try", flush=True)

        except InsufficientTilesError:
            logger.error(
                f"Extraction for slide {Path(slide_path).name} "
                f"failed due to insufficient tiles on slide."
            )
            continue

        except MPPNotAvailableError:
            logger.error(
                f"Extraction for slide {Path(slide_path).name} "
                f"failed due to the absence of MPP in metadata."
            )
            continue

        except KeyError as err:
            logger.error(err)
            continue

        except DecompressionBombError as err:
            logger.error(err)
            continue

        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.info(
                f"Extraction for slide {Path(slide_path).name} failed ("
                f"{err})."
            )
            continue

        # Save features.
        np.save(str(slide_features_path), features)
        logger.success(f"Features saved at {str(slide_features_path)}.")
        print("  Success", flush=True)

    print("All features have been extracted :D")

if __name__ == "__main__":
    extract_slide_features()  # pylint: disable=no-value-for-parameter
