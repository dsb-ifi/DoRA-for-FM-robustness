# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading module."""
import torch
from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

# from .slides_classification.camelyon16 import load_camelyon16
from .slides_classification.core import SlideFeaturesDataset
from .slides_classification.tcga import load_tcga
from .slides_classification.unn import load_unn
from .slides_classification.s36 import load_s36
from .slides_classification.nlst import load_nlst
#from .tiles_classification.camelyon17_wilds import load_camelyon17_wilds
#from .tiles_classification.nct_crc import load_nct_crc
from ..constants import AVAILABLE_COHORTS
from ..utils import preload_features


def load_dataset(
    cohort,
    features_root_dir: Optional[str] = None,
    label: Optional[str] = None,
    tile_size: int = 224,
    load_slide: Optional[bool] = True,
) -> pd.DataFrame:
    """For each dataset (and outcome if applicable), load the following data:

    * Slides datasets *
    "patient_id": patient ID
    "slide_id": slide ID
    "slide_path": path to the slide (raw slide)
    "coords_path": path to the existing tiles coordinates from the slide (numpy arrays)
    "label": outcome to predict

    * Tiles datasets *
    "image_id": image ID
    "image_path": path to the tile
    "center_id": center ID (optional)
    "label": tissue class (0 to 8, NCT-CRC) or presence of tumor (0 or 1, Camelyon17-WILDS)

    Parameters
    ----------
    cohort: str
        Name of the cohort:
        - For TCGA cohorts:
            ``'TCGA_COAD'``,``'TCGA_READ'``, etc.
            See ``slides_classification/tcga.py`` for details.
        - For Camelyon16 dataset:
            ``'CAMELYON16_TRAIN'``, ``'CAMELYON16_TEST'`` or``'CAMELYON16_FULL'``
            See ``slides_classification/camelyon16.py`` for details.
        - For Camelyon17-WILDS dataset:
            ``'CAMELYON17-WILDS_TRAIN'``, ``'CAMELYON17-WILDS_VALID'``, ``'CAMELYON17-WILDS_TEST'``
            or ``'CAMELYON17-WILDS_FULL'``.
            See ``tiles_classification/camelyon17_wilds.py`` for details.
        - For NCT-CRC dataset:
            ``'NCT-CRC_TRAIN'``, ``'NCT-CRC_VALID'`` or ``'NCT-CRC_FULL'``.
            See ``tiles_classification/nct_crc.py`` for details.
        - For internal cohorts:
            ``'UNN'``, ``'S36'``
            See ``slides_classification/unn.py`` for details

    features_root_dir: Union[str, Path]
        Path to the histology features' root directory e.g.
        * Slides datasets *
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/CAMELYON16_FULL/

        or

        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/TCGA/


        * Tiles datasets *
        /home/user/data/rl_benchmarks_data/preprocessed/
        tiles_classification/features/iBOTViTBasePANCAN/NCT-CRC_FULL/

        or

        /home/user/data/rl_benchmarks_data/preprocessed/
        tiles_classification/features/iBOTViTBasePANCAN/CAMELYON17-WILDS_FULL/

        If no features have been extracted yet, `features_path` is made of NaNs.

    label: Optional[str] = None
        Only needed for TCGA cohorts.
        The task-specific label. Valid labels are: ``'MOLECULAR_SUBTYPE'``,
        ``'HISTOLOGICAL_SUBTYPE'``, ``'TUMOR_TYPE'``, ``'CANCER_SUBTYPE'``,
        ``'SURVIVAL'``, ``'MSI'`` and ``'HRD'``. (added: ``'SURVIVAL_MINE'``)

    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
        This parameter is automatically picked up during feature extraction
        depending on the feature extractor at stake.
        See ``rl_benchmarks.constants.TILE_SIZES``.

    cohort : str = "TCGA_COAD"
        Name of the TCGA cohort to consider. Valid TCGA cohorts are: ``'COAD'`` (colon
        adenocarcinoma), ``'READ'`` (rectum adenocarcinoma), ``'LUAD'`` (lung
        adenocarcinoma), ``'LUSC'`` (lung squamous cell carcinoma), ``'BRCA'``
        (breast invasive carcinoma), ``'KIRC'`` (kidney renal clear cell carcinoma),
        ``'KIRP'`` (kidney renal papillary cell carcinoma), ``'KICH'`` (kidney
        chromophobe), ``'OV'`` (ovarian serous cystadenocarcinoma), ``'MESO'``
        (mesothelioma), ``'PAAD'`` (pancreatic adenocarcinoma), ``'PRAD'``
        (prostate adenocarcinoma).

    load_slide: bool = False
        Add slides paths if those are needed. This parameter should be set
        to ``False`` if slides paths are not needed, i.e. for downstream tasks
        as only features matter, or ``True`` for features extraction (features
        have not been generated from slides yet).
    """
    if cohort not in AVAILABLE_COHORTS:
        raise ValueError(
            f"Please specify a cohort in {AVAILABLE_COHORTS}. Cohort: {cohort} is not supported."
        )

    kwargs = {
        "features_root_dir": features_root_dir,
        "tile_size": tile_size,
        "load_slide": load_slide,
    }
    print("load slide", load_slide)

    # Slide-level datasets.
    # TCGA cohorts.
    if "TCGA" in cohort:
        if cohort == "TCGA_NSCLC":
            df_luad = load_tcga(cohort="TCGA_LUAD", label=label, **kwargs)
            df_lusc = load_tcga(cohort="TCGA_LUSC", label=label, **kwargs)
            dataset = pd.concat([df_luad, df_lusc], axis=0)
        elif cohort == "TCGA_CRC":
            df_coad = load_tcga(cohort="TCGA_COAD", label=label, **kwargs)
            df_read = load_tcga(cohort="TCGA_READ", label=label, **kwargs)
            dataset = pd.concat([df_coad, df_read], axis=0)
        elif cohort == "TCGA_RCC":
            df_kirc = load_tcga(cohort="TCGA_KIRC", label=label, **kwargs)
            df_kirp = load_tcga(cohort="TCGA_KIRP", label=label, **kwargs)
            df_kirch = load_tcga(cohort="TCGA_KICH", label=label, **kwargs)
            dataset = pd.concat([df_kirc, df_kirp, df_kirch], axis=0)
        else:
            dataset = load_tcga(cohort=cohort, label=label, **kwargs)
        drop_na_columns = ["label"]
        if load_slide:
            drop_na_columns += ["slide_path"]
            drop_na_columns += ["coords_path"]
        else:
            dataset["slide_path"] = None

    elif "UNN" in cohort:
        if cohort == "UNN_NSCLC":
            df_luad = load_unn(cohort="UNN_LUAD", label=label, **kwargs)
            df_lusc = load_unn(cohort="UNN_LUSC", label=label, **kwargs)
            dataset = pd.concat([df_luad, df_lusc], axis=0)
        else:
            dataset = load_unn(cohort=cohort, label=label, **kwargs)
        drop_na_columns = ["label"]
        if load_slide:
            drop_na_columns += ["slide_path"]
            drop_na_columns += ["coords_path"]
        else:
            dataset["slide_path"] = None

    elif "S36" in cohort:
        if cohort == "S36_NSCLC":
            df_luad = load_s36(cohort="S36_LUAD", label=label, **kwargs)
            df_lusc = load_s36(cohort="S36_LUSC", label=label, **kwargs)
            dataset = pd.concat([df_luad, df_lusc], axis=0)
        else:
            dataset = load_s36(cohort=cohort, label=label, **kwargs)
        drop_na_columns = ["label"]
        if load_slide:
            drop_na_columns += ["slide_path"]
            drop_na_columns += ["coords_path"]
        else:
            dataset["slide_path"] = None

    elif "NLST" in cohort:
        if cohort == "NLST_NSCLC":
            df_luad = load_nlst(cohort="NLST_LUAD", label=label, **kwargs)
            df_lusc = load_nlst(cohort="NLST_LUSC", label=label, **kwargs)
            dataset = pd.concat([df_luad, df_lusc], axis=0)
        else:
            dataset = load_nlst(cohort=cohort, label=label, **kwargs)
        drop_na_columns = ["label"]
        if load_slide:
            drop_na_columns += ["slide_path"]
            drop_na_columns += ["coords_path"]
        else:
            dataset["slide_path"] = None

    #print(dataset.head())
    dataset = dataset.dropna(subset=drop_na_columns, inplace=False)
    print("Dataset final shape", dataset.shape)
    return dataset


class SlideClassificationDataset(SlideFeaturesDataset):
    """Data loader for slide-classification downstream experiments based on
    ``SlideFeaturesDataset`` module. See ``load_dataset`` above function for
    a detailed documentation. Contrarily to slide-level tasks, data loading for
    classification (i.e. linear evaluation) is handled directly in
    ``rl_benchmarks/tools/tile_level_tasks/linear_evaluation.py``.

    Parameters
    ----------
    features_root_dir: Union[str, Path]
        Path to the histology features' root directory e.g.
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/CAMELYON16_FULL/
    cohort: str = None
        Name of the cohort, e.g ``'TCGA_READ'`` or ``'CAMELYON16_FULL'``.
    label: str = None
        Only needed for TCGA cohorts.
    n_tiles: int = 1000
        Number of tiles per slide.
    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
        This parameter is automatically picked up during feature extraction
        depending on the feature extractor at stake.
        See ``rl_benchmarks.constants.TILE_SIZES``.
    """

    def __init__(
        self,
        features_root_dir: Union[str, Path],
        cohort: str = None,
        label: str = None,
        n_tiles: int = 1_000,
        tile_size: int = 224,
        shuffle: bool = False,
    ):
        print("d1 load, tiles, shuffle:", n_tiles, shuffle)
        dataset = load_dataset(
            cohort=cohort,
            features_root_dir=features_root_dir,
            label=label,
            tile_size=tile_size,
            load_slide=False,
        )
        dataset = dataset[~dataset["features_path"].isna()].reset_index(
            drop=True
        )
        print("Final dataset shape", dataset.shape)
        if dataset.shape[0] == 0:
            log_cohort = f"TCGA/{cohort}" if "TCGA" in cohort else cohort
            raise AttributeError(
                f"No features exist at {str(features_root_dir)} for {log_cohort}."
            )
        # Preload features
        features = dataset["features_path"].values
        features, indices = preload_features(
            fpaths=features,
            n_tiles=n_tiles,
            shuffle=shuffle,
        )
        self.dataset = dataset.iloc[indices]
        labels = self.dataset.iloc[indices].label.values
        ids = [str(i).split("/")[-2] for i in dataset["features_path"]]
        super().__init__(features, labels=labels, ids=ids, n_tiles=n_tiles)

        self.patient_id = self.dataset.patient_id.values
        self.center_id = self.dataset.center_id.values
        self.n_slides = self.dataset.slide_id.nunique()
        self.n_patients = self.dataset.patient_id.nunique()
        self.n_features = features[0].shape[-1] - 3
        self.n_labels, self.stratified = self._set_stratification(
            cohort=cohort, label=label
        )
        self.slide_path = dataset["slide_path"]
        self.cohort = cohort

    def _set_stratification(
        self, cohort: str, label: str
    ) -> Tuple[int, np.array]:
        """Get number of classes and frequencies for stratification.
        For overall survival prediction, number of events serve as the
        stratification reference."""
        if "CAMELYON16" in cohort:
            n_labels = 1  # tumor presence (1) vs absence (0)
            stratified = self.dataset.label.values
        elif "TCGA" in cohort:
            if label in ["OS", "MSI", "HRD", "DSS", "my_survival"]:
                n_labels = 1
            elif label=="dataset":
                print("Setting n_labels to 3 in rl_../datasets/init")
                n_labels = 3
            elif label=="STAGE":
                n_labels = 1
            elif label == "CANCER_SUBTYPE":
                if cohort in ["TCGA_CRC", "TCGA_NSCLC"]:
                    n_labels = 1
                elif cohort == "TCGA_RCC":
                    n_labels = 3
                else:
                    n_labels = self.dataset.label.nunique()
            elif label == "HISTOLOGICAL_SUBTYPE":
                if cohort == "TCGA_BRCA":
                    n_labels = 1
                else:
                    n_labels = self.dataset.label.nunique()
            elif label == "MOLECULAR_SUBTYPE":
                if cohort == "TCGA_BRCA":
                    n_labels = 5
                else:
                    n_labels = self.dataset.label.nunique()
            else:
                n_labels = self.dataset.label.nunique()

            # Store statistics for further stratification in cross-validation.
            if label in ["OS", "DSS"]:
                stratified = 1 * (self.dataset.label.values > 0)
            elif label in [
                "MSI",
                "HRD",
                "CANCER_SUBTYPE",
                "HISTOLOGICAL_SUBTYPE",
                "MOLECULAR_SUBTYPE",
                "my_survival",
                "dataset",
                "STAGE"
            ]:
                stratified = self.dataset.label.values
            else:
                stratified = None
        elif ("UNN" in cohort) or ("S36" in cohort) or ("NLST" in cohort):
            if label in ["OS", "MSI", "HRD", "DSS", "my_survival", "STAGE"]:
                n_labels = 1
            elif label=="dataset":
                n_labels = 3
            elif label == "CANCER_SUBTYPE":
                n_labels = 1
            else:
                n_labels = self.dataset.label.nunique()

            if label in ["OS", "DSS"]:
                stratified = 1 * (self.dataset.label.values > 0)
            elif label in [
                "CANCER_SUBTYPE",
                "HISTOLOGICAL_SUBTYPE",
                "MOLECULAR_SUBTYPE",
                "my_survival",
                "dataset",
                "STAGE"
            ]:
                stratified = self.dataset.label.values
            else:
                stratified = None

        return n_labels, stratified

    def merge_datasets(self, s2):
        assert type(s2) == SlideClassificationDataset, "Both datasets must beof type SlideClassificationDataset."
        assert self.n_tiles == s2.n_tiles, "The datasets trying to merge have different n_tiles"
        assert self.transform == s2.transform, "The datasets trying to merge have different transforms"
        assert self.shuffle == s2.shuffle, "The datasets trying to merge have different shuffle strategies"

        self.dataset = pd.concat([self.dataset, s2.dataset], ignore_index=True)

        self.features = self.features + s2.features
        self.labels = np.append(self.labels, s2.labels)
        self.ids = self.ids + s2.ids       

        self.patient_id = self.dataset.patient_id.values
        self.center_id = self.dataset.center_id.values
        self.n_slides = self.dataset.slide_id.nunique()
        self.n_patients = self.dataset.patient_id.nunique()
        #self.n_features = features[0].shape[-1] - 3
        # self.n_labels, self.stratified = self._set_stratification(
        #     cohort=cohort, label=label
        # )
        self.stratified = np.append(self.stratified, s2.stratified)
        self.slide_path = self.dataset["slide_path"]
        self.cohort = [self.cohort, s2.cohort]