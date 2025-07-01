# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""UNN dataset loading for a given cohort and label."""


from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from ...constants import UNN_PATHS, UNN_TASKS


def load_histo(
    features_root_dir: Union[str, Path],
    tile_size: int = 224,
    cohort: str = "TCGA_COAD",
    load_slide: bool = False,
) -> pd.DataFrame:
    """Load TCGA histology data as a pandas dataframe from TCGA portal [1]_.

    Parameters
    ----------
    features_root_dir: Union[str, Path]
        Path to the histology features' root directory, e.g.
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/. The ``cohort``
        argument is used to scan ``features_root_dir / TCGA / cohort`` folder for
        features. If no features have been extracted yet, `features_path`
        is made of NaNs.
    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
    cohort : str
        Name of the TCGA cohort to consider. Valid TCGA cohorts are:``'TCGA_COAD'`` (colon
        adenocarcinoma), ``'TCGA_READ'`` (rectum adenocarcinoma), ``'TCGA_LUAD'`` (lung
        adenocarcinoma), ``'TCGA_LUSC'`` (lung squamous cell carcinoma), ``'TCGA_BRCA'``
        (breast invasive carcinoma), ``'TCGA_KIRC'`` (kidney renal clear cell carcinoma),
        ``'TCGA_KIRP'`` (kidney renal papillary cell carcinoma), ``'TCGA_KICH'`` (kidney
        chromophobe), ``'TCGA_OV'`` (ovarian serous cystadenocarcinoma), ``'TCGA_MESO'``
        (mesothelioma), ``'TCGA_PAAD'`` (pancreatic adenocarcinoma), ``'TCGA_PRAD'``
        (prostate adenocarcinoma).
    load_slide: bool = False
        Add slides paths if those are needed. This parameter should be set
        to ``False`` if slides paths are not needed, i.e. for downstream tasks
        as only features matter, or ``True`` for features extraction (features
        have not been generated from slides yet).

    Returns
    -------
    dataset: pd.DataFrame
        This dataset contains the following columns:
        "patient_id": TCGA patient ID
        "slide_id": TCGA slide ID
        "slide_path": path to the slide
        "coords_path": path to the coordinates
        "label": outcome to predict

    References
    ----------
    .. [1] TCGA Research Network: https://www.cancer.gov/tcga.
    """
    # Get paths
    slides_root_dir = Path(UNN_PATHS["SLIDES"]) #/ cohort.replace("_", "-")
    coords_root_dir = Path(UNN_PATHS["COORDS"](tile_size)) / cohort.split("_")[1]

    print("Slides root", slides_root_dir)
    # Get slides paths
    spaths = list(slides_root_dir.glob("*/*.svs"))
    df_slides = pd.DataFrame({"slide_path": spaths})
    df_slides["slide_id"] = df_slides.slide_path.apply(lambda x: x.name[:-4]) #str(int(  .split("-")[1].split("_")[0])))
    df_slides = df_slides[["slide_id", "slide_path"]]

    # Get paths of tiles coordinates arrays
    print("Coords root", coords_root_dir)
    cpaths = list(coords_root_dir.glob("*.svs/coords.npy"))
    df_coords = pd.DataFrame({"coords_path": cpaths})
    df_coords["slide_id"] = df_coords.coords_path.apply(
        lambda x: x.parent.name[:-4]
    )
    df_coords = df_coords[["slide_id", "coords_path"]]

    # Drop the slides that dont have coordinates (UNN slides is not divided between LUAD/LUSC)
    df_slides = df_slides[df_slides["slide_id"].isin(list(df_coords["slide_id"]))]

    # Get histology feature paths (if available).
    features_root_dir = features_root_dir / cohort
    fpaths = list(features_root_dir.glob("*.svs/features.npy"))
    df_features = pd.DataFrame({"features_path": fpaths})
    df_features["slide_id"] = df_features.features_path.apply(
        lambda x: x.parent.name[:-4]
    )
    df_features = df_features[["slide_id", "features_path"]]
    print("Feature dir", features_root_dir)
    print(f"Found {len(spaths)} slides, {len(cpaths)} coordinates, and {len(fpaths)} features.")

    # Merge dataframes
    if load_slide:
        df_histo = pd.merge(
            left=df_slides,
            right=df_coords,
            on=["slide_id"],
            how="outer",
            sort=False,
        )
    else:
        df_histo = df_coords.copy()
    df_histo = pd.merge(
        left=df_histo,
        right=df_features,
        on=["slide_id"],
        how="outer",
        sort=False,
    )
    df_histo["patient_id"] = df_histo.slide_id.apply(lambda x: x[4:8]) # :12

    return df_histo


def load_label(
    cohort: str,
    label: str,
) -> pd.DataFrame:
    """Load TCGA clinical data for a given cohort and outcome.
    Parameters
    ----------
    cohort: str
        Name of the TCGA cohort, e.g. ``'TCGA_COAD'``, ``'TCGA_LUAD'``, etc.
    label: str
        The task-specific label, e.g. ``'CANCER_SUBTYPE'``.

    Returns
    -------
    dataset: pd.DataFrame
        This dataset contains the following columns:
        "patient_id": TCGA patient ID
        "label": outcome to predict
    """

    def _load_clin(
        labels_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Load clinical data (useful for OS prediction)."""
        # Load data
        df_clin = pd.read_csv(labels_path)

        # Define survival outcomes
        df_clin["death"] = df_clin.apply(
            lambda row: row["days_to_death"] if row["death"] == 1 else -row["days_to_death"],
            axis=1,
        )
        # df_clin["PFI"] = df_clin.apply(
        #     lambda row: row["PFI.time"]
        #     if row["PFI"] == 1
        #     else -row["PFI.time"],
        #     axis=1,
        # )
        df_clin["cancer_specific_death"] = df_clin.apply(
            lambda row: row["days_to_death"] if row["cancer_specific_death"] == 1 else -row["days_to_death"],
            axis=1,
        )

        # Get proper names for main clinical variables
        to_rename = {
            "patient_id": "patient_id",
            # "age_at_initial_pathologic_diagnosis": "AGE",
            # "gender": "GENDER",
            # "race": "ETHNICITY",
            "p_stage": "STAGE",
            # "histological_grade": "GRADE",
            "death": "OS",
            "cancer_specific_death": "DSS",
        }
        df_clin = df_clin.rename(columns=to_rename)

        # Replace invalid values by NaN
        df_clin = df_clin.replace("[Not Available]", None)

        # Retrieve center
        #df_clin["CENTER"] = df_clin.patient_id.apply(lambda x: x.split("-")[1])
        
        def nr_to_str(n):
            s = str(n)
            return '0'*(4-len(s)) + s
        
        df_clin["patient_tag"] = df_clin.patient_id.apply(nr_to_str)

        # Only keep relevant columns
        to_keep = [
            "patient_id",
            "patient_tag",
            # "AGE",
            # "GENDER",
            # "ETHNICITY",
            "STAGE",
            # "GRADE",
            "OS",
            #"PFI",
            "DSS",
            "is_distinct_outcome"
        ]

        if "my_survival" in df_clin:
            to_keep.append("my_survival")
            
        df_clin = df_clin[to_keep]

        return df_clin

    def _load_msi(labels_path: Union[str, Path]) -> pd.DataFrame:
        """Load labels related to MSI prediction."""
        df_labels = pd.read_csv(labels_path, names=["patient_id", "label"])
        df_labels = df_labels[(df_labels.label != "Indeterminate")]
        df_labels["MSI"] = df_labels.label.apply(lambda l: float(l == "MSI-H"))
        df_labels = df_labels[["patient_id", "MSI"]]
        return df_labels

    def _load_hrd(
        labels_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Load labels related to HRD prediction."""
        df_labels = pd.read_csv(labels_path)
        df_labels = df_labels[["patient_id", "HRD"]]
        return df_labels

    def _load_subtypes(labels_path: str) -> pd.DataFrame:
        """Load labels related to molecular or histological subtypes prediction."""
        df_subtypes = pd.read_csv(labels_path, sep="\t")

        to_rename = {
            # "Patient ID": "patient_id",
            # "Sample ID": "sample_id",
            # "TCGA PanCanAtlas Cancer Type Acronym": "cohort",
            # "Subtype": "MOLECULAR_SUBTYPE",
            # "Cancer Type Detailed": "HISTOLOGICAL_SUBTYPE",
            # "Tumor Type": "TUMOR_TYPE",
            "histological_type": "HISTOLOGICAL_SUBTYPE",
        }
        df_subtypes.rename(columns=to_rename, inplace=True)

        to_keep = [
            "patient_id",
            #"cohort",
            #"MOLECULAR_SUBTYPE",
            "HISTOLOGICAL_SUBTYPE",
            #"TUMOR_TYPE",
        ]
        df_subtypes = df_subtypes[to_keep]
        df_subtypes.drop_duplicates(keep="first", inplace=True)

        return df_subtypes

    if label is None:
        df_labels = None
    # Get the dataframe corresponding to the input outcome to be predicted,
    # starting with survival
    elif label in ["OS", "PFI", "DSS", "my_survival"]:
        labels_path = Path(UNN_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)

    elif label=="dataset":
        labels_path = Path(UNN_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)
        df_labels["dataset"] = 1
    
    elif label=="STAGE":
        labels_path = Path(UNN_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)

        #convert_labels = {'III':'III', 'IIIA':'III', 'IIB':'IIB', 'IIIC':'III', 'IA':'IA', 'II':'II', 'IVA':'IV', 'I':'I', 'IB':'IB', '[DISCREPANCY]':'other', 'IIIB':'III', 'IV':'IV', 'IIA':'IIA', 'IVA':'IV', 'IVB':'IV', 'IIIC':'III'}
        convert_labels = {'Stage III':'III', 'Stage IIIa':'III', 'Stage IIb':'II', 'Stage IIIC':'III', 'Stage Ia':'I', 'Stage II':'II', 'Stage IVa':'IV', 'Stage I':'I', 'Stage Ib':'I', 'Stage IIIb':'III', 'Stage IV':'IV', 'Stage IIa':'II', 'Stage IVa':'IV', 'Stage IVb':'IV', 'Stage IIIC':'III'}
        conv_p2 = {'I':0, 'II':0, 'III':1, 'IV':1}
        df_labels = df_labels.replace({"STAGE": convert_labels})
        df_labels = df_labels.replace({"STAGE": conv_p2})
        print(set(df_labels["STAGE"]))

    # Cancer subtype prediction
    elif label == "CANCER_SUBTYPE":
        labels_path = Path(UNN_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)
        tcga_cohort_dict = {
            # NSCLC
            "UNN_LUAD": 0,
            "UNN_LUSC": 1,
            # CRC
            "UNN_COAD": 0,
            "UNN_READ": 1,
            # RCC
            "UNN_KIRC": 0,
            "UNN_KIRP": 1,
            "UNN_KICH": 2,
        }
        df_labels["CANCER_SUBTYPE"] = [tcga_cohort_dict[cohort]] * len(
            df_labels
        )

    # Molecular and histological subtypes, plus tumor type
    elif label in ["MOLECULAR_SUBTYPE", "HISTOLOGICAL_SUBTYPE", "TUMOR_TYPE"]:
        labels_path = Path(UNN_PATHS["LABELS"]["SUBTYPES"](cohort))
        df_labels = _load_subtypes(labels_path=labels_path)

        # this is specific to TCGA-BRCA
        # if (cohort == "TCGA_BRCA") and (label in ["HISTOLOGICAL_SUBTYPE"]):
        #     df_labels = df_labels[
        #         df_labels[label].isin(
        #             [
        #                 "Breast Invasive Lobular Carcinoma",
        #                 "Breast Invasive Ductal Carcinoma",
        #             ]
        #         )
        #     ]
        encoder = LabelEncoder().fit(
            df_labels.loc[~(df_labels[label].isna()), label]
        )
        df_labels.loc[~(df_labels[label].isna()), label] = encoder.transform(
            df_labels.loc[~(df_labels[label].isna()), label]
        )
        logger.info(f"Labels encoded: {encoder.classes_}")

    else:
        raise NotImplementedError
    
    def nr_to_str(n):
        s = str(n)
        return '0'*(4-len(s)) + s

    df_labels = df_labels.drop_duplicates()
    df_labels["label"] = df_labels[label].astype(float)
    df_labels["patient_id"] = df_labels["patient_id"].apply(nr_to_str)
    #df_labels["patient_id"] = df_labels["patient_id"].astype("string")
    df_labels = df_labels[["patient_id", "label"]]

    return df_labels


def load_unn(
    features_root_dir: Union[str, Path],
    tile_size: int = 224,
    cohort: str = "TCGA_COAD",
    label: str = "OS",
    load_slide: bool = False,
) -> pd.DataFrame:
    """Load task-specific data from TCGA, task being fully specified
    by the cohort and label.

    Parameters
    ----------
    features_root_dir: Union[str, Path]
        Path to the histology features' root directory, e.g.
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/. The ``cohort``
        argument is used to scan ``features_root_dir / TCGA / cohort`` folder for
        features. If no features have been extracted yet, `features_path`
        is made of NaNs.
    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
    cohort : str = "TCGA_COAD"
        Name of the TCGA cohort to consider. Valid TCGA cohorts are: ``'COAD'`` (colon
        adenocarcinoma), ``'READ'`` (rectum adenocarcinoma), ``'LUAD'`` (lung
        adenocarcinoma), ``'LUSC'`` (lung squamous cell carcinoma), ``'BRCA'``
        (breast invasive carcinoma), ``'KIRC'`` (kidney renal clear cell carcinoma),
        ``'KIRP'`` (kidney renal papillary cell carcinoma), ``'KICH'`` (kidney
        chromophobe), ``'OV'`` (ovarian serous cystadenocarcinoma), ``'MESO'``
        (mesothelioma), ``'PAAD'`` (pancreatic adenocarcinoma), ``'PRAD'``
        (prostate adenocarcinoma).
    label: str = "OS"
        The task-specific label. Valid labels are: ``'MOLECULAR_SUBTYPE'``,
        ``'HISTOLOGICAL_SUBTYPE'``, ``'TUMOR_TYPE'``, ``'CANCER_SUBTYPE'``,
        ``'SURVIVAL'``, ``'MSI'`` and ``'HRD'``.
    load_slide: bool = False
        Add slides paths if those are needed. This parameter should be set
        to ``False`` if slides paths are not needed, i.e. for downstream tasks
        as only features matter, or ``True`` for features extraction (features
        have not been generated from slides yet).

    Returns
    -------
    dataset: pd.DataFrame
        This dataset contains the following columns:
        "patient_id": TCGA patient ID
        "slide_id": TCGA slide ID
        "slide_path": path to the slide (optional, only if ``load_slide`` is True)
        "coords_path": path to the coordinates
        "label": outcome to predict (optional, only if ``label`` is not None)
        "center_id": TCGA tissue source site ID

    Raises
    ------
    NotImplementedError
        If the tuple (``label``, ``cohort``) refers to a task that has not been
        implemented.
    """
    assert (
        label in UNN_TASKS[cohort]
    ), f"Choose a label in {UNN_TASKS[cohort]} for cohort {cohort}. Provided {label}."
    dataset = load_histo(
        cohort=cohort,
        features_root_dir=features_root_dir,
        tile_size=tile_size,
        load_slide=load_slide,
    )
    dataset["patient_id"] = dataset["patient_id"].astype("string")

    if label is not None:
        df_labels = load_label(cohort=cohort, label=label)
        print(f"Labels shape {df_labels.shape}")
        # Remove patient ids that dont have coordinates
        df_labels = df_labels[df_labels["patient_id"].isin(list(dataset["patient_id"]))]

        if label=="my_survival":
            # Drop patients that are not distinct.
            df_labels.dropna(subset=["label"], inplace=True)
        
        dataset = pd.merge(
            left=df_labels,
            right=dataset,
            on=["patient_id"],
            how="outer",
            sort=False,
        )
    else:
        dataset["label"] = "NA"
    dataset["center_id"] = '0' # apply lambda row: row.patient_id.split("-")[1], axis=1

    return dataset
