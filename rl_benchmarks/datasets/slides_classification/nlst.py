# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NLST dataset loading for a given cohort and label."""


from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from ...constants import NLST_PATHS, NLST_TASKS


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
    slides_root_dir = Path(NLST_PATHS["SLIDES"]) #/ cohort.replace("_", "-")
    coords_root_dir = Path(NLST_PATHS["COORDS"](tile_size)) / cohort.split("_")[1]

    print("Slides root", slides_root_dir)
    # Get slides paths
    spaths = list(slides_root_dir.glob("*/*.svs"))    
    df_slides = pd.DataFrame({"slide_path": spaths})
    df_slides["slide_id"] = df_slides.slide_path.apply(lambda x: x.name[:-4]) 
    df_slides = df_slides[["slide_id", "slide_path"]]
    df_slides["slide_id_full"] = df_slides.slide_path.apply(lambda x: "_".join(str(x).split("/")[-2:])[:-4])
    df_slides["slide_id_labels"] = df_slides["slide_id_full"].copy().apply(lambda x: x.split("_")[0])

    # if load_slide:
    #     slide_to_patient = {}
    #     sd = list(df_slides["slide_id_full"])
    #     for s in sd:
    #         slide_to_patient[s.split("_")[1]] = s.split("_")[0]

    # Get paths of tiles coordinates arrays
    print("Coords root", coords_root_dir)
    cpaths = list(coords_root_dir.glob("*.svs/coords.npy"))
    df_coords = pd.DataFrame({"coords_path": cpaths})
    df_coords["slide_id"] = df_coords.coords_path.apply(lambda x: x.parent.name[:-4]) # pid_slide_id
    df_coords = df_coords[["slide_id", "coords_path"]]

    # Drop the slides that dont have coordinates (NLST slides is not divided between LUAD/LUSC)
    df_slides = df_slides[df_slides["slide_id"].isin(list(df_coords["slide_id"]))]

    # Get histology feature paths (if available).
    features_root_dir = features_root_dir / cohort
    fpaths = list(features_root_dir.glob("*.svs/features.npy"))
    df_features = pd.DataFrame({"features_path": fpaths})
    df_features["slide_id"] = df_features.features_path.apply(
        lambda x: x.parent.name[:-4].split("_")[1]
    )
    df_features = df_features[["slide_id", "features_path"]]
    print("Feature dir", features_root_dir)
    print(f"Found {len(spaths)} slides, {len(cpaths)} coordinates, and {len(fpaths)} features.")

    # Merge dataframes
    # Need to load slides
    df_histo = pd.merge(
        left=df_slides,
        right=df_coords,
        on=["slide_id"],
        how="outer",
        sort=False,
    )
    # else:
    #     df_histo = df_coords.copy()
    df_histo = pd.merge(
        left=df_histo,
        right=df_features,
        on=["slide_id"],
        how="outer",
        sort=False,
    )

    # if load_slide:
    #     df_histo["patient_id"] = df_histo["slide_id"].copy().map(slide_to_patient)
    # else:
    df_histo["patient_id"] = df_histo["slide_id"].copy()    #.map(slide_to_patient)

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
        df_clin["OS"] = df_clin.apply(
            lambda row: row["OS.time"] if row["OS"] == 1 else -row["OS.time"],
            axis=1,
        )
        # df_clin["PFI"] = df_clin.apply(
        #     lambda row: row["PFI.time"]
        #     if row["PFI"] == 1
        #     else -row["PFI.time"],
        #     axis=1,
        # )
        df_clin["DSS"] = df_clin.apply(
            lambda row: row["DSS.time"] if row["DSS"] == 1 else -row["DSS.time"],
            axis=1,
        )

        # Get proper names for main clinical variables
        to_rename = {
            "pid": "patient_id",
            # "age_at_initial_pathologic_diagnosis": "AGE",
            # "gender": "GENDER",
            # "race": "ETHNICITY",
            "Stage": "STAGE",
            # "histological_grade": "GRADE",
            "OS": "OS",
            "DSS": "DSS",
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
        labels_path = Path(NLST_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)

    elif label=="dataset":
        labels_path = Path(NLST_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)
        df_labels["dataset"] = 3

    elif label=="STAGE":
        labels_path = Path(NLST_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)

        conv_p2 = {'I':0, 'II':0, 'III':1, 'IIII':1}
        df_labels["STAGE"] = df_labels["STAGE"].apply(lambda x: int(str(x)[0]))
        df_labels["STAGE"] = df_labels["STAGE"].apply(lambda x: x*'I')
        df_labels = df_labels.replace({"STAGE": conv_p2})
        #print(set(df_labels["STAGE"]))

    # Cancer subtype prediction
    elif label == "CANCER_SUBTYPE":
        labels_path = Path(NLST_PATHS["LABELS"]["SURVIVAL"](cohort))
        df_labels = _load_clin(labels_path=labels_path)
        tcga_cohort_dict = {
            # NSCLC
            "TCGA_LUAD": 0,
            "TCGA_LUSC": 1,
            # CRC
            "TCGA_COAD": 0,
            "TCGA_READ": 1,
            # RCC
            "TCGA_KIRC": 0,
            "TCGA_KIRP": 1,
            "TCGA_KICH": 2,
        }
        df_labels["CANCER_SUBTYPE"] = [tcga_cohort_dict[cohort]] * len(
            df_labels
        )

    # Molecular and histological subtypes, plus tumor type
    elif label in ["MOLECULAR_SUBTYPE", "HISTOLOGICAL_SUBTYPE", "TUMOR_TYPE"]:
        labels_path = Path(NLST_PATHS["LABELS"]["SUBTYPES"](cohort))
        df_labels = _load_subtypes(labels_path=labels_path)

        # this is specific to TCGA-BRCA
        if (cohort == "TCGA_BRCA") and (label in ["HISTOLOGICAL_SUBTYPE"]):
            df_labels = df_labels[
                df_labels[label].isin(
                    [
                        "Breast Invasive Lobular Carcinoma",
                        "Breast Invasive Ductal Carcinoma",
                    ]
                )
            ]
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


def load_nlst(
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
        label in NLST_TASKS[cohort]
    ), f"Choose a label in {NLST_TASKS[cohort]} for cohort {cohort}. Provided {label}."
    dataset = load_histo(
        cohort=cohort,
        features_root_dir=features_root_dir,
        tile_size=tile_size,
        load_slide=load_slide,
    )
    dataset["patient_id"] = dataset["slide_id_labels"].copy().astype("string")

    if label is not None:
        df_labels = load_label(cohort=cohort, label=label)
        df_labels["patient_id"] = df_labels["patient_id"].astype("string")
        print("labels", df_labels[:10], type(list(df_labels["patient_id"])[0]))
        print(f"Labels shape {df_labels.shape}")
        # #print("100292".isin(dataset["patient_id"]))
        import IPython
        #IPython.embed()
        # Remove patient ids that dont have coordinates
        df_labels = df_labels[df_labels["patient_id"].isin(list(dataset["patient_id"]))]
        print("labels", df_labels[:10])

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
        print(dataset.keys())
        print(dataset.head(2))
    else:
        dataset["label"] = "NA"
    dataset["center_id"] = '0' # apply lambda row: row.patient_id.split("-")[1], axis=1

    return dataset
