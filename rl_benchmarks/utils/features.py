# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions related to features extraction. Those functions are
essential to run features extraction process, as done in
``"rl_benchmarks/tools/extract_features/"`` scripts."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openslide
import pandas as pd
import torch
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from ..models import Extractor

import torchvision

# import sys
# sys.path.append("/home/vilde/code/Phikon/HistoSSLscaling/rl_benchmarks/utils/")
# from test_aug import *
# import random
# import toml


class TilesMap(Dataset):
    """BaseTilesMap Dataset.
    From a slide given by slide_path, create a map-style PyTorch Dataset object
    that returns n_tiles tiles sampled, either randomly or sequentially.

    This dataset returns PIL.Image samples. In order to be used in addition with
    a `torch.utils.data.DataLoader`, you must transforms the samples to `torch.Tensor`.
    `torchvision.transforms.ToTensor` can by use for this purpose.

    Parameters
    ----------
    slide_path: str
        Path to slide from which to extract tiles and features.
    tissue_coords: np.ndarray
        Coordinates of tiles in matter.
    tiles_level: int
        Level to extract the tiles from.
    tiles_size: int
        Size of the tiles. The returned tiles will be PIL.Image
        object of size (tiles_size, tiles_size, 3)
    n_tiles: Optional[int] = None
        Number of tiles to sample.
    random_sampling: bool = False
        Sample either randomly or sequentially.
    transform: Optional[Callable] = None
        Transformation to apply to the PIL.Image after sampling.
    """

    def __init__(
        self,
        slide_path: Union[Path, str],
        tissue_coords: np.ndarray,
        tiles_level: int,
        tiles_size: int = 224,
        n_tiles: Optional[int] = None,
        random_sampling: bool = False,
        transform: Optional[Callable] = None,
        dinotype: Optional[int] = None,
    ) -> None:
        if n_tiles is None:
            n_tiles = len(tissue_coords)
        else:
            n_tiles = min(n_tiles, len(tissue_coords))

        if tissue_coords.ndim == 3:
            raise ValueError("tissue_coords must have only two dimensions.")

        self.slide_path = Path(slide_path)
        self.n_tiles = n_tiles
        self.transform = transform
        self.dinotype = dinotype
        self.random_sampling = random_sampling

        self.tiles_level = tiles_level
        self.tissue_coords = tissue_coords.astype(int)
        self.tiles_size = [tiles_size, tiles_size]

        self.wsi = openslide.open_slide(self.slide_path)
        self.dz = DeepZoomGenerator(
            self.wsi, tile_size=self.tiles_size[0], overlap=0
        )
        self.build_indices()

    def sample_new_tiles(self) -> None:
        """Permute tile indices to sample new tiles.
        Should be called at the end of every epoch.

        Raises
        ------
        ValueError: samples_new_tiles should only be called if random_sampling is set.
        """
        if not self.random_sampling:
            raise ValueError(
                "samples_new_tiles should only be called if random_sampling is set."
            )

    def build_indices(self) -> None:
        """Build indices for __getitem__ function."""
        if self.random_sampling:
            # Set a seed for reproducibility
            #np.random.seed(0)
            indices = np.random.permutation(
                np.arange(0, len(self.tissue_coords))
            )
            self.indices = indices[: self.n_tiles]
            #print("Picking random tiles:D", indices)
        else:
            print("Not using random tiles...")
            self.indices = np.arange(0, len(self.tissue_coords))[
                : self.n_tiles
            ]

    def __getitem__(self, item: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """Retrieve a tile from a ``openslide.deepzoom.DeepZoomGenerator``
        object, coordinates and tile level (in the DeepZoomGenerator system).

        Returns
        -------
        Tuple[Image.Image, Dict[str, Any]]
            ``tile``: histology tile with shape ``self.tiles_size``
            ``metadata``: dictionary with metadata to fully retrieve the tile.
        """
        # True index of the tile. If random_sampling is False, same index.
        #print("In getitem TILEMAP", flush=True)
        index = self.indices[item]
        coords = self.tissue_coords[index, :]
        tile = self.dz.get_tile(level=int(self.tiles_level), address=coords)

        if tile.size != self.tiles_size:
            # If the tile is on a border, we need to pad it.
            tile = np.array(tile)
            tile = np.pad(
                tile,
                pad_width=(
                    (0, self.tiles_size[0] - tile.shape[0]),
                    (0, self.tiles_size[1] - tile.shape[1]),
                    (0, 0),
                ),
            )
            tile = Image.fromarray(tile)

        if self.transform:
            if self.dinotype:
                tile = self.transform(tile, dinotype=self.dinotype)
            else:
                tile = self.transform(tile)
        
        # if "list" in str(type(tile)):
        #     #print("TILE", type(tile), type(tile[0]), type(tile[0][0]))
        #     #print(len(tile), len(tile[0]), len(tile[1]))

        metadata = {
            "slide_name": self.slide_path.name,
            "coords": coords,
            "level": self.tiles_level,
            "slide_length": len(self),
        }
        if hasattr(self, "label"):
            metadata["label"] = self.label
        #print(metadata)
        #print("Returning a tile of type", type(tile), tile.size())
        return (tile, metadata)

    def __len__(self) -> int:
        """Return number of tiles."""
        return self.n_tiles


def extract_from_slide(
    slide: openslide.OpenSlide,
    level: int,
    coords: np.ndarray,
    feature_extractor: Extractor,
    tile_size: int = 224,
    n_tiles: Optional[int] = None,
    random_sampling: bool = False,
    num_workers: int = 8,
    batch_size: int = 64,
    augmentation: bool = False,
    dinotype: int = 2,
):
    """Use a Feature Extractor to embed tiles sampled from the coordinates.

    Parameters:
        ----------
        slide: openslide.OpenSlide
            Slide to extract.
        level: int
            DeepZoom level.
        coords: np.ndarray
            Array of tile coordinates to extract, shape (N_tiles, 2).
        feature_extractor: Extractor
            A feature extractor.
        tile_size: int = 224
            Tile size (pixels).
        n_tiles: Optional[int] = None
            Number of tiles. If None (default), all tiles are extracted.
        random_sampling: bool = False
            Sample either randomly or sequentially.
        num_workers: int = 8
            Number of workers for the slides torch.utils.data.DataLoader. Useful to parallelize
            reads on several slides at the same time.
        batch_size: int = 64
            Batch size for the extractor.

    Returns
    -------
    features: np.ndarray
        Array of shape (N_TILES, 3 + N_FEATURES) where the 3 first coordinates
        are (``level``, x_coordinate, y_coordinate).
    """
    f_trans = feature_extractor.transform
    #f_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), f_trans])
    if augmentation:
        f_trans = torchvision.transforms.ToTensor()

    #print("transform", f_trans, flush=True)

    tile_map = TilesMap(
        slide,
        tissue_coords=coords,
        tiles_level=level,
        tiles_size=tile_size,
        n_tiles=n_tiles,
        random_sampling=random_sampling,
        transform=f_trans,
    )
    #print("TilesMap ok")    
    dataloader = DataLoader(
        tile_map,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
    )
    #print("dataloader done", flush=True)
    slide_features = []
    i=0
    slide_id = slide.name
    print("Slide ID", slide_id)

    if augmentation:
        from .augmentations import preprocess
    
    for images, metadata in dataloader:
        # The tiles dataset provide metadata on the tiles.
        i+=1
        batch_coords = metadata["coords"]
        batch_levels = metadata["level"]
        import transformers

        if type(images)==transformers.image_processing_base.BatchFeature:
            images = images["pixel_values"][0]

        if augmentation:
            #print("Use augmentations")
            ri2 = "-".join(slide_id.split(".")[0].split("-")[1:3])
            augmented_views = preprocess(images, local_crops_nr=0, global_crops_nr=1, dinotype=dinotype)
            #print("inp shape", images.shape)
            #print("Aug views shape", augmented_views.shape)
            #print("slide ID", ri2)
            
            # # Save one augmented and one original tile
            # org = images[2]
            # a = augmented_views[0][2]
            # if i==1:
            #     from torchvision.utils import save_image
            #     cite = "testUNN"
            #     save_image(org, "augtest/"+cite+"_"+ri2+"_org.png")
            #     save_image(a, "augtest/"+cite+"_"+ri2+"_aug.png")
            #     #import IPython
            #     #IPython.embed()
            #     orgnormP = torchvision.transforms.ToPILImage()(org)
            #     orgnorm = feature_extractor.transform(orgnormP)
            #     save_image(orgnorm, "augtest/"+cite+"_"+ri2+"_orgNorm.png")
            images = augmented_views[0] # Reduce 1. empy dim

        # Extracts the features.
        features = feature_extractor.extract_features_as_numpy(images)
        #print("Extraction done")
        # Concatenate the level, coords and features.
        features = np.concatenate(
            (
                batch_levels.unsqueeze(1).float().numpy(),
                batch_coords.float().numpy(),
                features,
            ),
            axis=1,
        )

        slide_features.append(features)
        if i%20000==0:
            print(f"Processing slide {i}", flush=True)

    return np.concatenate(slide_features, axis=0)


class TileImagesDataset(Dataset):
    """From a panda dataframe `dataset` containing the following information:
    "image_id": image ID
    "image_path": path to the tile
    "center_id": center ID (optional)
    "label": tissue class (0 to 8, NCT-CRC) or presence of tumor (0 or 1, Camelyon17-WILDS),

    create a `torch.utils.data.Dataset` that samples over the tile images and labels.

    Parameters
    ----------
    dataset: pd.DataFrame
        Input dataframe with image ids, paths and corresponding labels.
    size: int = 224
        Tile size (pixels) after resizing.
    transform: Optional[Callable] = None
        Function to be applied to tile images.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        tile_size: int = 224,
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self._tile_size = [tile_size, tile_size]

    def __getitem__(self, item: int) -> Image.Image:
        """Retrieve an image from ``self.dataset``."""
        row = self.dataset.iloc[item]

        image = Image.open(row.image_path).convert("RGB")

        if image.size != self._tile_size:
            image = image.resize(self._tile_size)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        """Returns length of the dataset."""
        return len(self.dataset)


def extract_from_tiles(
    dataset_tiles: pd.DataFrame,
    feature_extractor: Extractor,
    tile_size: int = 224,
    num_workers: int = 8,
    batch_size: int = 64,
) -> Tuple[List[np.array], np.array]:
    """Use a Feature Extractor to embed tiles sampled from the a tiles dataset
    such as Camelyon17-WILDS or NCT-CRC.

    Parameters
    ----------
    dataset_tiles: pd.DataFrame
        Data frame containing the following columns:
            "image_id"   : image ID
            "image_path" : path to the tile
            "center_id"  : center ID (optional)
            "label"      : tissue class (0 to 8, NCT-CRC) or presence of
                           tumor (0 or 1, Camelyon17-WILDS)

    feature_extractor: Extractor
        A feature extractor.
    tile_size: int = 224
        Tile size (pixels).
    num_workers: int = 8
        Number of workers for the tiles torch.utils.data.DataLoader.
        Useful to parallelize reads on several tiles at the same time.
    batch_size: int = 64
        Batch size for the extractor.

    Returns
    -------
    Tuple[List[np.array], np.array]
        List of tiles features (BS, N_FEATURES) and
        corresponding ids (N_TILES_DATASETS,). Length of features list times
        batch size roughly give the size of the tiles dataset.
    """
    tile_map = TileImagesDataset(
        dataset_tiles,
        tile_size=tile_size,
        transform=feature_extractor.transform,
    )

    dataloader = DataLoader(
        tile_map,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
    )
    #print("got dataloader", flush=True)

    tile_features = []
    i=0
    for batch in tqdm(dataloader, total=len(dataloader)):
        #print("new batch in features.py", flush=True)
        features = feature_extractor.extract_features_as_numpy(batch)
        tile_features.append(features)

    return (
        np.concatenate(tile_features),
        dataset_tiles.image_id.values,
    )


def preload_features(
    fpaths: List[Union[str, Path]],
    n_tiles: int = 1_000,
    shuffle: bool = False,
    with_memory_map: bool = True,
) -> List:
    """Preload all features from a list of features paths.

    Parameters
    ----------
    fpaths: List[Union[str, Path]]
        List of features paths or features numpy arrays.
    n_tiles: int = 1_000
        Number of tiles to keep for all slides.
    shuffle: bool = False
        If True, shuffle tiles in the input list ``fpaths``.
    with_memory_map: bool = True
        Use ``mmap_mode='r'`` when loading slides features (recommended).
    """
    features = []
    indices_features = []

    #print("preload_features. Shuffle is", shuffle)
    for i, slide_features in tqdm(enumerate(fpaths), total=len(fpaths)):
        # Using memory map not to load the entire np.array when we
        # only want `n_tiles <= len(slide_features)` tiles' features.
        mmap_mode = "r" if with_memory_map else None
        slide_features = np.load(slide_features, mmap_mode=mmap_mode)

        if n_tiles is not None:
            indices = np.arange(len(slide_features))
            if shuffle:
                # We do not shuffle inplace using `np.random.shuffle(slide_features)`
                # as this will load the whole numpy array, removing all benefits
                # of above `mmap_mode='r'`. Instead we shuffle indices and slice
                # into the numpy array.
                np.random.seed(0)
                np.random.shuffle(indices)

            # Take the desired amount of tiles.
            #print("Slide featrus shape", slide_features.shape)
            if len(slide_features.shape) > 1: # Don't if we have already averaged them out
                indices = indices[:n_tiles]

            # Indexing will make the array contiguous by loading it in RAM.
            slide_features = slide_features[indices]

        else:
            if shuffle:
                # Shuffle inplace
                np.random.seed(0)
                np.random.shuffle(slide_features)

        features.append(slide_features)
        indices_features.append(i)
    #print("preload returns shapes", features[0].shape)
    return features, indices_features


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, Any]],
    batch_first: bool = True,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.BoolTensor, Any]:
    """Pad together sequences of arbitrary lengths.
    Add a mask of the padding to the samples that can later be used
    to ignore padding in activation functions.
    For example, not all slides will have 1000 histopathology tiles available. Pad to load everything together in a batch/loader

    Expected to be used in combination of a torch.utils.datasets.DataLoader.

    Expect the sequences to be padded to be the first one in the sample tuples.
    Others members will be batched using ``torch.utils.data.dataloader.default_collate``.

    Parameters
    ----------
        batch: List[Tuple[torch.Tensor, Any]]
            List of tuples (features, Any). Features have shape (N_slides_tiles, F)
            with ``N_slides_tiles`` being specific to each slide depending on the
            number of extractable tiles in the tissue matter. ``F`` is the feature
            extractor output dimension.
        batch_first: bool = True
            Either return (B, N_TILES, F) or (N_TILES, B, F)
        max_len: Optional[int] = None
            Pre-defined maximum length for elements inside a batch.

    Returns
    -------
        padded_sequences, masks, Any: Tuple[torch.Tensor, torch.BoolTensor, Any]
            - if batch_first: Tuple[(B, N_TILES, F), (B, N_TILES, 1), ...]
            - else: Tuple[(N_TILES, B, F), (N_TILES, B, 1), ...]

            with N_TILES = max_len if max_len is not None
            or N_TILES = max length of the training samples.

    """
    # Does the dataloader handle multiple views of the feature?
    # Then, treat the aug. views in same ways. They will have same n_tiles as std features!
    #print("IN PAD")
    aug_views = False
    other_ind = 1
    if len(batch[0]) > 3:
        aug_views = True
        other_ind = 2

    # Expect the sequences to be the first one in the sample tuples   
    sequences = []
    sequences_aug = []
    others = []
    for sample in batch:
        sequences.append(sample[0])
        others.append(sample[other_ind:])
        if aug_views:
            sequences_aug.append(sample[1])

    if max_len is None:
        max_len = max([s.size(0) for s in sequences])

    # Dim of each feature
    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        padded_dims = (len(sequences), max_len) + trailing_dims
        masks_dims = (len(sequences), max_len, 1)
    else:
        padded_dims = (max_len, len(sequences)) + trailing_dims
        masks_dims = (max_len, len(sequences), 1)

    padded_sequences = sequences[0].data.new(*padded_dims).fill_(0.0)
    if aug_views:
        padded_sequences_aug = sequences_aug[0].data.new(*padded_dims).fill_(0.0)
    masks = torch.ones(*masks_dims, dtype=torch.bool)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            padded_sequences[i, :length, ...] = tensor[:max_len, ...]
            if aug_views:
                padded_sequences_aug[i, :length, ...] = tensor[:max_len, ...]           
            masks[i, :length, ...] = False
        else:
            padded_sequences[:length, i, ...] = tensor[:max_len, ...]
            if aug_views:
                padded_sequences_aug[:length, i, ...] = tensor[:max_len, ...]
            masks[:length, i, ...] = False

    # Batching other members of the tuple using default_collate
    # others has sizes bs, 3, (n_tiles, 1(label), 1(slide_id))
    #print("pad_collate", type(others), len(others), len(others[0]), others[0][0].shape)
    others = default_collate(others)
    #print("In custom collate!:D")

    if aug_views:
        return (padded_sequences, masks, padded_sequences_aug, *others)
    return (padded_sequences, masks, *others)
