"""ViT architecture with pre-loaded weights from iBOT training."""

from typing import Optional, Tuple

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

import torch
from torchvision import transforms
from loguru import logger

from .encoders import vit_base
from .core import Extractor
from ...constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_WEIGHTS


class UNI(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with UNI

    Parameters
    ----------
    architecture: str = 'vit_base_pancan'
        Model architecture. Must only be "vit_base_pancan" as of now.
    encoder: str = 'teacher'
        Whether to load the weights from the student or teacher encoder.
    mean: Tuple[float, float, float] = IMAGENET_MEAN
        Mean values used for mean/std normalization of image channels.
    std: Tuple[float, float, float] = IMAGENET_STD:
        Std values used for mean/std normalization of image channels.

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(UNI, self).__init__()

        # Load weights for iBOT[ViT-B]PanCancer.
        login(token=token, add_to_git_credential=True)
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.feature_extractor = model


    @property
    def transform(self):
        """Transform method to apply element wise.

        Returns
        -------
        transform: Callable[[Input], Transformed]
        """
        #print("transform function in uni.py")
        t = create_transform(**resolve_data_config(self.feature_extractor.pretrained_cfg, model=self.feature_extractor))
        return t
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute and return features.

        Parameters
        ----------
        images: torch.Tensor
            input of size (n_tiles, n_channels, dim_x, dim_y)

        Returns
        -------
        features : torch.Tensor
            tensor of size (n_tiles, features_dim)
        """
        # get the features from feature_extractor...    
        #images = self.transform(images)  #Don't do this
        with torch.inference_mode():
            features = self.feature_extractor(images)
        # print(f"UNI call worked. Shape {images.shape}, {features.shape}", flush=True)
        return features