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


class UNI2(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with UNI2

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
        super(UNI2, self).__init__()

        # Load weights for iBOT[ViT-B]PanCancer.
        login(token=token, add_to_git_credential=True)

        # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
        timm_kwargs = {
                    'img_size': 224, 
                    'patch_size': 14, 
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5, 
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667*2,
                    'num_classes': 0, 
                    'no_embed_class': True,
                    'mlp_layer': timm.layers.SwiGLUPacked, 
                    'act_layer': torch.nn.SiLU, 
                    'reg_tokens': 8, 
                    'dynamic_img_size': True
                }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)      
        self.feature_extractor = model


    @property
    def transform(self):
        """Transform method to apply element wise.

        Returns
        -------
        transform: Callable[[Input], Transformed]
        """
        print("transform function in UNI2.py")
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
        #print("Input to UNI2 has shape", images.shape)
        features = self.feature_extractor(images) # Output has feature_dim shape 257x1536
        #print("UNI2 model has been applied and output has shape (n_tiles, 257, 1536)", features.shape, flush=True)
        return features