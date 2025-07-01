"""ViT architecture with pre-loaded weights from iBOT training."""

from typing import Optional, Tuple

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from huggingface_hub import login

import torch
from torchvision import transforms
from loguru import logger

from .encoders import vit_base
from .core import Extractor
from ...constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_WEIGHTS, HF_TOKEN


class Virchow(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with Virchow

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
        super(Virchow, self).__init__()

        # Load weights for iBOT[ViT-B]PanCancer.
        login(token=HF_TOKEN, add_to_git_credential=True)
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        
        #model.eval()
        # (Alternatively, download the model weights and get them from local storage...)
        
        self.feature_extractor = model


    @property
    def transform(self):
        """Transform method to apply element wise.

        Returns
        -------
        transform: Callable[[Input], Transformed]
        """
        #print("transform function in virchow.py")
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
        # print("__Images input to call", images.shape)

        #images = self.transform(images)  #NO! Transform is already applied!
        #print("__Images after transform", images.shape, flush=True)
        #print("__alt", images.unsqueeze(0).shape, flush=True) 
        with torch.inference_mode():
            features = self.feature_extractor(images) # Output has feature_dim shape 257x1280
        #print("Virchow model has been applied and output has shape (n_tiles, 257, 1280)", features.shape, flush=True)
        class_token = features[:,0]     # Shape 1x1280
        patch_tokens = features[:,1:]   # Shape 256x1280
        #print("mean of patch_tokens has shape", patch_tokens.mean(1).shape, "(should be tiles,1280)")
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)# should have size 1x2560
        #print("Final embedding has size", embedding.shape, "expected tiles,2560.", flush=True)
        return embedding