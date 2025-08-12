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
from ...constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_WEIGHTS


class ProvGigaPath(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with ProvGigaPath

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
        super(ProvGigaPath, self).__init__()

        # Load weights for iBOT[ViT-B]PanCancer.
        login(token=token, add_to_git_credential=True)
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = timm.create_model("hf-hub:prov-gigapath/prov-gigapath", pretrained=True)
        
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
        print("transform function in ProvGigaPath.py")
        t = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
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
        features = self.feature_extractor(images) 
        #print("ProvGigaPath model has been applied and output has shape (n_tiles, x, y)", features.shape, flush=True)
        # Use full features or only class token??
        class_token = features[:,0]
        print("features shape", features.shape)
        print("CLS shape", class_token.shape, "evt", class_token.unsqueeze(0).shape, flush=True)
        return features