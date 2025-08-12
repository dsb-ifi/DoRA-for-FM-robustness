"""ViT architecture with pre-loaded weights from iBOT training."""

from typing import Optional, Tuple

#import timm
#from timm.data import resolve_data_config
#from timm.data.transforms_factory import create_transform
#from timm.layers import SwiGLUPacked
from huggingface_hub import login
from transformers import  AutoImageProcessor, AutoModel

import torch
from torchvision import transforms
from loguru import logger

from .encoders import vit_base
from .core import Extractor
from ...constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_WEIGHTS


class Phikon2(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with Phikon2

    Parameters
    ----------

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(Phikon2, self).__init__()

        # Load weights for iBOT[ViT-B]PanCancer.
        login(token=token, add_to_git_credential=True)
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        
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
        t = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
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

        with torch.inference_mode():
            outputs = self.feature_extractor(images) # Output has feature_dim shape 
        #print(f"Phikon2 model has been applied and output has shape f{outputs.shape}", flush=True)
        features = outputs.last_hidden_state[:, 0, :]   # Shape 1x1024 
        #print("Phikon features have shape", features.shape, flush=True)
        return features