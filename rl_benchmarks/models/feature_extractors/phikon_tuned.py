"""ViT architecture with pre-loaded weights from iBOT training."""

from typing import Optional, Tuple

#import timm
#from timm.data import resolve_data_config
#from timm.data.transforms_factory import create_transform
#from timm.layers import SwiGLUPacked
from huggingface_hub import login
from transformers import  AutoImageProcessor, ViTModel
from peft import LoraConfig, get_peft_model

import torch
from torchvision import transforms
from loguru import logger

from .encoders import vit_base
from .core import Extractor
from ...constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_WEIGHTS


class Phikon_tuned(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with Phikon_tuned

    Parameters
    ----------

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(Phikon_tuned, self).__init__()


    def load_pretrained(self, path, dinov2_tune=False):
        #if not from_lora:
        #    model = ViTModel.from_pretrained(path, add_pooling_layer=False)
        #if from_lora:
        # Load basic model and add the lora stuff, then load state dict
        use_dora = True
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        embed_dim = 768 #phikon: 768, phikon2:1024
        target_modules=["query", "key", "value", "projection"]
        print(f"Loading pretrained model. Dora={use_dora}, target_modules={target_modules}")
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
            use_dora=use_dora,
        )
        lora_model = get_peft_model(model, config)
        model_state_dict = torch.load(path+"/teacher_checkpoint.pth")['teacher_backbone']
        #import IPython
        #IPython.embed()
        if dinov2_tune:
            del model_state_dict["base_model.model.embeddings.mask_token"]
        lora_model.load_state_dict(model_state_dict)
        model = lora_model

        self.feature_extractor = model

    @property
    def transform(self):
        """Transform method to apply element wise.

        Returns
        -------
        transform: Callable[[Input], Transformed]
        """
        #print("transform function in virchow.py")
        t = AutoImageProcessor.from_pretrained("owkin/phikon")
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

        outputs = self.feature_extractor(images)
        features = outputs.last_hidden_state[:,0,:] # Shape BS, 768
        #print("Features have shape ", features.size())
        return features