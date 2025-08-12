"""ViT architecture with pre-loaded weights from iBOT training."""

from typing import Optional, Tuple

# import timm
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
# from timm.layers import SwiGLUPacked
from huggingface_hub import login
from transformers import  AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model

import torch
from torchvision import transforms
from loguru import logger

from .encoders import vit_base
from .core import Extractor
from ...constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_WEIGHTS

class P2_tuned(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with P2_tuned (Phikon2)

    Parameters
    ----------

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(P2_tuned, self).__init__()


    def load_pretrained(self, path, dinov2_tune=False):
        #if not from_lora:
        #    model = ViTModel.from_pretrained(path, add_pooling_layer=False)
        #if from_lora:
        # Load basic model and add the lora stuff, then load state dict
        use_dora = True
        login(token=token, add_to_git_credential=True)
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        embed_dim = 1024 #phikon: 768, phikon2:1024
        target_modules=["query", "key", "value"]
        print(f"Loading pretrained p2_tuned model. Dora={use_dora}, target_modules={target_modules}")
        
        use_rank = 16
        if "128" in path:
            print("128 in path -> p2 lora rank 128")
            use_rank = 128
        if "lokr" in path:
            print("Using LoKr setup instead of normal LoRA.")
            config = LoKrConfig(
                task_type="VISION",
                r=use_rank,
                alpha=16,
                target_modules=target_modules,
                module_dropout=0.1,
                rank_dropout=0.1,
                init_weights=True,
            )
        else:
            config = LoraConfig(
                r=use_rank,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
                use_dora=use_dora,
            )

        lora_model = get_peft_model(model, config)
        #map_location=torch.device('cuda:0')
        model_state_dict = torch.load(path+"/teacher_checkpoint.pth", map_location=None)['teacher_backbone']

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
        with torch.inference_mode():
            outputs = self.feature_extractor(images) # Output has feature_dim shape 
        #print(f"Phikon2 model has been applied and output has shape f{outputs.shape}", flush=True)
        features = outputs.last_hidden_state[:, 0, :]   # Shape 1x1024 
        #print("Phikon features have shape", features.shape, flush=True)
        return features