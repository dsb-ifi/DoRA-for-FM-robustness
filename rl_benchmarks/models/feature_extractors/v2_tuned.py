"""ViT architecture with pre-loaded weights from iBOT training."""

from typing import Optional, Tuple

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from peft import LoKrConfig, LoKrModel

import torch
from torchvision import transforms
from loguru import logger

from .encoders import vit_base
from .core import Extractor
from ...constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_WEIGHTS

class V2_tuned(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with V2_tuned (Virchow2)

    Parameters
    ----------

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(V2_tuned, self).__init__()


    def load_pretrained(self, path, dinov2_tune=False):
        #if not from_lora:
        #    model = ViTModel.from_pretrained(path, add_pooling_layer=False)
        #if from_lora:
        # Load basic model and add the lora stuff, then load state dict
        use_dora = True
        login(token=token, add_to_git_credential=True)
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        embed_dim = 2560
        target_modules=["qkv"]
        print(f"Loading pretrained model. Dora={use_dora}, target_modules={target_modules}")
        
        use_rank = 16
        if "128" in path:
            print("128 in path -> v2 lora rank 128")
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
        features = self.feature_extractor(images) # Output has feature_dim shape 257x1280
        #print("Virchow2 model has been applied and output has shape (n_tiles, 257, 1280)", features.shape, flush=True)
        class_token = features[:,0]     # Shape 1 x 261 x 1280
        # Tokens 1-4 are register tokens, and should be ignored here
        patch_tokens = features[:,5:]   # Shape 256x1280
        #print("mean of patch_tokens has shape", patch_tokens.mean(1).shape, "(should be tiles,1280)")
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)# size 1x2560
        #print("Final embedding has size", embedding.shape, "expected tiles,2560.", flush=True)
        return embedding