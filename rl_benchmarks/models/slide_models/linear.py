# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LINEAR pooling aggregation algorithm."""

from typing import List, Optional

import torch
from torch import nn

from .utils.attention import GatedAttention
from .utils.mlp import MLP
from .utils.tile_layers import TilesMLP


class LINEAR(nn.Module):
    """Attention-based MIL classification model (See [1]_).

    Example:
        >>> module = LINEAR(in_features=128, out_features=1)
        >>> logits, attention_scores = module(slide, mask=mask)
        >>> attention_scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    d_model_attention: int = 128
        Dimension of attention scores.
    temperature: float = 1.0
        GatedAttention softmax temperature.
    tiles_mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in first MLP.
    mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in last MLP.
    mlp_dropout: Optional[List[float]] = None,
        Dropout rate for last MLP.
    mlp_initial_dropout: Optional[float] = 0.0, Dropout rate applied to input embeddings.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation for last MLP.
    bias: bool = True
        Add bias to the first MLP.
    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 

    References
    ----------
    .. [1] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based
    deep multiple instance learning. In Jennifer Dy and Andreas Krause,
    editors, Proceedings of the 35th International Conference on Machine
    Learning, volume 80 of Proceedings of Machine Learning Research,
    pages 2127–2136. PMLR, 10–15 Jul 2018.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        #d_model_attention: int = 128,
        #temperature: float = 1.0,
        bias: bool = True,
        metadata_cols: int = 0,
    ) -> None:
        super(LINEAR, self).__init__()
        print("Using LINEAR aggregation model")

        self.mlp = torch.nn.Linear(in_features, out_features)
        self.metadata_cols = metadata_cols

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None, shred: Optional[torch.BoolTensor] = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features: torch.Tensor
            (B, N_TILES, D+3)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, OUT_FEATURES), (B, N_TILES)
        """
        # if shred:
        #     features = features[..., self.metadata_cols:]
        
        # if int(mask.sum()) > 0:
        #     # Discard what is masked (just feature=zero), which would ruin the average
        #     # But need to keep the different slides in the batch separate
        #     # This approach is slow but should work
        #     use_feat = torch.empty(features.shape[0], features.shape[-1]).to(features.device)
        #     for batch in range(features.shape[0]):
        #         feat = features[batch,:,:]
        #         m = mask[batch,:,:]
        #         #print("feat", feat.shape)
        #         f = feat[~m.expand_as(feat)]
        #         f = f.view(-1,feat.shape[-1]) # valid_tiles, feature_dim
        #         #print("f", f.shape)
        #         use_feat[batch,:] = f.mean(axis=0) # average over the valid tiles
        #     pooled = use_feat
        # else:
        #     pooled = features.mean(axis=1) # Average out the 1000 tiles
        #print("in linear:", features.shape)
        #import IPython
        #IPython.embed()
        #sys.exit()
        logits = self.mlp(features)
        #print("End logit shape", logits.shape)
        return logits
