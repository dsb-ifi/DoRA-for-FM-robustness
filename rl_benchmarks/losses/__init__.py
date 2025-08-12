# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module covering losses for slide-level downstream tasks."""

from .bce_with_logits_loss import BCEWithLogitsLoss
from .cox_loss import CoxLoss
from .cross_entropy_loss import CrossEntropyLoss
from .mse_loss import MeanSquaredErrorLoss
from .mmd_loss import MMD
from .entropy_loss import Joint_entropy

# OS prediction.
SURVIVAL_LOSSES = (CoxLoss, MeanSquaredErrorLoss)
# Binary and multi-categorial outcome prediction, respectively.
CLASSIFICATION_LOSSES = (BCEWithLogitsLoss, CrossEntropyLoss)
