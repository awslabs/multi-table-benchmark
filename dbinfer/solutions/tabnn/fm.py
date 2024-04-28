# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


from typing import Tuple, Dict, Optional, List, Any, Union
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPTabNN, MLPTabNNConfig
from . import registry

class FMLayer(nn.Module):
    """Factorization Machine layer.

    Parameters
    ----------
    num_fields : int
        Number of fields of the input.

    Forward Parameters
    ------------------
    x : torch.Tensor
        Input of shape (batch_size, num_fields, embed_size)

    Forward Returns
    ---------------
    y : torch.Tensor
        Output of shape (batch_size, (num_fields * (num_fields - 1))/2)
    """
    def __init__(self, num_fields):
        super().__init__()
        self.num_fields = num_fields
        assert num_fields > 1, 'FM model requires number of fields > 1'

    @property
    def out_size(self):
        return (self.num_fields * (self.num_fields - 1)) // 2

    def forward(self, x):
        inter = torch.matmul(x, x.transpose(-2, -1))
        indices = torch.triu_indices(
            self.num_fields, self.num_fields, 1, device=x.device)
        return inter[:, indices[0], indices[1]].view(inter.size(0), -1)

    def __repr__(self):
        return f"""{self.__class__.__name__}(
    num_fields={self.num_fields},
    out_size={self.out_size}
)"""

DeepFMTabNNConfig = MLPTabNNConfig

@registry.tabnn
class DeepFMTabNN(nn.Module):
    """Deep Factorization Machine."""
    name = "deepfm"
    config_class = DeepFMTabNNConfig
    def __init__(
        self,
        config : DeepFMTabNNConfig,
        num_fields : int,
        field_size : int,
        out_size : int
    ):
        super().__init__()
        self.fm = FMLayer(num_fields)
        self.fm_out = nn.Linear(self.fm.out_size, out_size)
        self.mlp = MLPTabNN(config, num_fields, field_size, out_size)

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        y_fm = self.fm_out(self.fm(X))
        y_mlp = self.mlp(X)
        y = y_fm + y_mlp
        return y
