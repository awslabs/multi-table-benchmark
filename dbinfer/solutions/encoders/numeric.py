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


from typing import List, Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dbinfer_bench import DBBColumnDType

from .base import BaseFeatEncoder, feat_encoder

@feat_encoder(DBBColumnDType.float_t)
class NumericEncoder(BaseFeatEncoder):

    def __init__(
        self,
        config : Dict[str, Any],
        out_size : Optional[int] = None
    ):
        in_size = config['in_size']
        if out_size is None:
            # Do not perform projection.
            out_size = in_size
            self.has_proj = False
        else:
            self.has_proj = True
        super().__init__(config, out_size)
        if self.has_proj:
            self.proj = nn.Linear(in_size, out_size, bias=False)

    def forward(self, input_feat : torch.Tensor) -> torch.Tensor:
        if input_feat.ndim == 1:
            input_feat = input_feat.view(-1, 1)
        if self.has_proj:
            return self.proj(input_feat)
        else:
            return input_feat
