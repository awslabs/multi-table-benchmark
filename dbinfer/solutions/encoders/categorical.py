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

@feat_encoder(DBBColumnDType.category_t)
class CategoricalEncoder(BaseFeatEncoder):

    def __init__(
        self,
        config : Dict[str, Any],
        out_size : Optional[int] = None
    ):
        num_categories = config['num_categories']
        if out_size is None:
            # Decide the embedding size based on rules.
            # Borrowed from https://github.com/fastai/fastai/blob/master/fastai/tabular/model.py#L12
            out_size = int(min(128, 1.6 * num_categories ** 0.56))
        super().__init__(config, out_size)
        self.embed = nn.Embedding(num_categories, out_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, input_feat : torch.Tensor) -> torch.Tensor:
        return self.embed(input_feat.view(-1))
