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


"""Encoder for IDs."""

from typing import Tuple, Dict, Optional, List, Any
import torch
import torch.nn as nn

__all__ = ['ConstIdEncoder', 'EmbeddingIdEncoder', 'IdDictEncoder']

class ConstIdEncoder(nn.Module):
    def __init__(
        self,
        out_size : Optional[int],
        const_val : Optional[float] = 0.1
    ):
        super().__init__()
        if out_size is None:
            self.out_size = 1
        else:
            self.out_size = out_size
        self.const_val = const_val

    def forward(self, ids : torch.Tensor) -> torch.Tensor:
        return torch.full(
            (len(ids), self.out_size),
            self.const_val,
            device=ids.device
        )

    def __repr__(self):
        return f"""{self.__class__.__name__}(
    out_size={self.out_size},
    const_val={self.const_val}
)"""

class EmbeddingIdEncoder(nn.Module):
    def __init__(
        self,
        capacity : int,
        out_size : Optional[int]
    ):
        super().__init__()
        if out_size is None:
            # Decide the embedding size based on rules.
            # Borrowed from https://github.com/fastai/fastai/blob/master/fastai/tabular/model.py#L12
            self.out_size = int(min(128, 1.6 * capacity ** 0.56))
        else:
            assert out_size > 0
            self.out_size = out_size
        self.embed = nn.Embedding(capacity, self.out_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, ids : torch.Tensor) -> torch.Tensor:
        return self.embed(ids)

class IdDictEncoder(nn.Module):
    def __init__(
        self,
        capacity : Dict[str, int],
        embed_id_types : List[str],
        out_size : Optional[int]
    ):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.embed_id_types = embed_id_types
        for id_type in embed_id_types:
            self.encoders[id_type] = EmbeddingIdEncoder(capacity[id_type], out_size)
        for id_type in capacity:
            if id_type not in self.encoders:
                self.encoders[id_type] = ConstIdEncoder(out_size)

        self.out_size_dict = {
            id_type : encoder.out_size
            for id_type, encoder in self.encoders.items()
        }

    def forward(self, id_dict : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            id_type : self.encoders[id_type](ids)
            for id_type, ids in id_dict.items()
            if id_type in self.encoders
        }

    def get_embedding_dict(self) -> Dict[str, torch.Tensor]:
        return {
            embed_id_type : self.encoders[embed_id_type].embed.weight
            for embed_id_type in self.embed_id_types
        }
