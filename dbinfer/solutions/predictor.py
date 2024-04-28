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


from enum import Enum
from typing import Tuple, Dict, Optional, List, Any
import torch
import torch.nn as nn
import pydantic
from dbinfer_bench import DBBTaskType

from .graph_dataset_config import TaskConfig

class PredictorConfig(pydantic.BaseModel):
    num_layers : Optional[int] = 1
    hid_size : Optional[int] = 128
    dropout : Optional[float] = 0.3

class Predictor(nn.Module):
    def __init__(
        self,
        task_config : TaskConfig,
        predictor_config : PredictorConfig,
        seed_embed_size : int,
        seed_ctx_embed_size : int
    ):
        super().__init__()
        self.task_config = task_config
        self.predictor_config = predictor_config

        in_size = seed_embed_size * task_config.num_seeds + seed_ctx_embed_size
        out_size = self.get_out_size(task_config)

        if predictor_config.num_layers == 1:
            # Use one linear layer.
            self.model = nn.Linear(in_size, out_size)

        else:
            # Use an MLP.
            self.model = nn.Sequential(
                nn.Linear(in_size, predictor_config.hid_size),
                nn.ReLU(),
                nn.Dropout(self.predictor_config.dropout)
            )
            for i in range(predictor_config.num_layers - 1):
                self.model += nn.Sequential(
                    nn.Linear(predictor_config.hid_size, predictor_config.hid_size),
                    nn.ReLU(),
                    nn.Dropout(self.predictor_config.dropout)
                )
            self.model.append(nn.Linear(predictor_config.hid_size, out_size))

    def forward(self, seed_embeds, seed_ctx_embeds):
        """Forward

        Input shape
          seed_embeds : (N, K1, D1) or (N, D1)
          (optional) seed_ctx_embeds : (N, D2)

        Output logits or target of shape
          ret: (N, C). If C == 1, shape (N, )
        """
        N = seed_embeds.shape[0]
        embeds = seed_embeds.view(N, -1)
        if seed_ctx_embeds is not None:
            embeds = torch.cat([embeds, seed_ctx_embeds], dim=1)
        return self.model(embeds).squeeze(-1)

    @staticmethod
    def get_out_size(task_config : TaskConfig) -> int:
        if task_config.task_type == DBBTaskType.classification:
            out_size = task_config.num_classes
        elif task_config.task_type == DBBTaskType.regression:
            out_size = 1
        elif task_config.task_type == DBBTaskType.retrieval:
            out_size = 1
        else:
            raise ValueError(f"Unsupported task type {task_config.task_type}.")
        return out_size

class SeedLookup(nn.Module):
    def __init__(self, target_type):
        super().__init__()
        self.target_type = target_type

    def forward(
        self,
        node_embed_dict : Dict[str, torch.Tensor],
        seed_lookup_idx : torch.Tensor,
    ) -> torch.Tensor:
        """Look up seed embeddings from node embeddings.

        Input shape:
            node_embed_dict : each item is of shape (M_t, D)
            seed_lookup_idx : (N, K) or None
                None means identity.

        Output shape:
            seed_embeds : (N, K, D)
        """
        if len(self.target_type.split(":")) != 3:
            # Node-level prediction.
            node_embed = node_embed_dict[self.target_type]
            if seed_lookup_idx is None:
                return node_embed
            else:
                return node_embed[seed_lookup_idx]
        else:
            # Edge-level prediction.
            src_type, _, dst_type = self.target_type.split(":")
            src_embed = node_embed_dict[src_type]
            dst_embed = node_embed_dict[dst_type]
            if seed_lookup_idx is None:
                src_seed_embed = src_embed
                dst_seed_embed = dst_embed
            else:
                src_seed_embed = src_embed[seed_lookup_idx[:,0]]
                dst_seed_embed = dst_embed[seed_lookup_idx[:,1]]
            return torch.stack([src_seed_embed, dst_seed_embed], dim=1)
