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


from typing import Dict, Optional, List
import numpy as np
import pydantic
import logging

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import gml_solution
from .base_gml_solution import BaseGMLSolution, BaseGNN, BaseGNNSolutionConfig
from .graph_dataset_config import GraphConfig
from .predictor import PredictorConfig
from .gnn import (
    block_to_homogeneous,
    convert_dstdata_to_dict,
    EdgeHGTConvConfig,
    EdgeHGTConv,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class HGTSolutionConfig(BaseGNNSolutionConfig):
    hid_size: int
    dropout: float
    conv : EdgeHGTConvConfig
    
    # Round up the hidden size to multiples of num_heads here.
    # Otherwise most of num_heads will be forced to 1 in instantiation of
    # EdgeHGTConv during hyperparameter search.
    @pydantic.root_validator
    def roundup_to_num_heads(cls, data):
        hid_size = data['hid_size']
        num_heads = data['conv'].num_heads
        if hid_size % num_heads != 0:
            new_hid_size = (hid_size // num_heads + 1) * num_heads
            logger.warning(
                f'Cannot divide hid_size ({hid_size}) by num_heads ({num_heads}). '
                f'Rounding up hid_size to {new_hid_size}.'
            )
            data['hid_size'] = new_hid_size
        return data

class FieldSum(nn.Module):
    def __init__(self, field_size):
        super().__init__()
        self.field_size = field_size

    def forward(self, embed_dict : Dict[str, torch.Tensor]):
        new_embed_dict = {}
        for ty, embed in embed_dict.items():
            assert embed.shape[1] % self.field_size == 0
            num_fields = embed.shape[1] // self.field_size
            embed = embed.reshape(-1, num_fields, self.field_size)
            new_embed_dict[ty] = embed.sum(1)
        return new_embed_dict

class HGTGNN(nn.Module):
    def __init__(
        self,
        graph_config : GraphConfig,
        solution_config : HGTSolutionConfig,
        node_in_size_dict : Dict[str, int],
        edge_in_size_dict : Dict[str, int],
        out_size : Optional[int],
        num_layers : int,
    ):
        super().__init__()
        if out_size is None:
            out_size = solution_config.hid_size
        self.out_size = out_size

        num_ntypes = len(graph_config.ntypes)
        num_etypes = len(graph_config.etypes)

        if solution_config.feat_encode_size is None:
            self.node_embed_mixer = dglnn.HeteroLinear(
                node_in_size_dict, solution_config.hid_size)
            node_in_size = solution_config.hid_size
            if len(edge_in_size_dict) != 0:
                self.edge_embed_mixer = dglnn.HeteroLinear(
                    edge_in_size_dict, solution_config.hid_size)
                edge_in_size = solution_config.hid_size
            else:
                edge_in_size = 0
        else:
            self.node_embed_mixer = FieldSum(solution_config.feat_encode_size)
            node_in_size = solution_config.feat_encode_size
            if len(edge_in_size_dict) != 0:
                self.edge_embed_mixer = FieldSum(solution_config.feat_encode_size)
                edge_in_size = solution_config.feat_encode_size
            else:
                edge_in_size = 0

        assert num_layers > 0
        self.layers = nn.ModuleList()
        self.fc_self = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                cur_layer_in_size = node_in_size
            else:
                cur_layer_in_size = solution_config.hid_size
            if i == num_layers - 1:
                cur_layer_out_size = out_size
            else:
                cur_layer_out_size = solution_config.hid_size

            self.layers.append(
                EdgeHGTConv(
                    solution_config.conv,
                    cur_layer_in_size,
                    edge_in_size,
                    cur_layer_out_size,
                    num_ntypes,
                    num_etypes,
                )
            )
            self.fc_self.append(nn.Linear(cur_layer_in_size, cur_layer_out_size))
        self.dropout = nn.Dropout(solution_config.dropout)

    def forward(
        self,
        mfgs,
        X_node_dict: Dict[str, torch.Tensor],
        X_edge_dicts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        assert len(mfgs) == len(self.layers)
        H_node_dict = self.node_embed_mixer(X_node_dict)
        for i, layer in enumerate(self.layers):
            if len(X_edge_dicts[i]) == 0:
                H_edge_dict = {}
            else:
                H_edge_dict = self.edge_embed_mixer(X_edge_dicts[i])
            # There are several steps to conduct HGT on a DGLBlock.
            # 1. Convert the DGLBlock to a homogeneous graph. The
            # homogeneous graph is based on the source nodes of the DGLBlock.
            # 2. Run HGTConv on the homogeneous graph to compute homogeneous
            # features.
            # 3. Extract the heterogeneous features of the destination nodes
            # from the homogeneous features.
            (
                homo_g,
                H_node,
                H_edge,
                info
            ) = block_to_homogeneous(mfgs[i], H_node_dict, H_edge_dict)
            # Run hgtconv on the homogeneous graph.
            H_node_new = layer(
                homo_g,
                homo_g.ndata[dgl.NTYPE],
                homo_g.edata[dgl.ETYPE],
                H_node,
                H_edge,
                presorted=True
            )
            # Residual connection.
            H_node = H_node_new + self.fc_self[i](H_node)
            # Activation.
            if i != len(self.layers) - 1:
                H_node = self.dropout(F.relu(H_node))
            # Convert the output features back to a dict.
            H_node_dict = convert_dstdata_to_dict(H_node, mfgs[i], info)
        return H_node_dict


class HGT(BaseGNN):
    def create_gnn(
        self,
        node_feat_size_dict: Dict[str, int],
        edge_feat_size_dict: Dict[str, int],
        seed_feat_size: int,
        out_size: Optional[int],
    ) -> nn.Module:
        gnn = HGTGNN(
            self.data_config.graph,
            self.solution_config,
            node_feat_size_dict,
            edge_feat_size_dict,
            out_size,
            num_layers=len(self.solution_config.fanouts),
        )
        return gnn

@gml_solution
class HGTSolution(BaseGMLSolution):
    config_class = HGTSolutionConfig
    name = "hgt"

    def create_model(self):
        return HGT(self.solution_config, self.data_config)
