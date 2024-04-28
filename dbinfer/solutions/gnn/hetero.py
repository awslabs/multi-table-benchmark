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

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_dataset_config import GraphConfig

class HeteroGNN(nn.Module):
    """Heterogeneous GNN wrapper."""
    def __init__(
        self,
        graph_config : GraphConfig,
        node_in_size_dict : Dict[str, int],
        edge_in_size_dict : Dict[str, int],
        num_layers : int,
        hid_size : int,
        layer_class,
        layer_config,
    ):
        super().__init__()
        self._out_size = hid_size
        etypes = [tuple(et.split(':')) for et in graph_config.etypes]
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            dgl.nn.HeteroGraphConv({
                (st, et, dt) : layer_class(
                    layer_config,
                    (node_in_size_dict[st], node_in_size_dict[dt]),
                    edge_in_size_dict[f"{st}:{et}:{dt}"],
                    hid_size
                )
                for (st, et, dt) in etypes
            }, aggregate='mean')
        )
        for i in range(num_layers - 1):
            self.conv_layers.append(
                dgl.nn.HeteroGraphConv({
                    (st, et, dt) : layer_class(
                        layer_config,
                        (hid_size, hid_size),
                        edge_in_size_dict[f"{st}:{et}:{dt}"],
                        hid_size
                    )
                    for (st, et, dt) in etypes
                }, aggregate='mean')
            )

    @property
    def out_size(self) -> int:
        return self._out_size

    def forward(
        self,
        mfgs,
        X_node_dict : Dict[str, torch.Tensor],
        X_edge_dicts : List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        assert len(mfgs) == len(self.conv_layers)
        H_node_dict = X_node_dict
        for i, conv in enumerate(self.conv_layers):
            # Convert edge dict key from src_type:etype:dst_type to etype.
            X_edge_dict = {
                key.split(':')[1] : {'X_edge' : X}
                for key, X in X_edge_dicts[i].items()
            }
            H_node_dict = conv(mfgs[i], H_node_dict, mod_kwargs=X_edge_dict)
        return H_node_dict


class HeteroGNNLayer(nn.Module):
    """Heterogeneous GNN layer wrapper."""
    def __init__(
        self,
        graph_config : GraphConfig,
        in_size_dict : Dict[str, int],
        out_size : int,
        edge_in_size_dict : Dict[str, int],
        layer_class,
        layer_config,
    ):
        super().__init__()
        etypes = [tuple(et.split(':')) for et in graph_config.etypes]
        self.conv = dgl.nn.HeteroGraphConv({
                (st, et, dt) : layer_class(
                    layer_config,
                    (in_size_dict[st], in_size_dict[dt]),
                    edge_in_size_dict[f"{st}:{et}:{dt}"],
                    out_size,
                )
                for (st, et, dt) in etypes
            }, aggregate='sum')
        self.loop_fc = nn.ModuleDict(
            {
                ntype : nn.Linear(in_size_dict[ntype], out_size)
                for ntype in graph_config.ntypes
            }
        )

    def forward(
        self,
        mfg,
        X_node_dict : Dict[str, torch.Tensor],
        X_edge_dict : Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Convert edge dict key from src_type:etype:dst_type to etype.
        X_edge_dict = {
            key.split(':')[1] : {'X_edge' : X}
            for key, X in X_edge_dict.items()
        }
        H_node_dict = self.conv(mfg, X_node_dict, mod_kwargs=X_edge_dict)
        X_dstnode_dict = {
            ntype : X[: mfg.number_of_dst_nodes(ntype)]
            for ntype, X in X_node_dict.items()
        }
        H_node_dict = {
            ntype : H_node_dict[ntype] + self.loop_fc[ntype](X_dstnode_dict[ntype])
            for ntype in H_node_dict
        }
        return H_node_dict
