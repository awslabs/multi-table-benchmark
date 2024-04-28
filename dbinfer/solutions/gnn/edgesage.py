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
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeSAGEConvConfig(pydantic.BaseModel):
    has_bias: bool = True

class EdgeSAGEConv(nn.Module):

    def __init__(
        self,
        config : EdgeSAGEConvConfig,
        in_size: Union[int, Tuple[int, int]],
        edge_in_size: int,
        out_size: int,
    ):
        super(EdgeSAGEConv, self).__init__()
        self.config = config

        self._in_size_src, self._in_size_dst = dgl.utils.expand_as_pair(in_size)
        self._edge_in_size = edge_in_size
        self._out_size = out_size

        self.fc_neigh = nn.Linear(self._in_size_src, out_size, bias=False)
        self.fc_self = nn.Linear(self._in_size_dst, out_size, bias=config.has_bias)
        
        if self._edge_in_size > 0:
            self.fc_edge = nn.Linear(self._edge_in_size, out_size, bias=False)

    def _compute_node_src_aggre(self, graph, X_src):
        
        with graph.local_scope():

            graph.srcdata["h"] = (
                self.fc_neigh(X_src)
            )
            graph.update_all(dgl.function.copy_u("h", "m"), dgl.function.mean("m", "neigh"))

            return graph.dstdata["neigh"]


    def _compute_edge_aggre(self, graph, X_edge):
        
        with graph.local_scope():

            assert X_edge.shape[0] == graph.num_edges()

            graph.edata["e"] = self.fc_edge(X_edge)
            graph.update_all(dgl.function.copy_e("e", "m_e"), dgl.function.mean("m_e", "neigh_e"))
            
            return graph.dstdata["neigh_e"]

    def forward(self, graph, X, X_edge=None):
        with graph.local_scope():
            if isinstance(X, tuple):
                X_src, X_dst = X
            else:
                X_src = X_dst = X
                if graph.is_block:
                    X_dst = X_src[: graph.number_of_dst_nodes()]

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                return self.fc_self(X_dst)
        
            # Message Passing
            h_neigh = self._compute_node_src_aggre(graph, X_src)

            if X_edge is not None:
                h_neigh = h_neigh + self._compute_edge_aggre(graph, X_edge)

            rst = self.fc_self(X_dst) + h_neigh

            return rst
