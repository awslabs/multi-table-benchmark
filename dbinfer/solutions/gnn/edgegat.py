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
import logging

import pydantic
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.functional import edge_softmax

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class EdgeGATConvConfig(pydantic.BaseModel):
    num_heads: int
    feat_drop: float = 0.0
    attn_drop: float = 0.0
    negative_slope: float = 0.2
    allow_zero_in_degree: bool = True
    has_bias: bool = True

class EdgeGATConv(nn.Module):

    def __init__(
        self,
        config : EdgeGATConvConfig,
        in_size: Union[int, Tuple[int, int]],
        edge_in_size: int,
        out_size: int,
    ):
        super(EdgeGATConv, self).__init__()
        self.config = config

        self._in_size_src, self._in_size_dst = dgl.utils.expand_as_pair(in_size)
        self._edge_in_size = edge_in_size
        self._out_size = out_size

        if out_size % config.num_heads != 0:
            logger.warning(
                f"Cannot divide out_size {out_size} by "
                f"num_heads {config.num_heads}. Force num_heads to be 1."
            )
            self._num_heads = 1
        else:
            self._num_heads = self.config.num_heads
        self._head_size = out_size // self._num_heads

        self.fc_src = nn.Linear(
            self._in_size_src, out_size, bias=False
        )
        self.fc_dst = nn.Linear(
            self._in_size_dst, out_size, bias=False
            )
        self.attn_src = nn.Parameter(
            torch.FloatTensor(size=(1, self._num_heads, self._head_size))
        )
        self.attn_dst = nn.Parameter(
            torch.FloatTensor(size=(1, self._num_heads, self._head_size))
        )

        self.feat_drop = nn.Dropout(self.config.feat_drop)
        self.attn_drop = nn.Dropout(self.config.attn_drop)
        self.leaky_relu = nn.LeakyReLU(self.config.negative_slope)

        if self._edge_in_size > 0:
            self.fc_edge = nn.Linear(
                self._edge_in_size, out_size, bias=False)
            self.attn_edge = nn.Parameter(
                torch.FloatTensor(size=(1, self._num_heads, self._head_size))
            )

        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_src, gain=gain)
        nn.init.xavier_normal_(self.attn_dst, gain=gain)

        if self._edge_in_size > 0:
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
            nn.init.xavier_normal_(self.attn_edge, gain=gain)


    def forward(self, graph, X, X_edge=None):

        with graph.local_scope():
            if not self.config.allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise dgl.DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(X, tuple):
                X_src = self.feat_drop(X[0])
                X_dst = self.feat_drop(X[1])
                src_prefix_shape = X[0].shape[:-1]
                dst_prefix_shape = X[1].shape[:-1]
            else:
                X_src = X_dst = self.feat_drop(X)
                src_prefix_shape = dst_prefix_shape = X.shape[:-1]
                
                if graph.is_block:
                    X_dst = X_src[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]

            h_src = self.fc_src(X_src).view(
                X_src.shape[0], self._num_heads, self._head_size
            )
            h_dst = self.fc_dst(X_dst).view(
                X_dst.shape[0], self._num_heads, self._head_size
            )
            h_neigh = self.aggregate_src(graph, h_src, h_dst)

            if X_edge is not None:
                h_edge = self.fc_edge(X_edge).view(
                    X_edge.shape[0], self._num_heads, self._head_size
                )
                h_edge = self.aggregate_edge(graph, h_edge, h_dst)
                h_neigh = h_neigh + h_edge

            # Residual connection
            h = h_dst + h_neigh

            h = h.view(h_neigh.shape[0], self._num_heads * self._head_size)

            return h

    def aggregate_src(self, graph, h_src, h_dst):
        with graph.local_scope():
            a = self.compute_node_attention(graph, h_src, h_dst)
            graph.srcdata["h_src"] = h_src
            graph.edata["a"] = a
            graph.update_all(dgl.function.u_mul_e("h_src", "a", "m"), dgl.function.sum("m", "h_neigh"))
            return graph.dstdata["h_neigh"]

    def compute_node_attention(self, graph, h_src, h_dst):
        with graph.local_scope():

            e_src = (h_src * self.attn_src).sum(dim=-1).unsqueeze(-1)
            e_dst = (h_dst * self.attn_dst).sum(dim=-1).unsqueeze(-1)

            graph.srcdata["e_src"] = e_src
            graph.dstdata["e_dst"] = e_dst
            
            graph.apply_edges(dgl.function.u_add_v("e_src", "e_dst", "e_node"))
            e = graph.edata.pop('e_node')
            e = self.leaky_relu(e)
            a = self.attn_drop(edge_softmax(graph, e))
            return a

    def aggregate_edge(self, graph, h_edge, h_dst):
        with graph.local_scope():
            a = self.compute_edge_attention(graph, h_edge, h_dst)
            graph.edata["m_edge"] = a * h_edge
            graph.update_all(dgl.function.copy_e("m_edge", "m"), dgl.function.sum("m", "h_edge"))
            return graph.dstdata["h_edge"]

    def compute_edge_attention(self, graph, h_edge, h_dst):
        with graph.local_scope():

            e_edge = (h_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            e_dst = (h_dst * self.attn_dst).sum(dim=-1).unsqueeze(-1)

            graph.edata["e_edge"] = e_edge
            graph.dstdata["e_dst"] = e_dst
            
            graph.apply_edges(dgl.function.e_add_v("e_edge", "e_dst", "e_edge"))
            e = graph.edata.pop('e_edge')
            e = self.leaky_relu(e)
            a = self.attn_drop(edge_softmax(graph, e))
            return a
