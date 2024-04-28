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


import logging
from typing import Tuple, Dict, Optional, List, Any, Union
from enum import Enum

import pydantic
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


class AggregatorEnum(str, Enum):
    mean = "mean"
    max = "max"
    min = "min"
    std = "std"
    var = "var"
    sum = "sum"

class ScalarEnum(str, Enum):
    identity = "identity"
    amplification = "amplification"
    attenuation = "attenuation"


class PNAConvConfig(pydantic.BaseModel):
    has_bias: bool = True
    negative_slope: float = 0.01
    tower_dropout: float = 0.0
    num_towers: int = 1
    aggregators: List[AggregatorEnum] = [
        AggregatorEnum.mean,
        AggregatorEnum.max,
        AggregatorEnum.min,
        AggregatorEnum.std,
    ]
    scalers: List[ScalarEnum] = [ScalarEnum.identity]
    delta: int = 1

class PNAConvTower(nn.Module):
    """A single PNA tower in PNA layers"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        aggregators: List[AggregatorEnum],
        scalers: List[ScalarEnum],
        delta: int = 1,
        dropout: float = 0.0,
        edge_in_size: int = 0,
    ):
        super(PNAConvTower, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.edge_in_size = edge_in_size

        self.agg_funcs = [
            getattr(self, f"_agg_{agg}") for agg in aggregators
        ]
        self.scaler_funcs = [
            getattr(self, f"_scale_{scaler}") for scaler in scalers
        ]

        self.fc_self = nn.Linear(in_size, in_size)

        mixer_in_size = len(aggregators) * len(scalers) * in_size

        if edge_in_size > 0:
            self.fc_edge = nn.Linear(edge_in_size, in_size, bias=False)
            mixer_in_size = mixer_in_size * 2
        
        self.U = nn.Linear(mixer_in_size, out_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, X, X_edge=None, D=None):
        """compute the forward pass of a single tower in PNA convolution layer"""

        with graph.local_scope():
            X = self.fc_self(X)

            h = [agg_fn(graph, X, is_ndata=True) for agg_fn in self.agg_funcs]
            h = torch.cat(h, dim=1)
            h = [scaler_fn(h, D, self.delta) for scaler_fn in self.scaler_funcs]

            if self.edge_in_size > 0:
                assert X_edge is not None, "Edge features must be provided."
                X_edge = self.fc_edge(X_edge)
                h_e = [agg_fn(graph, X_edge, is_ndata=False) for agg_fn in self.agg_funcs]
                h_e = torch.cat(h_e, dim=1)
                h_e = [scaler_fn(h_e, D, self.delta) for scaler_fn in self.scaler_funcs]
                h = h + h_e
            
            h = torch.cat(h, dim=1)
            h = self.U(h)
            return self.dropout(h)

    def _agg_mean(self, graph, X_or_X_edge, is_ndata=True):
        with graph.local_scope():
            if is_ndata:
                graph.srcdata['h'] = X_or_X_edge
                graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'))
            else:
                graph.edata["a"] = X_or_X_edge
                graph.update_all(dgl.function.copy_e("a", "m"), dgl.function.mean('m', 'h'))
            return graph.dstdata['h']
        
    def _agg_sum(self, graph, X_or_X_edge, is_ndata=True):
        with graph.local_scope():
            if is_ndata:
                graph.srcdata['h'] = X_or_X_edge
                graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            else:
                graph.edata["a"] = X_or_X_edge
                graph.update_all(dgl.function.copy_e("a", "m"), dgl.function.sum('m', 'h'))
            return graph.dstdata['h']
    
    def _agg_min(self, graph, X_or_X_edge, is_ndata=True):
        with graph.local_scope():
            if is_ndata:
                graph.srcdata['h'] = X_or_X_edge
                graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.min('m', 'h'))
            else:
                graph.edata["a"] = X_or_X_edge
                graph.update_all(dgl.function.copy_e("a", "m"), dgl.function.min('m', 'h'))
            return graph.dstdata['h']
        
    def _agg_max(self, graph, X_or_X_edge, is_ndata=True):
        with graph.local_scope():
            if is_ndata:
                graph.srcdata['h'] = X_or_X_edge
                graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.max('m', 'h'))
            else:
                graph.edata["a"] = X_or_X_edge
                graph.update_all(dgl.function.copy_e("a", "m"), dgl.function.max('m', 'h'))
            return graph.dstdata['h']

    def _agg_var(self, graph, X_or_X_edge, is_ndata=True):
        diff = self._agg_mean(graph, X_or_X_edge ** 2, is_ndata) - self._agg_mean(graph, X_or_X_edge, is_ndata) ** 2
        return F.relu(diff)

    def _agg_std(self, graph, X, is_ndata=True):
        return torch.sqrt(self._agg_var(graph, X, is_ndata) + 1e-20)

    def _scale_identity(self, h, D=None, delta=1.0):
        return h

    def _scale_amplification(self, h, D, delta):
        return h * (np.log(D + 1) / delta)

    def _scale_attenuation(self, h, D, delta):
        return h * (delta / np.log(D + 1))


class PNAConv(nn.Module):

    def __init__(
        self,
        config : PNAConvConfig,
        in_size: Union[int, Tuple[int, int]],
        edge_in_size: int,
        out_size: int,
    ):
        super(PNAConv, self).__init__()
        self.config = config

        self._in_size_src, self._in_size_dst = dgl.utils.expand_as_pair(in_size)
        self._edge_in_size = edge_in_size
        self._out_size = out_size

        if self._in_size_src % config.num_towers != 0:
            logger.warning(
                f"Cannot divide in_size_src {self._in_size_src} by "
                f"num_towers {config.num_towers}. Force num_towers to be 1."
            )
            self.num_towers = 1
        else:
            self.num_towers = self.config.num_towers
    
        if out_size % config.num_towers != 0:
            logger.warning(
                f"Cannot divide out_size {out_size} by "
                f"num_towers {config.num_towers}. Force num_towers to be 1."
            )
            self.num_towers = 1
        else:
            self.num_towers = self.config.num_towers
        
        self.tower_in_size = self._in_size_src // self.num_towers
        self.tower_out_size = self._out_size // self.num_towers

        self.aggregators = self.config.aggregators
        self.scalers = self.config.scalers
        self.delta = self.config.delta
        self.tower_dropout = self.config.tower_dropout

        self.towers = nn.ModuleList(
            [
                PNAConvTower(
                    self.tower_in_size,
                    self.tower_out_size,
                    self.aggregators,
                    self.scalers,
                    self.delta,
                    dropout = self.tower_dropout,
                    edge_in_size=self._edge_in_size,
                )
                for _ in range(self.num_towers)
            ]
        )

        self.mixing_layer = nn.Sequential(
            nn.Linear(out_size, out_size, bias=False),
            nn.LeakyReLU(negative_slope=self.config.negative_slope)
        )

        self.fc_dst = nn.Linear(
            self._in_size_dst, out_size, bias=self.config.has_bias
        )

    def forward(self, graph, X, X_edge=None, D=None):
        with graph.local_scope():
            if isinstance(X, tuple):
                X_src, X_dst = X
            else:
                X_src = X_dst = X
                if graph.is_block:
                    X_dst = X_src[: graph.number_of_dst_nodes()]

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                return self.fc_dst(X_dst)
        
            # Message Passing
            h_cat = torch.cat(
                [
                    tower(
                        graph,
                        X_src[
                            :,
                            tower_idx * self.tower_in_size : (tower_idx + 1) * self.tower_in_size,
                        ],
                        X_edge,
                        D,
                    )
                    for tower_idx, tower in enumerate(self.towers)
                ],
                dim=1,
            )
            h_out = self.mixing_layer(h_cat)
            
            rst = self.fc_dst(X_dst) + h_out

            return rst
