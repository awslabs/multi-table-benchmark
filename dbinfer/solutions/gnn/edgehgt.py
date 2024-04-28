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


"""Heterogeneous Graph Transformer"""
from typing import Tuple, Dict, Optional, List, Any, Union
import logging
import pydantic
import math
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import TypedLinear
from dgl.nn.functional import edge_softmax

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class EdgeHGTConvConfig(pydantic.BaseModel):
    num_heads: int
    dropout : float = 0.0
    use_norm : bool = False

class EdgeHGTConv(nn.Module):

    def __init__(
        self,
        config: EdgeHGTConvConfig,
        in_size : Union[int, Tuple[int, int]],
        edge_in_size: int,
        out_size: int,
        num_ntypes,
        num_etypes,
    ):
        super().__init__()
        self.config = config
        self.in_size = in_size
        if out_size % config.num_heads != 0:
            logger.warning(
                f"Cannot divide out_size {out_size} by "
                f"num_heads {config.num_heads}. Force num_heads to be 1."
            )
            self.num_heads = num_heads = 1
        else:
            self.num_heads = num_heads = config.num_heads
        self.head_size = head_size = out_size // self.num_heads

        self.sqrt_d = math.sqrt(head_size)
        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(
            head_size * num_heads, head_size * num_heads, num_ntypes
        )

        self.relation_pri = nn.ParameterList(
            [nn.Parameter(torch.ones(num_etypes)) for i in range(num_heads)]
        )
        self.relation_att = nn.ModuleList(
            [
                TypedLinear(head_size, head_size, num_etypes)
                for i in range(num_heads)
            ]
        )
        self.relation_msg = nn.ModuleList(
            [
                TypedLinear(head_size, head_size, num_etypes)
                for i in range(num_heads)
            ]
        )
        if edge_in_size != 0:
            self.linear_edge_att = TypedLinear(
                edge_in_size, head_size * head_size * num_heads, num_etypes)
            self.linear_edge_msg = TypedLinear(
                edge_in_size, head_size * num_heads, num_etypes)
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(config.dropout)
        if config.use_norm:
            self.norm = nn.LayerNorm(out_size)
        if in_size != out_size:
            self.residual_w = nn.Parameter(
                torch.Tensor(in_size, out_size)
            )
            nn.init.xavier_uniform_(self.residual_w)

    def forward(
        self,
        g : dgl.DGLGraph,
        ntype : torch.Tensor,
        etype : torch.Tensor,
        x_node : torch.Tensor,
        x_edge : Optional[torch.Tensor],
        *,
        presorted=False
    ):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        x_node : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        x_edge : torch.Tensor, optional
            An optional 2D tensor of edge features. Shape: :math:`(|E|, D_{in})`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        self.presorted = presorted
        if g.is_block:
            x_src = x_node
            x_dst = x_node[: g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[: g.num_dst_nodes()]
        else:
            x_src = x_node
            x_dst = x_node
            srcntype = ntype
            dstntype = ntype
        with g.local_scope():
            k = self.linear_k(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            q = self.linear_q(x_dst, dstntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            v = self.linear_v(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            g.srcdata["k"] = k
            g.dstdata["q"] = q
            g.srcdata["v"] = v
            g.edata["etype"] = etype
            if x_edge is not None:
                edge_att = self.linear_edge_att(x_edge, etype, presorted).view(
                    -1, self.num_heads, self.head_size, self.head_size
                )
                edge_msg = self.linear_edge_msg(x_edge, etype, presorted).view(
                    -1, self.num_heads, self.head_size
                )
                g.edata["edge_att"] = edge_att
                g.edata["edge_msg"] = edge_msg

            # Message passing.
            g.apply_edges(self.message)

            # Softmax.
            g.edata["m"] = g.edata["m"] * edge_softmax(
                g, g.edata["a"]
            ).unsqueeze(-1)

            # Aggregate.
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            h = g.dstdata["h"].view(-1, self.num_heads * self.head_size)

            # target-specific aggregation
            h = self.drop(self.linear_a(h, dstntype, presorted))
            alpha = torch.sigmoid(self.skip[dstntype]).unsqueeze(-1)
            if x_dst.shape != h.shape:
                h = h * alpha + (x_dst @ self.residual_w) * (1 - alpha)
            else:
                h = h * alpha + x_dst * (1 - alpha)
            if self.config.use_norm:
                h = self.norm(h)
            return h

    def message(self, edges):
        """Message function."""
        has_edata = "edge_att" in edges.data
        head_att, head_msg = [], []
        etype = edges.data["etype"]
        k = torch.unbind(edges.src["k"], dim=1)
        q = torch.unbind(edges.dst["q"], dim=1)
        v = torch.unbind(edges.src["v"], dim=1)
        if has_edata:
            edge_att = torch.unbind(edges.data["edge_att"], dim=1)
            edge_msg = torch.unbind(edges.data["edge_msg"], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)  # (E, O)
            if has_edata:
                kw_edge = torch.bmm(k[i].unsqueeze(1), edge_att[i]).squeeze(1) # (E, O) @ (E, O, O) => (E, O)
                kw += kw_edge
            head_att.append(
                (kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d
            )  # (E,)
            msg = self.relation_msg[i](v[i], etype, self.presorted)  # (E, O)
            if has_edata:
                msg += edge_msg[i]
            head_msg.append(msg)
        return {"a": torch.stack(head_att, dim=1), "m": torch.stack(head_msg, dim=1)}
