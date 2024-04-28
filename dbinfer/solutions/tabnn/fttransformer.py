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
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registry


# Modified from https://github.com/lucidrains/tab-transformer-pytorch
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        include_norm = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim) if include_norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)  # q, k, v: (B, L, H*d)
        q = q.view(B, L, H, -1).transpose(-2, -3)  # (B, H, L, d)
        k = k.view(B, L, H, -1).transpose(-2, -3)  # (B, H, L, d)
        v = v.view(B, L, H, -1).transpose(-2, -3)  # (B, H, L, d)
        q = q * self.scale

        sim = torch.matmul(q, k.transpose(-1, -2))  # (B, H, L, d) @ (B, H, d, L) -> (B, H, L, L)
        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = torch.matmul(dropped_attn, v)  # (B, H, L, L) @ (B, H, L, d) -> (B, H, L, d)

        out = out.transpose(-2, -3).reshape(B, L, -1)  # (B, L, H*d)
        out = self.to_out(out)

        return out, attn


# transformer

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        include_first_norm=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout,
                          include_norm = (i > 0 or include_first_norm)),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

class FTTransformerTabNNConfig(pydantic.BaseModel):
    hid_size : Optional[int] = 128
    dropout : Optional[float] = 0.7
    num_layers : Optional[int] = 3
    attn_dropout : Optional[float] = 0.0
    num_heads : Optional[int] = 8
    use_token : Optional[bool] = True
    include_first_norm : Optional[bool] = False

@registry.tabnn
class FTTransformerTabNN(nn.Module):
    name = "fttransformer"
    config_class = FTTransformerTabNNConfig
    def __init__(self,
                 config : FTTransformerTabNNConfig,
                 num_fields : int,
                 field_size : int,
                 out_size : int):
        super().__init__()
        self.config = config
        self.proc_in = nn.Linear(field_size, config.hid_size)
        self.transformer_enc = TransformerEncoder(
            dim=config.hid_size,
            depth=config.num_layers,
            heads=config.num_heads,
            dim_head=config.hid_size // config.num_heads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.dropout,
            include_first_norm=config.include_first_norm,
        )
        self.fc_out = nn.Sequential(
            nn.LayerNorm(config.hid_size),
            nn.ReLU(),
            nn.Linear(config.hid_size, out_size),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hid_size))
        self.use_token = config.use_token

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        X = self.proc_in(X)
        X = X.view(X.shape[0], -1, self.config.hid_size)
        if self.use_token:
            X = torch.cat([self.cls_token.expand(X.shape[0], 1, -1), X], 1)
            X = self.transformer_enc(X)
            X = X[:, 0]
        else:
            X = self.transformer_enc(X).mean(1)
        X = self.fc_out(X)
        return X
