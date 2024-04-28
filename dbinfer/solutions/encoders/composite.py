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


"""Composite class to encode a dictionary of features based on their config."""
from collections import defaultdict
import logging
from typing import Tuple, Dict, Optional, List, Any
import torch
import torch.nn as nn

from .base import get_encoder_class
from ..graph_dataset_config import (
    GraphDatasetConfig,
    FeatureConfig,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class FeatDictEncoder(nn.Module):

    def __init__(
        self,
        feat_configs : Dict[Any, FeatureConfig],
        feature_groups : Optional[List[List[Any]]],
        feat_encode_size : Optional[int]
    ):
        super().__init__()
        fg_cfgs = []
        self.ft2gid = {}
        if feature_groups is not None:
            for fg in feature_groups:
                gid = len(fg_cfgs)
                fg_cfgs.append(feat_configs[fg[0]])
                for ft in fg:
                    self.ft2gid[ft] = gid
        for ft in sorted(feat_configs.keys()):
            cfg = feat_configs[ft]
            if ft not in self.ft2gid:
                self.ft2gid[ft] = len(fg_cfgs)
                fg_cfgs.append(cfg)

        # Create encoders.
        self.encoders = nn.ModuleList()
        for i, cfg in enumerate(fg_cfgs):
            encoder_class = get_encoder_class(cfg.dtype)
            self.encoders.append(encoder_class(cfg.extra_fields, feat_encode_size))

        self.out_size_dict = {
            ft : self.encoders[gid].out_size
            for ft, gid in self.ft2gid.items()
        }

    def forward(self, input_feat_dict : Dict[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
        return {
            ft : self.encoders[self.ft2gid[ft]](val)
            for ft, val in input_feat_dict.items()
            # FIXME: what to do if a key in input_feat_dict does not exist in the encoders?
            # (e.g. itemId in Diginetica-clicks shouldn't be encoded?)
            if ft in self.ft2gid
        }

    def __repr__(self):
        super_repr = super().__repr__()
        ft_groups = [[] for i in range(len(self.encoders))]
        for ft, gid in self.ft2gid.items():
            ft_groups[gid].append(ft)
        ft_group_str = [str(fg) for fg in ft_groups]
        extra_repr = "  feat_groups=[\n"
        for fg in ft_groups:
            extra_repr += f"    {fg}\n"
        extra_repr += "  ]\n)"
        return super_repr[:-1] + extra_repr

FeatDict = Dict[str, Dict[str, torch.Tensor]]

class GraphFeatDictEncoder(nn.Module):

    @staticmethod
    def is_valid_feat(cfg : FeatureConfig) -> bool:
        return not cfg.is_time and get_encoder_class(cfg.dtype, allow_missing=True) is not None

    def __init__(
        self,
        data_config : GraphDatasetConfig,
        feat_encode_size : Optional[int]
    ):
        super().__init__()
        # Get feature groups.
        feature_groups = []
        if data_config.feature_groups is not None:
            feature_groups += [
                [(ft.type, ft.name) for ft in fg]
                for fg in data_config.feature_groups
            ]
        def _get_or_add_fg(feat_type, feat_name):
            for fg in feature_groups:
                for ty, name in fg:
                    if ty == feat_type and name == feat_name:
                        return fg
            feature_groups.append([(feat_type, feat_name)])
            return feature_groups[-1]
        # Get feature configs.
        feat_configs = {}
        for ntype, nt_cfgs in data_config.node_features.items():
            for feat_name, cfg in nt_cfgs.items():
                if self.is_valid_feat(cfg):
                    feat_configs[(ntype, feat_name)] = cfg
                else:
                    _raise_ignore_warning(ntype, feat_name, cfg)
        for etype, et_cfgs in data_config.edge_features.items():
            for feat_name, cfg in et_cfgs.items():
                if self.is_valid_feat(cfg):
                    feat_configs[(etype, feat_name)] = cfg
                else:
                    _raise_ignore_warning(etype, feat_name, cfg)
        target_type = data_config.task.target_type
        for feat_name in sorted(data_config.seed_features.keys()):
            cfg = data_config.seed_features[feat_name]
            if self.is_valid_feat(cfg):
                feat_configs[("__seed__", feat_name)] = cfg
                if (target_type, feat_name) in feat_configs:
                    _get_or_add_fg(target_type, feat_name).append(("__seed__", feat_name))
            else:
                _raise_ignore_warning("__seed__", feat_name, cfg)

        self.encoder = FeatDictEncoder(feat_configs, feature_groups, feat_encode_size)

        self.node_out_size_dict = defaultdict(lambda: 0)
        for ntype, nt_cfgs in data_config.node_features.items():
            for feat_name in nt_cfgs:
                self.node_out_size_dict[ntype] += \
                    self.encoder.out_size_dict.get((ntype, feat_name), 0)

        self.edge_out_size_dict = defaultdict(lambda: 0)
        for etype, et_cfgs in data_config.edge_features.items():
            for feat_name in et_cfgs:
                self.edge_out_size_dict[etype] += \
                    self.encoder.out_size_dict.get((etype, feat_name), 0)

        self.seed_ctx_out_size = 0
        for feat_name in data_config.seed_features:
            self.seed_ctx_out_size += \
                self.encoder.out_size_dict.get(('__seed__', feat_name), 0)

    def forward(
        self,
        feat_dict : FeatDict,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward function.

        Support both node feature dict and edge feature dict. For edge feature dict,
        the edge type is a string of form "src_type:edge_type:dst_type".
        """
        flat_feat_dict = {}
        for ty, ty_feat_dict in feat_dict.items():
            for feat_name, feat in ty_feat_dict.items():
                flat_feat_dict[(ty, feat_name)] = feat

        flat_feat_dict = self.encoder(flat_feat_dict)

        # Revert to nested dict.
        new_feat_dict = {}
        for ty, ty_feat_dict in feat_dict.items():
            if ty not in new_feat_dict:
                new_feat_dict[ty] = {}
            for feat_name, feat in ty_feat_dict.items():
                new_feat_dict[ty][feat_name] = flat_feat_dict[(ty, feat_name)]
        return new_feat_dict

    def __repr__(self):
        super_repr = super().__repr__()
        extra_repr = (f"  node_out_size_dict={dict(self.node_out_size_dict)}\n"
                      f"  edge_out_size_dict={dict(self.edge_out_size_dict)}\n"
                      f"  seed_ctx_out_size={self.seed_ctx_out_size}\n)")
        return super_repr[:-1] + extra_repr


def _raise_ignore_warning(feat_type, feat_name, cfg):
    logger.warning(f"Ignore feature ({feat_type}, {feat_name}) with dtype {cfg.dtype}.")
