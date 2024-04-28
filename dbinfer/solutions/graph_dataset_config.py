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


"""Metadata used to create GML solutions.

Despite the similarity, the classes here are of different purposes than the metadata
classes in `dbinfer_bench`. The classes here are to unify the interface of different
ML solutions, making them more extensible against future changes.
"""
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
import pydantic
import copy
from dbinfer_bench import (
    DBBColumnDType,
    DBBTaskType,
    DBBTaskEvalMetric,
    DBBGraphDataset,
    DBBGraphFeatureID,
    DBBGraphTaskMeta,
)

from .tabular_dataset_config import FeatureConfig

NType = str
EType = str  # Colon-separated string. E.g., author:write:paper

__all__ = [
    'GraphConfig',
    'FeatureConfig',
    'TaskConfig',
    'GraphDatasetConfig',
    'parse_config_from_graph_dataset',
]

class GraphConfig(pydantic.BaseModel):

    # Number of nodes of each type.
    num_nodes : Dict[NType, int]

    # Number of edges of each type.
    num_edges : Dict[EType, int]

    # Name of each node types.
    ntypes : List[NType]

    # Name of each edge types in colon-separated string. E.g., author:write:paper
    etypes : List[EType]

    # From an etype name to its reverse etype.
    reverse_etypes_mapping : Dict[EType, EType]

TaskConfig = DBBGraphTaskMeta

class GraphDatasetConfig(pydantic.BaseModel):
    graph : GraphConfig
    node_features : Dict[NType, Dict[str, FeatureConfig]]
    edge_features : Dict[EType, Dict[str, FeatureConfig]]
    seed_features : Dict[str, FeatureConfig]
    task : TaskConfig
    feature_groups : Optional[List[List[DBBGraphFeatureID]]]

def parse_config_from_graph_dataset(
    dataset : DBBGraphDataset,
    task_name : str
) -> GraphDatasetConfig:
    # Parse graph config.
    ntypes = list(dataset.graph.num_nodes.keys())
    etypes = list(dataset.graph.num_edges.keys())
    reverse_etypes_mapping = {}
    for etype in etypes:
        etuple = etype.split(':')
        if etuple[1].startswith('reverse_'):
            rev_type = etuple[1][8:]
        else:
            rev_type = 'reverse_' + etuple[1]
        reverse_etypes_mapping[etype] = f"{etuple[2]}:{rev_type}:{etuple[0]}"
    g_cfg = GraphConfig(
        num_nodes=dataset.graph.num_nodes,
        num_edges=dataset.graph.num_edges,
        ntypes=ntypes,
        etypes=etypes,
        reverse_etypes_mapping=reverse_etypes_mapping
    )

    # Parse feature config.
    feature_groups = []
    if dataset.tgif_metadata.feature_groups is not None:
        feature_groups = dataset.tgif_metadata.feature_groups

    nfeat_cfg = defaultdict(dict)
    efeat_cfg = defaultdict(dict)
    for feat_yaml in dataset.yaml_data['feature_data']:
        extra_fields = copy.deepcopy(feat_yaml['extra_fields'])
        dtype = extra_fields.pop('dtype')
        feat_name = feat_yaml['name']
        if dtype == DBBColumnDType.float_t:
            extra_fields['in_size'] = dataset.feature.size(
                feat_yaml['domain'], feat_yaml['type'], feat_yaml['name'])[0]
        elif dtype == DBBColumnDType.category_t:
            assert 'num_categories' in extra_fields, \
                "Categorical features require num_categories to be specified."
        feat_cfg = FeatureConfig(
            dtype=dtype,
            is_time=False,
            extra_fields=extra_fields
        )
        feat_type = feat_yaml['type']
        if feat_yaml['domain'] == 'node':
            nfeat_cfg[feat_type][feat_name] = feat_cfg
        elif feat_yaml['domain'] == 'edge':
            efeat_cfg[feat_type][feat_name] = feat_cfg

    # Parse task config.
    task_cfg = dataset.graph_tasks[task_name].metadata

    # Parse task-specific seed features.
    sfeat_cfg = {}
    for feat_schema in task_cfg.seed_feature_schema:
        feat_name = feat_schema.name
        assert feat_name not in [
            task_cfg.seed_timestamp,
            task_cfg.key_prediction_label_column,
            task_cfg.key_prediction_query_idx_column,
        ]
        sfeat_cfg[feat_name] = FeatureConfig(
            dtype=feat_schema.dtype,
            is_time=False,
            extra_fields=dict(feat_schema))

    return GraphDatasetConfig(
        graph=g_cfg,
        node_features=nfeat_cfg,
        edge_features=efeat_cfg,
        seed_features=sfeat_cfg,
        task=task_cfg,
        feature_groups=feature_groups
    )
