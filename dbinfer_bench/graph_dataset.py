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


from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pydantic
import os
import numpy as np

from dgl.graphbolt import (
    OnDiskDataset,
    ItemSetDict,
)
from dgl.graphbolt.impl import (
    OnDiskFeatureDataFormat,
    OnDiskFeatureDataDomain,
    OnDiskTaskData,
)
from .ondisk_dataset_creator import (
    OnDiskDatasetCreator,
    OnDiskTaskCreator,
)
from . import yaml_utils

from .dataset_meta import (
    DBBTaskMeta,
    DBBTaskType,
    DBBTaskEvalMetric,
    TASK_EXTRA_FIELDS,
    DBBColumnDType,
    DBBColumnID,
    DBBColumnSchema,
)
from .download import download_or_get_path

__all__ = [
    'DBBGraphDataset',
    'DBBGraphDatasetMeta',
    'DBBGraphDatasetCreator',
    'DBBGraphTask',
    'DBBGraphTaskMeta',
    'DBBGraphTaskCreator',
    'DBBGraphFeatureID',
    'load_graph_data',
]

class DBBGraphFeatureID(pydantic.BaseModel):
    class Config:
        use_enum_values = True
    domain : OnDiskFeatureDataDomain
    type : str
    name : str

class DBBGraphDatasetMeta(pydantic.BaseModel):
    dataset_name : str
    feature_groups : Optional[List[List[DBBGraphFeatureID]]]

class DBBGraphTaskMeta(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.allow
        use_enum_values = True
    # Fields shared with DBBTaskMeta.
    name : str
    evaluation_metric : DBBTaskEvalMetric
    task_type : DBBTaskType
    key_prediction_label_column: Optional[str] = "label"
    key_prediction_query_idx_column: Optional[str] = "query_idx"

    # Fields unique to DBBGraphTaskMeta.

    # Seed ID type. For entity prediction, this is the
    # node type. For relation prediction, this is the relation type
    # in the form of "src_type:edge_type:dst_type".
    seed_type : str
    # Target type name. This is the type name that the target table of this task
    # is mapped to.
    target_type : str
    # Number of seed IDs in each item. For entity prediction, it is equal to one.
    # For retrieval, it is >= 2.
    num_seeds : int
    # Seed feature schemas.
    seed_feature_schema : List[DBBColumnSchema]
    # Seed timestamp.
    seed_timestamp : Optional[str] = None

@dataclass
class DBBGraphTask:
    metadata : DBBGraphTaskMeta
    train_set : ItemSetDict
    validation_set : ItemSetDict
    test_set : ItemSetDict

class DBBGraphDataset(OnDiskDataset):

    def __init__(self, path : Path):
        super().__init__(path, include_original_edge_id=True)

        self._path = Path(path)
        self._tgif_metadata = self._load_tgif_metadata()

    def _load_tgif_metadata(self):
        return yaml_utils.load_pyd(DBBGraphDatasetMeta, self._path / 'tgif_metadata.yaml')

    def load(self):
        super().load()
        self._graph_tasks = {}
        for task in self.tasks:
            task_meta = DBBGraphTaskMeta.parse_obj(task.metadata)
            self._graph_tasks[task_meta.name] = DBBGraphTask(
                task_meta,
                task.train_set,
                task.validation_set,
                task.test_set
            )
        return self

    @property
    def tgif_metadata(self) -> DBBGraphDatasetMeta:
        return self._tgif_metadata

    @property
    def graph_tasks(self) -> Dict[str, DBBGraphTask]:
        return self._graph_tasks

class DBBGraphTaskCreator(OnDiskTaskCreator):
    def __init__(self, name : str):
        super().__init__(name)
        self.seed_feature_schema = []

    def set_seed_type(self, seed_type : str):
        return self.add_extra_field('seed_type', seed_type)

    def set_target_type(self, target_type : str):
        return self.add_extra_field('target_type', target_type)

    def add_meta(self, task_meta : DBBTaskMeta):
        self.add_extra_field('evaluation_metric', task_meta.evaluation_metric)
        self.add_extra_field('task_type', task_meta.task_type)
        for task_extra_field in TASK_EXTRA_FIELDS[task_meta.task_type]:
            self.add_extra_field(task_extra_field, getattr(task_meta, task_extra_field))
        return self

    def set_seeds(
        self,
        train_seeds : np.ndarray,
        validation_seeds : np.ndarray,
        test_seeds : np.ndarray,
        type : Optional[str] = None,
    ):
        """Set seed (nodes, edges, etc.) IDs.

        The allowed values are:
          - 1D array: corresponds to seed node IDs.
          - 2D array of shape (N,k): corresponds to seed tuples. When k==2,
            it corresponds to edges.
        """
        if train_seeds.ndim == 1:
            seed_key = "seed_nodes"
            self.add_extra_field('num_seeds', 1)
        elif train_seeds.ndim == 2:
            seed_key = "node_pairs"
            self.add_extra_field('num_seeds', train_seeds.shape[1])
        else:
            seed_key = "seeds"
            self.add_extra_field('num_seeds', train_seeds.shape[1])
        self.add_item(seed_key, train_seeds, validation_seeds, test_seeds, type=type)
        return self

    def set_labels(
        self,
        train_labels : Optional[np.ndarray],
        validation_labels : np.ndarray,
        test_labels : np.ndarray,
        type : Optional[str] = None
    ):
        """Set labels.

        Required by classification or regression tasks. Retrieval task typically
        do not provide train_labels (pass-in None).
        """
        key = "labels"
        self.add_item(key, train_labels, validation_labels, test_labels, type=type)
        return self

    def set_seed_timestamp(
        self,
        train_ts : Optional[np.ndarray],
        validation_ts : Optional[np.ndarray],
        test_ts : Optional[np.ndarray],
        type : Optional[str] = None
    ):
        """Set seed timestamps."""
        key = "timestamp"
        self.add_extra_field('seed_timestamp', key)
        return self.add_item(key, train_ts, validation_ts, test_ts, type=type)

    def add_seed_feature(
        self,
        name : str,
        train_data : Optional[np.ndarray],
        validation_data : Optional[np.ndarray],
        test_data : Optional[np.ndarray],
        type : Optional[str] = None,
        **extra_meta
    ):
        seed_feat_meta = DBBColumnSchema(name=name, **extra_meta)
        self.seed_feature_schema.append(seed_feat_meta)
        return self.add_item(name, train_data, validation_data, test_data, type)

    def add_item(
        self,
        name : str,
        train_data : Optional[np.ndarray],
        validation_data : Optional[np.ndarray],
        test_data : Optional[np.ndarray],
        type : Optional[str] = None,
    ):
        save_format = OnDiskFeatureDataFormat.NUMPY
        in_memory = True
        return super().add_item(
            name, save_format, train_data, validation_data, test_data,
            in_memory=in_memory, type=type)

    def done(self, path : Path):
        self.add_extra_field("seed_feature_schema", self.seed_feature_schema)
        return super().done(path)

def load_graph_data(name_or_path : str) -> DBBGraphDataset:
    path = download_or_get_path(name_or_path)
    return DBBGraphDataset(path).load()

class DBBGraphDatasetCreator(OnDiskDatasetCreator):

    def __init__(self, name : str):
        super().__init__(name)
        self.name = name
        self.feature_groups = None

    def add_feature_group(
        self,
        feat_group : List[Tuple[OnDiskFeatureDataDomain, str, str]]
    ):
        if self.feature_groups is None:
            self.feature_groups = []
        feat_group = [
            DBBGraphFeatureID(
                domain=domain,
                type=type,
                name=name
            )
            for domain, type, name in feat_group
        ]
        self.feature_groups.append(feat_group)
        return self

    def done(self, path : Path):
        path = Path(path)
        super().done(path)

        tgif_metadata = DBBGraphDatasetMeta(
            dataset_name=self.name,
            feature_groups=self.feature_groups
        )
        yaml_utils.save_pyd(tgif_metadata, path / "tgif_metadata.yaml")
