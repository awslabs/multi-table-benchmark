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


from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
import pydantic
from enum import Enum
import numpy as np
import pandas as pd
import os

from dgl.graphbolt import OnDiskDataset
from dgl.graphbolt.impl import (
    OnDiskFeatureDataFormat,
    OnDiskTVTSetData,
    OnDiskTVTSet,
    OnDiskFeatureDataDomain,
    OnDiskFeatureData,
    OnDiskTaskData,
)

from . import yaml_utils
from .dataset_meta import TIMESTAMP_FEATURE_NAME

__all__ = ['OnDiskTaskCreator', 'OnDiskDatasetCreator']

class OnDiskDatasetEdgesFormat(str, Enum):
    CSV = "csv"
    NUMPY = "numpy"

class OnDiskDatasetNodesMeta(pydantic.BaseModel):
    type : Optional[str] = None
    num : int

class OnDiskDatasetEdgesMeta(pydantic.BaseModel):
    class Config:
        use_enum_values = True
    type : Optional[str] = None
    format : OnDiskDatasetEdgesFormat
    path : str

class OnDiskDatasetGraphMeta(pydantic.BaseModel):
    nodes : List[OnDiskDatasetNodesMeta]
    edges : List[OnDiskDatasetEdgesMeta]
    feature_data : Optional[List[OnDiskFeatureData]] = []

class OnDiskDatasetMeta(pydantic.BaseModel):
    # This data model differs from the internal OnDiskDatasetMeta
    # because it is designed for dataset creation.
    dataset_name: str
    graph: OnDiskDatasetGraphMeta
    feature_data: Optional[List[OnDiskFeatureData]] = []
    tasks: Optional[List[OnDiskTaskData]] = []

class OnDiskTaskCreator:

    def __init__(self, name : str):
        self.name = name
        self.extra_fields = {}
        self.train_set = {}
        self.validation_set = {}
        self.test_set = {}

    def add_extra_field(self, key : str, val : Any):
        self.extra_fields[key] = val
        return self

    def add_item(
        self,
        name : str,
        save_format : OnDiskFeatureDataFormat,
        train_data : Optional[np.ndarray],
        validation_data : Optional[np.ndarray],
        test_data : Optional[np.ndarray],
        in_memory : bool = True,
        type : Optional[str] = None
    ):
        assert save_format == OnDiskFeatureDataFormat.NUMPY
        if type not in self.train_set:
            self.train_set[type] = []
            self.validation_set[type] = []
            self.test_set[type] = []
        if train_data is not None:
            self.train_set[type].append({
                "name" : name,
                "format" : save_format,
                "in_memory" : in_memory,
                "data" : train_data
            })
        if validation_data is not None:
            self.validation_set[type].append({
                "name" : name,
                "format" : save_format,
                "in_memory" : in_memory,
                "data" : validation_data
            })
        if test_data is not None:
            self.test_set[type].append({
                "name" : name,
                "format" : save_format,
                "in_memory" : in_memory,
                "data" : test_data
            })
        return self

    def _validate(self):
        def _validate_one_set(item_set):
            for type, items in item_set.items():
                item_len = len(items[0]["data"])
                for item in items:
                    if len(item["data"]) != item_len:
                        raise ValueError(
                            f"Expect items in train/val/test sets to have"
                            f" the same length. But len({items[0]['name']})={item_len}"
                            f" while len({item['name']})={len(item['data'])}."
                        )
        _validate_one_set(self.train_set)
        _validate_one_set(self.validation_set)
        _validate_one_set(self.test_set)

    def done(self, path : Path) -> OnDiskTaskData:
        self._validate()

        task_path = Path(path) / self.name
        if not task_path.exists():
            task_path.mkdir(parents=True, exist_ok=True)

        # Write data.
        train_set_dir = task_path / "train_set"
        train_set_dir.mkdir(parents=True, exist_ok=True)
        for type, items in self.train_set.items():
            type_str = "" if type is None else f"{type}_"
            for item in items:
                item_name, item_data = item["name"], item["data"]
                data_path = train_set_dir / f"{type_str}{item_name}.npy"
                np.save(data_path, item_data, allow_pickle=True)
                item.pop("data")
                item["path"] = str(data_path.relative_to(path))

        validation_set_dir = task_path / "validation_set"
        validation_set_dir.mkdir(parents=True, exist_ok=True)
        for type, items in self.validation_set.items():
            type_str = "" if type is None else f"{type}_"
            for item in items:
                item_name, item_data = item["name"], item["data"]
                data_path = validation_set_dir / f"{type_str}{item_name}.npy"
                np.save(data_path, item_data, allow_pickle=True)
                item.pop("data")
                item["path"] = str(data_path.relative_to(path))

        test_set_dir = task_path / "test_set"
        test_set_dir.mkdir(parents=True, exist_ok=True)
        for type, items in self.test_set.items():
            type_str = "" if type is None else f"{type}_"
            for item in items:
                item_name, item_data = item["name"], item["data"]
                data_path = test_set_dir / f"{type_str}{item_name}.npy"
                np.save(data_path, item_data, allow_pickle=True)
                item.pop("data")
                item["path"] = str(data_path.relative_to(path))

        # Create and return metadata.
        train_set_meta = []
        for type, items in self.train_set.items():
            train_set_meta.append(OnDiskTVTSet.parse_obj({
                "type" : type,
                "data" : items,
            }))

        validation_set_meta = []
        for type, items in self.validation_set.items():
            validation_set_meta.append(OnDiskTVTSet.parse_obj({
                "type" : type,
                "data" : items,
            }))

        test_set_meta = []
        for type, items in self.test_set.items():
            test_set_meta.append(OnDiskTVTSet.parse_obj({
                "type" : type,
                "data" : items,
            }))

        return OnDiskTaskData(
            name=self.name,
            train_set=train_set_meta,
            validation_set=validation_set_meta,
            test_set=test_set_meta,
            **self.extra_fields
        )

class OnDiskDatasetCreator:

    def __init__(self, name : str):

        self.name = name
        self.nodes = []
        self.edges = []
        # Graph attributes used for sampling, e.g., node/edge timestamps.
        self.graph_attributes = []
        self.features = []
        self.tasks = []

    def add_nodes(self, num : int, type : Optional[str] = None):
        self.nodes.append({"type" : type, "num" : num})
        return self

    def add_edges(
        self,
        data : np.ndarray,
        type : Optional[str] = None
    ):
        # Expect data to be of shape (2, E)
        save_format = OnDiskDatasetEdgesFormat.NUMPY
        if type is not None:
            assert len(type.split(':')) == 3, "Edge type name must be a colon-spearated triplet."
        self.edges.append({"data" : data, "format" : save_format, "type" : type})
        return self

    def add_node_feature(
        self,
        type : str,
        name : str,
        save_format : OnDiskFeatureDataFormat,
        data : np.ndarray,
        in_memory : bool = True,
        **extra_kwargs
    ):
        assert save_format == OnDiskFeatureDataFormat.NUMPY
        feat_item = {
            "domain" : OnDiskFeatureDataDomain.NODE,
            "type" : type,
            "name" : name,
            "format" : save_format,
            "in_memory" : in_memory,
            "data" : data,
        }
        feat_item.update(extra_kwargs)
        self.features.append(feat_item)
        return self

    def add_node_timestamp(
        self,
        type : str,
        data : np.ndarray
    ):
        save_format = OnDiskFeatureDataFormat.NUMPY
        feat_item = {
            "domain" : OnDiskFeatureDataDomain.NODE,
            "type" : type,
            "name" : TIMESTAMP_FEATURE_NAME,
            "format" : save_format,
            "in_memory" : True,
            "data" : data,
        }
        self.graph_attributes.append(feat_item)
        return self

    def add_edge_feature(
        self,
        type : str,
        name : str,
        save_format : OnDiskFeatureDataFormat,
        data : np.ndarray,
        in_memory : bool = True,
        **extra_kwargs
    ):
        assert save_format == OnDiskFeatureDataFormat.NUMPY
        assert len(type.split(':')) == 3, "Edge type name must be a colon-spearated triplet."
        feat_item = {
            "domain" : OnDiskFeatureDataDomain.EDGE,
            "type" : type,
            "name" : name,
            "format" : save_format,
            "in_memory" : in_memory,
            "data" : data
        }
        feat_item.update(extra_kwargs)
        self.features.append(feat_item)
        return self

    def add_edge_timestamp(
        self,
        type : str,
        data : np.ndarray
    ):
        save_format = OnDiskFeatureDataFormat.NUMPY
        feat_item = {
            "domain" : OnDiskFeatureDataDomain.EDGE,
            "type" : type,
            "name" : TIMESTAMP_FEATURE_NAME,
            "format" : save_format,
            "in_memory" : True,
            "data" : data,
        }
        self.graph_attributes.append(feat_item)
        return self

    def add_task(self, task_creator : OnDiskTaskCreator):
        self.tasks.append(task_creator)
        return self

    def _validate(self):
        # Validate data integrity.
        ntypes = [ndict['type'] for ndict in self.nodes]
        etypes = [edict['type'] for edict in self.edges]

        # Check timestamp
        types_with_ts = set([
            gattr_dict['type']
            for gattr_dict in self.graph_attributes
            if gattr_dict['name'] == TIMESTAMP_FEATURE_NAME
        ])
        ntypes_has_ts = [nt in types_with_ts for nt in ntypes]
        etypes_has_ts = [et in types_with_ts for et in etypes]
        if any(ntypes_has_ts) and not all(ntypes_has_ts):
            raise ValueError(
                "Timestamps need to present for all node types if specified."
                f" But got {list(zip(ntypes, ntypes_has_ts))}."
            )
        if any(etypes_has_ts) and not all(etypes_has_ts):
            raise ValueError(
                "Timestamps need to present for all edge types if specified."
                f" But got {list(zip(etypes, etypes_has_ts))}."
            )

    def done(self, path : Path):
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if not path.is_dir():
            raise ValueError(f"Provided path {path} must be a directory.")

        self._validate()

        # Write edges.
        edges_dir = path / "edges"
        edges_dir.mkdir(parents=True, exist_ok=True)
        for edge_item in self.edges:
            edge_type, edge_data = edge_item["type"], edge_item["data"]
            type_str = "" if edge_type is None else f"{edge_type}_"
            if edge_item["format"] == OnDiskDatasetEdgesFormat.CSV:
                df = pd.DataFrame({"src" : edge_data[0], "dst" : edge_data[1]})
                data_path = edges_dir / f"{type_str}edges.csv"
                df = pd.DataFrame({"src" : edge_data[0], "dst" : edge_data[1]})
                df.to_csv(data_path, index=False, header=False)
            else:
                data_path = edges_dir / f"{type_str}edges.npy"
                np.save(data_path, edge_data)
            edge_item.pop("data")
            edge_item["path"] = str(data_path.relative_to(path))

        # Write features
        features_dir = path / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        for feat_item in self.features + self.graph_attributes:
            feat_type, feat_data, feat_name = feat_item["type"], feat_item["data"], feat_item["name"]
            type_str = "" if feat_type is None else f"{feat_type}_"
            data_path = features_dir / f"{type_str}{feat_name}.npy"
            np.save(data_path, feat_data, allow_pickle=True)
            feat_item.pop("data")
            feat_item["path"] = str(data_path.relative_to(path))

        # Write tasks
        task_meta = []
        for task_creator in self.tasks:
            task_meta.append(task_creator.done(path))

        # Write metadata
        metadata = OnDiskDatasetMeta.parse_obj({
            "dataset_name" : self.name,
            "graph" : {
                "nodes" : self.nodes,
                "edges" : self.edges,
                "feature_data": self.graph_attributes
            },
            "feature_data" : self.features,
            "tasks" : task_meta,
        })

        yaml_utils.save_pyd(metadata, path / "metadata.yaml")
