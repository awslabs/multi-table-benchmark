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


"""Solution base class."""
from enum import Enum
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import abc
import pydantic
import numpy as np
import dgl.graphbolt as gb
from dbinfer_bench import DBBGraphDataset, DBBRDBDataset

from ..device import DeviceInfo

__all__ = [
    'FitSummary',
    'GraphMLSolutionConfig',
    'GraphMLSolution',
    'SweepChoice',
    'gml_solution',
    'get_gml_solution_class',
    'get_gml_solution_choice',
    'TabularMLSolutionConfig',
    'TabularMLSolution',
    'tabml_solution',
    'get_tabml_solution_class',
    'get_tabml_solution_choice',
]

class FitSummary:

    def __init__(self):
        self.train_metric = None
        self.val_metric = None

class GraphMLSolutionConfig(pydantic.BaseModel):
    lr : float
    batch_size : int
    eval_batch_size : int
    feat_encode_size : Optional[int] = None
    fanouts : List[int]
    eval_fanouts : Optional[List[int]] = None
    negative_sampling_ratio : Optional[int] = 5
    patience : Optional[int] = 15
    epochs : Optional[int] = 200
    embed_ntypes : Optional[List[str]] = []
    enable_temporal_sampling : Optional[bool] = True
    time_budget : Optional[float] = 0  # Unit is second. 0 means unlimited budget.

class GraphMLSolution:

    config_class = GraphMLSolutionConfig
    name = "base_gml"

    @staticmethod
    @abc.abstractstaticmethod
    def create_from_dataset(
        config : pydantic.BaseModel,
        dataset : DBBGraphDataset,
    ):
        pass

    @abc.abstractmethod
    def fit(
        self,
        dataset : DBBGraphDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
    ) -> float:
        pass

    @abc.abstractmethod
    def checkpoint(self, ckpt_path):
        pass

    @abc.abstractmethod
    def load_from_checkpoint(self, ckpt_path):
        pass


class SweepChoice(str, Enum):
    random = "random"
    grid = "grid"
    bayes = "bayes"


_GML_SOLUTION_REGISTRY = {}

def gml_solution(solution_class):
    global _GML_SOLUTION_REGISTRY
    _GML_SOLUTION_REGISTRY[solution_class.name] = solution_class
    return solution_class

def get_gml_solution_class(name : str):
    global _GML_SOLUTION_REGISTRY
    solution_class = _GML_SOLUTION_REGISTRY.get(name, None)
    if solution_class is None:
        raise ValueError(f"Cannot find the solution class of name {name}.")
    return solution_class

def get_gml_solution_choice():
    """Get an enum class of all the available GML solutions."""
    names = _GML_SOLUTION_REGISTRY.keys()
    return Enum("GMLSolutionChoice", {name.upper() : name for name in names})


class TabularMLSolutionConfig(pydantic.BaseModel):
    lr : float
    batch_size : int
    eval_batch_size : int
    negative_sampling_ratio : Optional[int] = 5
    patience : Optional[int] = 15
    epochs : Optional[int] = 200
    embed_keys : Optional[List[str]] = []
    time_budget : Optional[float] = 0  # Unit is second. 0 means unlimited budget.

class TabularMLSolution:

    config_class = TabularMLSolutionConfig
    name = "base_table"

    @staticmethod
    @abc.abstractstaticmethod
    def create_from_dataset(
        config : pydantic.BaseModel,
        dataset : DBBRDBDataset,
    ):
        pass

    @abc.abstractmethod
    def fit(
        self,
        dataset : DBBRDBDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        table : Dict[str, np.ndarray],
        device : DeviceInfo,
    ) -> float:
        pass

    @abc.abstractmethod
    def checkpoint(self, ckpt_path):
        pass

    @abc.abstractmethod
    def load_from_checkpoint(self, ckpt_path):
        pass

_TabML_SOLUTION_REGISTRY = {}

def tabml_solution(solution_class):
    global _TabML_SOLUTION_REGISTRY
    _TabML_SOLUTION_REGISTRY[solution_class.name] = solution_class
    return solution_class

def get_tabml_solution_class(name : str):
    global _TabML_SOLUTION_REGISTRY
    solution_class = _TabML_SOLUTION_REGISTRY.get(name, None)
    if solution_class is None:
        raise ValueError(f"Cannot find the solution class of name {name}.")
    return solution_class

def get_tabml_solution_choice():
    """Get an enum class of all the available TabML solutions."""
    names = _TabML_SOLUTION_REGISTRY.keys()
    return Enum("TabMLSolutionChoice", {name.upper() : name for name in names})
