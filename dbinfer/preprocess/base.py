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


from enum import Enum
from typing import Tuple, Dict, Optional, List
import abc
from pathlib import Path
import pydantic

from dbinfer_bench import (
    DBBColumnDType,
    DBBRDBDataset,
)

from ..device import DeviceInfo

class RDBDatasetPreprocess:

    config_class : pydantic.BaseModel = None
    name : str = "base"
    default_config = None

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def run(
        self,
        dataset : DBBRDBDataset,
        output_path : Path,
        device : DeviceInfo
    ):
        pass

_RDB_PREPROCESS_REGISTRY = {}

def rdb_preprocess(preprocess_class):
    global _RDB_PREPROCESS_REGISTRY
    _RDB_PREPROCESS_REGISTRY[preprocess_class.name] = preprocess_class
    return preprocess_class

def get_rdb_preprocess_class(name : str):
    global _RDB_PREPROCESS_REGISTRY
    preprocess_class = _RDB_PREPROCESS_REGISTRY.get(name, None)
    if preprocess_class is None:
        raise ValueError(f"Cannot find the preprocess class of name {name}.")
    return preprocess_class

def get_rdb_preprocess_choice():
    names = _RDB_PREPROCESS_REGISTRY.keys()
    return Enum("RDBPreprocessChoice", {name.upper() : name for name in names})
