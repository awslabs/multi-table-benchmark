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
from typing import Tuple, Dict, Optional, List
import abc
import pydantic
import numpy as np
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo

@dataclass
class ColumnData:
    metadata : Dict
    data : np.ndarray

    def __repr__(self):
        return f"Column(dtype: {str(self.metadata['dtype'])}, len: {len(self.data)})"

@dataclass
class RDBData:
    tables : Dict[str, Dict[str, ColumnData]]
    column_groups : Optional[List[List[Tuple[str, str]]]] = None
    relationships : Optional[List[Tuple[str, str, str, str]]] = None

    def __repr__(self):
        ret = "{\n"
        for table_name, table in self.tables.items():
            ret += f"  Table(\n"
            ret += f"    name={table_name}\n"
            ret +=  "    columns={\n"
            for col_name, col in table.items():
                ret += f"      {col_name}: {col}\n"
            ret +=  "    })\n"
        ret += f"  column_groups: {self.column_groups}\n"
        ret += f"  relationships: {self.relationships}\n"
        ret += "}"
        return ret

def is_task_table(table_name : str) -> bool:
    return table_name.startswith('__task__:')

def make_task_table_name(
    task_name : str,
    target_table_name : str
) -> str:
    return f'__task__:{task_name}:{target_table_name}'

def unmake_task_table_name(
    task_table_name : str
) -> Tuple[str, str]:
    parts = task_table_name.split(':')
    assert len(parts) == 3
    return parts[1], parts[2]

class RDBTransform:

    config_class : pydantic.BaseModel = None
    name : str = "base"

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        pass

    @abc.abstractmethod
    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        pass

    def fit_transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        self.fit(rdb_data, device)
        return self.transform(rdb_data, device)

_RDB_TRANSFORM_REGISTRY = {}
def rdb_transform(transform_class):
    global _RDB_TRANSFORM_REGISTRY
    _RDB_TRANSFORM_REGISTRY[transform_class.name] = transform_class
    return transform_class

def get_rdb_transform_class(name : str):
    global _RDB_TRANSFORM_REGISTRY
    transform_class =  _RDB_TRANSFORM_REGISTRY.get(name, None)
    if transform_class is None:
        raise RuntimeError(f"Cannot find RDB transform class {name}.")
    return transform_class

class ColumnTransform:

    # Config class.
    config_class : pydantic.BaseModel = None
    # Name of this transform.
    name : str = "base"
    # Input column data type.
    input_dtype : DBBColumnDType = None
    # Output column data type.
    output_dtypes : List[DBBColumnDType] = None
    # String formatters to generate new column names.
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        pass

    @abc.abstractmethod
    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        pass

_COLUMN_TRANSFORM_REGISTRY = {}
def column_transform(transform_class):
    global _COLUMN_TRANSFORM_REGISTRY
    _COLUMN_TRANSFORM_REGISTRY[transform_class.name] = transform_class
    return transform_class

def get_column_transform_class(name : str):
    global _COLUMN_TRANSFORM_REGISTRY
    transform_class =  _COLUMN_TRANSFORM_REGISTRY.get(name, None)
    if transform_class is None:
        raise RuntimeError(f"Cannot find column transform class {name}.")
    return transform_class
