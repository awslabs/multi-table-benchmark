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


from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
import pydantic
import logging
from collections import defaultdict
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from ... import datetime_utils
from .base import (
    RDBTransform,
    rdb_transform,
    ColumnData,
    RDBData,
    is_task_table,
    unmake_task_table_name,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class KeyMappingConfig(pydantic.BaseModel):
    pass

@rdb_transform
class KeyMapping(RDBTransform):
    """Encode primary foreign keys into integers.

    - Values appear in foreign key but not in primary key will be added to primary key set,
      which means the table with primary key will expand to include rows with null values.
    - NA keys will be mapped to -1.
    - Unseen keys encountered in `transform` will be converted to -1 too (the behavior could
      be changed in the future).
    """

    config_class = KeyMappingConfig
    name = "key_mapping"

    def __init__(self, config : KeyMappingConfig):
        super().__init__(config)
        

    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        self.pk_to_fk_list = defaultdict(list)
        self.fk_to_pk = {}
        for rel in rdb_data.relationships:
            fk_tbl, fk_col, pk_tbl, pk_col = rel
            self.pk_to_fk_list[(pk_tbl, pk_col)].append((fk_tbl, fk_col))
            self.fk_to_pk[(fk_tbl, fk_col)] = (pk_tbl, pk_col)
            self.fk_to_pk[(pk_tbl, pk_col)] = (pk_tbl, pk_col)

        # Search for primary keys in task table to add extra relationships.
        for tbl_name, tbl in rdb_data.tables.items():
            if not is_task_table(tbl_name):
                continue
            for col_name, col in tbl.items():
                if col.metadata['dtype'] == DBBColumnDType.primary_key:
                    _, target_table_name = unmake_task_table_name(tbl_name)
                    self.fk_to_pk[(tbl_name, col_name)] = (target_table_name, col_name)
                    self.pk_to_fk_list[(target_table_name, col_name)].append((tbl_name, col_name))
        
        # Remapping
        self.mappings = {}
        for (pk_tbl, pk_col), fk_list in self.pk_to_fk_list.items():
            logger.info(f"Fitting key_mapping for ({pk_tbl}, {pk_col}) and {fk_list}.")
            key_data = [rdb_data.tables[pk_tbl][pk_col].data]
            for (fk_tbl, fk_col) in fk_list:
                fk_data = rdb_data.tables[fk_tbl][fk_col].data
                key_data.append(fk_data)
            key_data = np.concatenate(key_data)
            _, categories = pd.factorize(key_data, use_na_sentinel=True)
            self.mappings[(pk_tbl, pk_col)] = categories

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        for tbl_name, tbl in rdb_data.tables.items():
            if is_task_table(tbl_name):
                for col_name, col in tbl.items():
                    if col.metadata['dtype'] in [
                        DBBColumnDType.primary_key,
                        DBBColumnDType.foreign_key
                    ]:
                        _, target_table = unmake_task_table_name(tbl_name)
                        pk = self.fk_to_pk[(target_table, col_name)]
                        col.data = self._map_col(col.data, self.mappings[pk])
                        col.metadata['capacity'] = len(self.mappings[pk])
            else:
                num_of_miss = 0
                for col_name, col in tbl.items():
                    if col.metadata['dtype'] == DBBColumnDType.primary_key:
                        if (tbl_name, col_name) not in self.mappings:
                            # dangling primary key
                            n = len(col.data)
                        else:
                            n = len(self.mappings[(tbl_name, col_name)])
                        num_of_miss = n - len(col.data)
                        # Add an extra value `n` for unseen value.
                        col.data = np.arange(n, dtype=np.int64)
                        col.metadata['capacity'] = n
                    elif col.metadata['dtype'] == DBBColumnDType.foreign_key:
                        pk = self.fk_to_pk[(tbl_name, col_name)]
                        col.data = self._map_col(col.data, self.mappings[pk])
                        col.metadata['capacity'] = len(self.mappings[pk])
                # If the primary key in this table is added with new values, then
                # all other columns in this table should be added with null values.
                if num_of_miss > 0:
                    logger.debug(f"Appending ({num_of_miss}) new row(s) to table {tbl_name}.")
                    for col_name, col in tbl.items():
                        if col.metadata['dtype'] == DBBColumnDType.primary_key:
                            continue
                        elif col.metadata['dtype'] == DBBColumnDType.foreign_key:
                            col.data = np.append(col.data, np.full(num_of_miss, -1))
                        else:
                            col.data = self._add_null(col, num_of_miss)

        return rdb_data

    def _map_col(
        self,
        data : np.ndarray,
        categories: np.ndarray,
    ) -> np.ndarray:
        new_data = pd.Categorical(data, categories=categories).codes.copy()
        new_data = new_data.astype('int64')
        return new_data

    def _add_null(
        self,
        column : ColumnData,
        num: int
    ) -> np.ndarray:
        null_val = self._get_null_val(column.metadata['dtype'])
        shape = (num,) + column.data.shape[1:]
        null_arr = np.full(shape, null_val)
        data = np.concatenate([column.data, null_arr], axis=0)
        return data

    def _get_null_val(self, dtype: DBBColumnDType):
        if dtype == DBBColumnDType.datetime_t:
            return np.datetime64('NaT')
        elif dtype == DBBColumnDType.text_t:
            return ""
        elif dtype == DBBColumnDType.timestamp_t:
            return datetime_utils.dt2ts(np.datetime64('NaT'))
        else:
            return np.nan
