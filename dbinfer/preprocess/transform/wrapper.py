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
from collections import defaultdict
import numpy as np
import pydantic
import logging
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    RDBTransform,
    get_column_transform_class,
    ColumnData,
    RDBData,
    is_task_table,
    unmake_task_table_name,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class RDBTransformWrapper(RDBTransform):

    def __init__(
        self,
        col_xform_name : str,
        col_xform_config : Dict
    ):
        self.col_xform_class = get_column_transform_class(col_xform_name)
        self.col_xform_config = self.col_xform_class.config_class.parse_obj(col_xform_config)
        self.col_groups = []
        self.col_xforms = []

    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        if self.col_xform_class.input_dtype == DBBColumnDType.text_t:
            self.col_groups = self.get_text_column_groups(rdb_data)
        else:
            self.col_groups = self.get_column_groups(rdb_data)

        if len(self.col_groups) == 0:
            return

        # Create & fit transforms.
        for cg in self.col_groups:
            logger.info(f"Fitting transform {self.col_xform_class.name} for column group {cg}.")
            col_xform = self.col_xform_class(self.col_xform_config)
            cg_meta = _get_cg_meta(cg, rdb_data)
            cg_data = [rdb_data.tables[tbl_name][col_name].data for tbl_name, col_name in cg]
            cg_data = np.concatenate(cg_data, axis=0)
            col_xform.fit(ColumnData(cg_meta, cg_data), device)
            self.col_xforms.append(col_xform)

    def get_text_column_groups(self, rdb_data : RDBData):
        text_cg = []
        for tbl_name, table in rdb_data.tables.items():
            for col_name, col in table.items():
                if col.metadata['dtype'] == DBBColumnDType.text_t:
                    text_cg.append((tbl_name, col_name))
        if len(text_cg) == 0:
            return []
        else:
            return [text_cg]

    def get_column_groups(self, rdb_data : RDBData):
        # Get column group.
        col_groups = []
        scanned = set()
        # Add column groups explicitly defined.
        if rdb_data.column_groups is not None:
            for cg in rdb_data.column_groups:
                cg_meta = _get_cg_meta(cg, rdb_data)
                if cg_meta['dtype'] == self.col_xform_class.input_dtype:
                    col_groups.append(cg)
                    for tbl_name, col_name in cg:
                        scanned.add((tbl_name, col_name))
        # Scan rest of the columns.
        for tbl_name, table in rdb_data.tables.items():
            for col_name, col in table.items():
                if (tbl_name, col_name) in scanned:
                    continue
                if col.metadata['dtype'] == self.col_xform_class.input_dtype:
                    col_groups.append([(tbl_name, col_name)])
        return col_groups

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        to_xform = {}
        for cg, xform in zip(self.col_groups, self.col_xforms):
            for tbl_name, col_name in cg:
                to_xform[(tbl_name, col_name)] = xform
        new_tables = {tbl_name : {} for tbl_name in rdb_data.tables}
        col_name_mapping = defaultdict(list)
        for tbl_name, table in rdb_data.tables.items():
            for col_name, col in table.items():
                # Get transform.
                xform = to_xform.get((tbl_name, col_name), None)
                if xform is None and is_task_table(tbl_name):
                    _, target_table = unmake_task_table_name(tbl_name)
                    xform = to_xform.get((target_table, col_name), None)
                # Conduct transformation.
                if xform is not None:
                    logger.info(f"Transforming table/column {tbl_name}/{col_name}.")
                    new_cols = xform.transform(col, device)
                    assert len(new_cols) == len(xform.output_name_formatters)
                    for formatter, new_col in zip(xform.output_name_formatters, new_cols):
                        if len(new_col.data) != len(col.data):
                            raise ValueError(
                                "Column transform is not allowed to change the length of columns."
                                f" {len(col.data)} -> {len(new_col.data)}."
                            )
                        new_col_name = formatter.format(name=col_name)
                        new_tables[tbl_name][new_col_name] = new_col
                        col_name_mapping[(tbl_name, col_name)].append((tbl_name, new_col_name))
                else:
                    new_tables[tbl_name][col_name] = col
                    col_name_mapping[(tbl_name, col_name)].append((tbl_name, col_name))
        col_name_mapping = dict(col_name_mapping)
        # Re-create column groups.
        column_groups = None
        if rdb_data.column_groups is not None:
            column_groups = []
            for cg in rdb_data.column_groups:
                mapped_cg = [col_name_mapping[name_pair] for name_pair in cg]
                for i in range(len(mapped_cg[0])):
                    column_groups.append([mapped_cg[j][i] for j in range(len(mapped_cg))])
        return RDBData(new_tables, column_groups, rdb_data.relationships)

def _get_cg_meta(cg, rdb_data):
    tbl_name, col_name = cg[0]
    return rdb_data.tables[tbl_name][col_name].metadata
