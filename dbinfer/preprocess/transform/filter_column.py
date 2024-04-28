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
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class FilterColumnConfig(pydantic.BaseModel):
    pass

@rdb_transform
class FilterColumn(RDBTransform):
    """Filter columns.

    Criterion:
    - Non-key columns with identical values.
    """

    config_class = FilterColumnConfig
    name = "filter_column"

    def __init__(self, config : FilterColumnConfig):
        super().__init__(config)

    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        self.columns_to_filer = defaultdict(list)
        for tbl_name, tbl in rdb_data.tables.items():
            for col_name, col in tbl.items():
                if col.metadata['dtype'] in [
                    DBBColumnDType.primary_key,
                    DBBColumnDType.foreign_key,
                    DBBColumnDType.text_t,
                ]:
                    continue
                if col.data.ndim > 1:
                    # Ignore vector embeddings.
                    continue
                logger.info(f"Fitting filter_column for {col_name}.")
                unique_data = pd.unique(col.data)
                if len(unique_data) <= 1:
                    self.columns_to_filer[tbl_name].append(col_name)

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        for tbl_name, tbl in rdb_data.tables.items():
            for col_name in self.columns_to_filer[tbl_name]:
                if col_name in tbl:
                    logger.info(f"Filter column {tbl_name}/{col_name}.")
                    tbl.pop(col_name)
        return rdb_data
