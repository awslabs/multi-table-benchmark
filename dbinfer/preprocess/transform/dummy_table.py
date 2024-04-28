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

class HandleDummyTableConfig(pydantic.BaseModel):
    pass

@rdb_transform
class HandleDummyTable(RDBTransform):

    config_class = HandleDummyTableConfig
    name = "handle_dummy_table"

    def __init__(self, config : HandleDummyTableConfig):
        super().__init__(config)


    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        pass

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        if rdb_data.relationships is None:
            return rdb_data
        pk_to_fk_list = defaultdict(list)
        for fk_tbl, fk_col, pk_tbl, pk_col in rdb_data.relationships:
            pk_to_fk_list[(pk_tbl, pk_col)].append((fk_tbl, fk_col))
        dummy_tbl_to_pk = {}
        for (pk_tbl, pk_col), fk_list in pk_to_fk_list.items():
            if pk_tbl not in rdb_data.tables:
                if pk_tbl in dummy_tbl_to_pk:
                    other_pk_col = dummy_tbl_to_pk[pk_tbl]
                    raise ValueError(
                        f"Dummy table {pk_tbl} can only have one PK column but got"
                        f" '{pk_col}' and '{other_pk_col}'."
                    )
                logger.info(f"Add dummy table '{pk_tbl}' with PK column '{pk_col}'.")
                pk_data = np.unique(np.concatenate([
                    rdb_data.tables[fk_tbl][fk_col].data
                    for fk_tbl, fk_col in fk_list]))
                pk_meta = {'dtype' : 'primary_key'}
                rdb_data.tables[pk_tbl] = {pk_col : ColumnData(pk_meta, pk_data)}
                dummy_tbl_to_pk[pk_tbl] = pk_col
        return rdb_data
