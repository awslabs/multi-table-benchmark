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
from dbinfer_bench import DBBColumnDType, TIMESTAMP_FEATURE_NAME

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

class FillTimestampConfig(pydantic.BaseModel):
    pass

@rdb_transform
class FillTimestamp(RDBTransform):

    config_class = FillTimestampConfig
    name = "fill_timestamp"

    def __init__(self, config : FillTimestampConfig):
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
        for tbl_name, tbl in rdb_data.tables.items():
            if is_task_table(tbl_name):
                continue
            has_ts = any([col.metadata.get('is_time_column', False) for col in tbl.values()])
            if not has_ts:
                logger.info(f"Fill default timestamp for table {tbl_name}.")
                assert TIMESTAMP_FEATURE_NAME not in tbl
                tbl_size = len(list(tbl.values())[0].data)
                tbl[TIMESTAMP_FEATURE_NAME] = ColumnData(
                    {'dtype' : DBBColumnDType.timestamp_t, 'is_time_column' : True},
                    np.zeros((tbl_size,), dtype=np.int64)
                )
        return rdb_data
