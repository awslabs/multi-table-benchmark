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
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class CanonicalizeNumericConfig(pydantic.BaseModel):
    pass

@column_transform
class CanonicalizeNumericTransform(ColumnTransform):
    """Cast column data type to its canonical type."""
    config_class = CanonicalizeNumericConfig
    name = "canonicalize_numeric"
    input_dtype = DBBColumnDType.float_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        pass

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        column.data = column.data.astype('float32')
        return [column]

class CanonicalizeDatetimeConfig(pydantic.BaseModel):
    pass

@column_transform
class CanonicalizeDatetimeTransform(ColumnTransform):
    """Cast column data type to its canonical type."""
    config_class = CanonicalizeDatetimeConfig
    name = "canonicalize_datetime"
    input_dtype = DBBColumnDType.datetime_t
    output_dtypes = [DBBColumnDType.datetime_t]
    output_name_formatters : List[str] = ["{name}"]

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        pass

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        column.data = column.data.astype('datetime64[ns]')
        return [column]
