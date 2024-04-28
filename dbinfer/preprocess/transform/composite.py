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
import pydantic
import logging

from ...device import DeviceInfo
from .base import (
    RDBTransform,
    rdb_transform,
    get_column_transform_class,
    ColumnData,
    RDBData,
)
from .wrapper import RDBTransformWrapper

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class _NameAndConfig(pydantic.BaseModel):
    name : str
    config : Optional[Dict] = {}

class ColumnTransformChainConfig(pydantic.BaseModel):
    transforms : List[_NameAndConfig]

@rdb_transform
class ColumnTransformChain(RDBTransform):

    config_class = ColumnTransformChainConfig
    name = "column_transform_chain"

    def __init__(self, config : ColumnTransformChainConfig):
        super().__init__(config)
        self.transforms = [
            RDBTransformWrapper(xcfg.name, xcfg.config)
            for xcfg in config.transforms
        ]
        logger.debug([xform.col_xform_config for xform in self.transforms])

    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        self.fit_transform(rdb_data, device)

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        for xform in self.transforms:
            rdb_data = xform.transform(rdb_data, device)
        return rdb_data

    def fit_transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        for xform in self.transforms:
            xform.fit(rdb_data, device)
            rdb_data = xform.transform(rdb_data, device)
        return rdb_data
