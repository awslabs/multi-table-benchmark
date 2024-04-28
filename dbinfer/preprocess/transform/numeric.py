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


import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import logging
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

class NormNumericTransformConfig(pydantic.BaseModel):
    impute_strategy : Optional[str] = "median"
    skew_threshold : Optional[float] = 0.99

@column_transform
class NormNumericTransform(ColumnTransform):
    config_class = NormNumericTransformConfig
    name = "norm_numeric"
    input_dtype = DBBColumnDType.float_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : NormNumericTransformConfig):
        super().__init__(config)

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : 1 if column.data.ndim == 1 else column.data.shape[1]
        }

        if column.data.ndim > 1:
            # Ignore vector embeddings.
            return

        skew_score = pd.Series(column.data).skew()
        if np.abs(skew_score) > self.config.skew_threshold:
            logger.debug(f'\tSkew values detected. Skew score: {skew_score:.6f}.')
            self.transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.config.impute_strategy)),
                ('scaler', QuantileTransformer(output_distribution='normal'))])
        else:
            self.transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.config.impute_strategy)),
                ('scaler', StandardScaler())])
        self.transformer.fit(column.data.reshape(-1, 1))

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        if column.data.ndim > 1:
            new_data = column.data.astype('float32')
        else:
            new_data = self.transformer.transform(column.data.reshape(-1, 1)).reshape(-1)
            new_data = new_data.astype('float32')
        return [ColumnData(self.new_meta, new_data)]
