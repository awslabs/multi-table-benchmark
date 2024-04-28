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


"""Metadata used to create Tabular ML solutions.

Despite the similarity, the classes here are of different purposes than the metadata
classes in `dbinfer_bench`. The classes here are to unify the interface of different
ML solutions, making them more extensible against future changes.
"""
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
import pydantic
import copy

from dbinfer_bench import (
    DBBColumnSchema,
    DBBColumnDType,
    DBBTaskMeta,
    DBBRDBDataset,
)

__all__ = [
    'FeatureConfig',
    'TabularTaskConfig',
    'TabularDatasetConfig',
    'parse_config_from_tabular_dataset',
]

class FeatureConfig(pydantic.BaseModel):

    # Feature data type. In general, only the following types are expected.
    #  - Numeric
    #  - Categorical
    # All other data types should have been preprocessed into one of the above
    # during data preprocessing.
    dtype : DBBColumnDType

    # Whether the feature is a special time feature used by sampling. The feature
    # will be *excluded* from node features. To include it, turn on the
    # corresponding option in data preprocessing.
    is_time : bool

    # Extra fields about this feature.
    #
    # For numeric features,
    #  - in_size : The size of the feature dimension.
    #
    # For categorical features,
    #  - num_categories : The total number of categories.
    extra_fields : Dict[str, Any]

TabularTaskConfig = DBBTaskMeta

class TabularDatasetConfig(pydantic.BaseModel):
    features : Dict[str, FeatureConfig]
    task : TabularTaskConfig

def _col_schema_to_feat_cfg(
    schema : DBBColumnSchema,
    is_time : bool
) -> FeatureConfig:
    schema = dict(schema)
    dtype = schema.pop('dtype')
    return FeatureConfig(dtype=dtype, is_time=is_time, extra_fields=schema)

def parse_config_from_tabular_dataset(
    dataset : DBBRDBDataset,
    task_name : str
) -> TabularDatasetConfig:
    task_meta = dataset.get_task(task_name).metadata
    # Parse feature config.
    feat_cfg = {
        schema.name : _col_schema_to_feat_cfg(
            schema, is_time=(schema.name == task_meta.time_column))
        for schema in task_meta.columns
    }
    return TabularDatasetConfig(
        features=feat_cfg,
        task=task_meta
    )
