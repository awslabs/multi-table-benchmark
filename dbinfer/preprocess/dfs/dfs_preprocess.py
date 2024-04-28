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
from pathlib import Path
import pydantic
import logging
import numpy as np
import pandas as pd
import featuretools as ft
import tqdm
from dbinfer_bench import (
    DBBColumnDType,
    DBBRDBDataset,
    DBBRDBTask,
    DBBRDBTaskCreator,
    DBBRDBDatasetCreator,
    DBBTableSchema,
    DBBTaskType,
    DBBTaskMeta,
)

from ...device import DeviceInfo
from ..base import RDBDatasetPreprocess, rdb_preprocess
from . import core

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class DFSPreprocessConfig(pydantic.BaseModel):
    dfs : core.DFSConfig

@rdb_preprocess
class DFSPreprocess(RDBDatasetPreprocess):

    config_class = DFSPreprocessConfig
    name : str = "dfs"
    default_config = DFSPreprocessConfig(
        dfs=core.DFSConfig())

    def __init__(self, config : DFSPreprocessConfig):
        super().__init__(config)

    def run(
        self,
        dataset : DBBRDBDataset,
        output_path : Path,
        device : DeviceInfo
    ):
        new_task_df = {}
        new_features = {}
        for task_id, task in enumerate(dataset.tasks):
            logger.info(f"Running DFS for task {task.metadata.name} ...")
            dfs_engine_class = core.get_dfs_engine(self.config.dfs.engine)
            dfs_engine = dfs_engine_class(self.config.dfs, dataset, task_id)
            dfsed_df, dfs_features = dfs_engine.run()
            new_task_df[task.metadata.name] = dfsed_df
            new_features[task.metadata.name] = dfs_features
        self.output_dataset(new_task_df, new_features, dataset, output_path)

    def output_dataset(
        self,
        task_dataframes : Dict[str, pd.DataFrame],
        dfs_features : Dict[str, ft.FeatureBase],
        orig_dataset : DBBRDBDataset,
        output_path : Path
    ):
        table_schemas = {schema.name: schema for schema in orig_dataset.metadata.tables}
        new_name = orig_dataset.metadata.dataset_name + f'-dfs-{self.config.dfs.max_depth}'
        logger.debug(f"Generating new dataset {new_name}.")
        ctor = DBBRDBDatasetCreator(new_name)
        ctor.replace_tables_from(orig_dataset)
        for orig_task in orig_dataset.tasks:
            task_name = orig_task.metadata.name
            task_ctor = DBBRDBTaskCreator(task_name)
            task_ctor.copy_fields_from(orig_task.metadata)

            # Add features created by DFS.
            task_df = task_dataframes[task_name]
            num_train, num_val, num_test = _get_num(orig_task)
            split_idx = [num_train, num_train + num_val]
            for col_name in task_df:
                col_data = task_df[col_name].to_numpy()
                col_expr = dfs_features[task_name][col_name]
                try:
                    dtype = _infer_dtype(col_expr, table_schemas, orig_task.metadata)
                except KeyError:
                    logger.warning(
                        f"Cannot infer the data type of {col_expr} from table schema. "
                        f"Skipping..."
                    )
                    continue
                logger.debug(f"Casting feature {col_expr} to data type {dtype}...")
                col_data = _cast_dtype(col_data, dtype)
                train_data, val_data, test_data = np.split(col_data, split_idx)
                task_ctor.add_task_data(col_name, train_data, val_data, test_data, dtype)

            # Add original features.
            for col_meta in orig_task.metadata.columns:
                col_meta = dict(col_meta)
                col_name = col_meta.pop('name')
                task_ctor.add_task_data(
                    col_name,
                    orig_task.train_set[col_name],
                    orig_task.validation_set[col_name],
                    orig_task.test_set[col_name],
                    **col_meta
                )
            if orig_task.metadata.task_type == DBBTaskType.retrieval:
                for extra_col in [
                    orig_task.metadata.key_prediction_label_column,
                    orig_task.metadata.key_prediction_query_idx_column,
                ]:
                    task_ctor.add_task_data(
                        extra_col,
                        None,
                        orig_task.validation_set[extra_col],
                        orig_task.test_set[extra_col],
                        dtype=None
                    )
            ctor.add_task(task_ctor)

        ctor.done(output_path)

def _get_num(task : DBBRDBTask) -> Tuple[int, int, int]:
    key = list(task.train_set.keys())[0]
    return (len(task.train_set[key]),
            len(task.validation_set[key]),
            len(task.test_set[key]))


def _infer_dtype(
    col_expr : ft.FeatureBase,
    table_schemas : Dict[str, DBBTableSchema],
    task_meta : DBBTaskMeta,
) -> DBBColumnDType:
    if isinstance(col_expr, ft.DirectFeature):
        return _infer_dtype(col_expr.base_features[0], table_schemas, task_meta)
    elif isinstance(col_expr, ft.AggregationFeature):
        agg_func = col_expr.primitive.name
        if agg_func == 'count':
            return DBBColumnDType.float_t
        else:
            # All other aggregation functions do not change the logical types.
            return _infer_dtype(col_expr.base_features[0], table_schemas, task_meta)
    else:       # IdentityFeature
        dataframe_name = col_expr.dataframe_name
        col_name = col_expr.get_name()
        if dataframe_name == '__task__':
            # Task table
            return task_meta.column_dict[col_name].dtype
        else:
            return table_schemas[dataframe_name].column_dict[col_name].dtype


def _cast_dtype(
    col : np.ndarray,
    dtype : DBBColumnDType,
) -> np.ndarray:
    if dtype == DBBColumnDType.float_t:
        # The output values could be either NaN or scalar when the input is scalar,
        # or either NaN scalar or vector when the input is vector.
        # So we check:
        # * If all the entries are NaN, do nothing (and cast it to a normal float32 vector).
        # * Otherwise, check the first entry that is not NaN.
        #   * If the first such entry is a scalar, convert the column to a normal float32 vector.
        #   * Otherwise, fill the NaNs with zero vectors, and convert the thing into a matrix.
        col = pd.Series(col)
        if col.isnull().all():
            col = col.values.astype('float32')
        else:
            head = col.dropna().iloc[0]
            if not isinstance(head, (list, np.ndarray)):
                col = col.values.astype('float32')
            else:
                dim = len(head)
                # Cannot use fillna() as it does not accept ndarrays or lists as argument.
                col = col.apply(lambda x: np.asarray(x).astype('float32') if not np.isnan(x).all() else np.zeros(dim, dtype='float32'))
                col = np.asarray(col.values)
    elif dtype == DBBColumnDType.category_t:
        col = col.astype(object)
    elif dtype == DBBColumnDType.datetime_t:
        assert np.issubdtype(col.dtype, np.datetime64)
    elif dtype == DBBColumnDType.text_t:
        # Replace None, NaN etc with empty string.
        # Also use dtype=object to avoid storing fixed-length strings since text can
        # vary greatly in length.
        col = np.array(['' if not x else x for x in col], dtype=object)
    elif dtype == DBBColumnDType.timestamp_t:
        col = col.astype('int64')
    if isinstance(col[0],(np.ndarray,list)):
        col = np.stack(col)
    return col
