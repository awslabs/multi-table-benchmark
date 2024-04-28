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

from dbinfer_bench import (
    DBBColumnDType,
    DBBRDBDataset,
    DBBRDBTask,
    DBBRDBTaskCreator,
    DBBRDBDatasetCreator,
    DBBTaskType,
)

from ..device import DeviceInfo
from .base import RDBDatasetPreprocess, rdb_preprocess
from .transform import (
    RDBData,
    ColumnData,
    get_rdb_transform_class,
    is_task_table,
    make_task_table_name,
    unmake_task_table_name,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class _NameAndConfig(pydantic.BaseModel):
    name : str
    config : Optional[Dict] = {}

class RDBTransformPreprocessConfig(pydantic.BaseModel):
    transforms : List[_NameAndConfig]

@rdb_preprocess
class RDBTransformPreprocess(RDBDatasetPreprocess):

    config_class = RDBTransformPreprocessConfig
    name : str = "transform"
    default_config = RDBTransformPreprocessConfig.parse_obj({
        "transforms" : [
            { "name" : "handle_dummy_table" },
            { "name" : "key_mapping" },
            {
                "name" : "column_transform_chain",
                "config" : {
                    "transforms": [
                        { "name" : "canonicalize_numeric" },
                        { "name" : "canonicalize_datetime" },
                        { "name" : "glove_text_embedding" },
                        {
                            "name" : "featurize_datetime",
                            "config" : {
                                "methods" : [
                                    "YEAR",
                                    "MONTH",
                                    "DAY",
                                    "DAYOFWEEK",
                                    "TIMESTAMP",
                                ],
                            },
                        },
                        { "name" : "norm_numeric" },
                        { "name" : "remap_category" },
                    ]
                },
            },
            { "name" : "filter_column" },
            { "name" : "fill_timestamp" },
        ],
    })

    def __init__(self, config : RDBTransformPreprocessConfig):
        super().__init__(config)
        self.transforms = []
        for xcfg in config.transforms:
            xform_class = get_rdb_transform_class(xcfg.name)
            xform_config = xform_class.config_class.parse_obj(xcfg.config)
            self.transforms.append(xform_class(xform_config))

    def run(
        self,
        dataset : DBBRDBDataset,
        output_path : Path,
        device : DeviceInfo
    ):
        logger.debug("Fit & transform RDB data.")
        rdb_data = self.extract_data(dataset)
        task_data_fit, task_data_transform = self.extract_task_data(dataset)
        all_data_fit = _merge_rdb_and_task(rdb_data, task_data_fit)
        for xform in self.transforms:
            all_data_fit = xform.fit_transform(all_data_fit, device)
        new_task_data_transform = {}
        for task_name, task_data in task_data_transform.items():
            logger.debug(f"Transform data of task {task_name}.")
            for xform in self.transforms:
                task_data = xform.transform(task_data, device)
            new_task_data_transform[task_name] = task_data
        new_rdb_data, new_task_data_fit = _split_rdb_and_task(all_data_fit)

        logger.debug(f"New RDB data:\n{new_rdb_data}")
        for task_name in new_task_data_fit:
            logger.debug(f"New task data ({task_name}):\n"
                         f"{new_task_data_fit[task_name]}\n"
                         f"{new_task_data_transform[task_name]}")

        new_name = dataset.metadata.dataset_name
        logger.debug(f"Generating new dataset {new_name}.")
        ctor = DBBRDBDatasetCreator(new_name)
        self.output_data(ctor, new_rdb_data)
        self.output_column_groups(ctor, new_rdb_data)
        self.output_tasks(ctor, new_task_data_fit, new_task_data_transform, dataset)
        ctor.done(output_path)

    def extract_data(self, dataset : DBBRDBDataset) -> RDBData:
        tables = {tbl_name : {} for tbl_name in dataset.tables}
        for tbl_schema in dataset.metadata.tables:
            tbl_name = tbl_schema.name
            for col_schema in tbl_schema.columns:
                col_name = col_schema.name
                col_meta = dict(col_schema)
                if col_name == tbl_schema.time_column:
                    col_meta['is_time_column'] = True
                tables[tbl_name][col_name] = ColumnData(
                    col_meta, dataset.tables[tbl_name][col_name])
        column_groups = None
        if dataset.metadata.column_groups is not None:
            column_groups = []
            for cg in dataset.metadata.column_groups:
                column_groups.append([(cid.table, cid.column) for cid in cg])
        relationships = None
        if dataset.metadata.relationships is not None:
            relationships = []
            for rel in dataset.metadata.relationships:
                relationships.append((
                    rel.fk.table, rel.fk.column, rel.pk.table, rel.pk.column))
        return RDBData(tables, column_groups, relationships)

    def extract_task_data(
        self,
        dataset : DBBRDBDataset,
    ) -> Tuple[RDBData, Dict[str, RDBData]]:
        fit_table = {}
        transform_tables = {
            task.metadata.name : {}
            for task in dataset.tasks
        }
        for task_id, task in enumerate(dataset.tasks):
            task_name = task.metadata.name
            target_table_name = task.metadata.target_table
            task_table_name = make_task_table_name(task_name, target_table_name)
            for tbl_schema in dataset.metadata.tables:
                if tbl_schema.name == target_table_name:
                    target_tbl_schema = tbl_schema
                    break

            fit_table[task_table_name] = {}
            transform_tables[task_name][task_table_name] = {}
            for col_schema in task.metadata.columns:
                col = col_schema.name
                if (task.metadata.task_type == DBBTaskType.retrieval
                    and col in [
                        task.metadata.key_prediction_label_column,
                        task.metadata.key_prediction_query_idx_column,
                    ]):
                    # Skip extra val/test columns needed by retrieval task
                    # because they have been correctly processed.
                    continue
                col_data = np.concatenate([
                    task.train_set[col],
                    task.validation_set[col],
                    task.test_set[col]
                ], axis=0)
                col_meta = dict(col_schema)
                if col == task.metadata.time_column:
                    col_meta['is_time_column'] = True
                if col in target_tbl_schema.columns:
                    # Columns also in the target table are drawn from the same
                    # data distribution, thus only requiring transform.
                    transform_tables[task_name][task_table_name][col] = \
                        ColumnData(col_meta, col_data)
                else:
                    # Task-specific data needs fit-and-transform
                    fit_table[task_table_name][col] = ColumnData(col_meta, col_data)
        task_data_fit = RDBData(fit_table)
        task_data_transform = {
            task_name : RDBData(task_table)
            for task_name, task_table in transform_tables.items()
        }
        return task_data_fit, task_data_transform

    def output_data(self, ctor : DBBRDBDatasetCreator, rdb: RDBData):
        for tbl_name, table in rdb.tables.items():
            ctor.add_table(tbl_name)
            time_col = None
            for col_name, col in table.items():
                metadata = col.metadata
                if metadata.get('is_time_column', False):
                    metadata.pop('is_time_column')
                    time_col = col_name
                ctor.add_column(tbl_name, col_name, col.data, **metadata)
            if time_col is not None:
                ctor.set_time_column(tbl_name, time_col)

    def output_column_groups(self, ctor : DBBRDBDatasetCreator, rdb: RDBData):
        if rdb.column_groups is not None:
            for cg in rdb.column_groups:
                ctor.add_column_group(cg)

    def output_tasks(
        self,
        ctor : DBBRDBDatasetCreator,
        task_data_fit : Dict[str, RDBData],
        task_data_transform : Dict[str, RDBData],
        ds: DBBRDBDataset
    ):
        for orig_task in ds.tasks:
            task_name = orig_task.metadata.name
            tgt_table_name = orig_task.metadata.target_table
            task_table_name = make_task_table_name(task_name, tgt_table_name)
            all_data = {
                **task_data_fit[task_name].tables[task_table_name],
                **task_data_transform[task_name].tables[task_table_name]
            }
            num_train, num_val, num_test = _get_num(orig_task)
            split_idx = [num_train, num_train + num_val]

            task_ctor = DBBRDBTaskCreator(task_name)
            task_ctor.copy_fields_from(orig_task.metadata)
            time_col = None
            for col_name, col in all_data.items():
                train_data, val_data, test_data = np.split(col.data, split_idx)
                metadata = col.metadata
                if 'name' in metadata:
                    assert col_name == metadata.pop('name')
                if metadata.get('is_time_column', False):
                    metadata.pop('is_time_column')
                    time_col = col_name
                task_ctor.add_task_data(col_name, train_data, val_data, test_data, **metadata)
            if time_col is not None:
                task_ctor.set_target_time_column(time_col)

            # Handle task extra meta.
            if orig_task.metadata.task_type == DBBTaskType.classification:
                label_col = all_data[orig_task.metadata.target_column].data
                num_classes = len(np.unique(label_col))
                task_ctor.add_task_field('num_classes', num_classes)
            elif orig_task.metadata.task_type == DBBTaskType.retrieval:
                # Copy extra columns needed by retrieval task.
                for col_name in [
                    orig_task.metadata.key_prediction_label_column,
                    orig_task.metadata.key_prediction_query_idx_column,
                ]:
                    task_ctor.add_task_data(
                        col_name,
                        None,
                        orig_task.validation_set[col_name],
                        orig_task.test_set[col_name],
                        dtype=None
                    )
            ctor.add_task(task_ctor)


def _merge_rdb_and_task(rdb_data : RDBData, task_data_fit : RDBData) -> RDBData:
    return RDBData(
        {**rdb_data.tables, **task_data_fit.tables},
        rdb_data.column_groups,
        rdb_data.relationships
    )

def _split_rdb_and_task(all_data_fit : RDBData) -> Tuple[RDBData, Dict[str, RDBData]]:
    rdb_data = RDBData({}, all_data_fit.column_groups, all_data_fit.relationships)
    task_data = {}
    for table_name, table in all_data_fit.tables.items():
        if is_task_table(table_name):
            task_name, _ = unmake_task_table_name(table_name)
            task_data[task_name] = RDBData({table_name : table})
        else:
            rdb_data.tables[table_name] = table
    return rdb_data, task_data

def _get_num(task : DBBRDBTask) -> Tuple[int, int, int]:
    key = list(task.train_set.keys())[0]
    return len(task.train_set[key]), len(task.validation_set[key]), len(task.test_set[key])
